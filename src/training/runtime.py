# =============================================================================
# RUNTIME: DEVICE, SEEDS, OPERATOR INPUTS, ONE EPOCH, INFERENCE, CHECKPOINTS
#
# Everything here is shared by training (trainer.py), evaluation
# (analysis/metrics.py) and timing (analysis/benchmark.py), so the three
# always feed the networks in exactly the same way.
#
# INPUT SCALING (fixed from the TRAINING cases, stored in the checkpoint):
#     phi_hat = (phi - sdf_mean) / sdf_std          one mean / std for all points
#
# OPERATOR INPUTS built from phi_hat (shape (B, Ny*Nx)):
#     DeepONet branch : phi_hat                                  (B, 12288)
#     DeepONet trunk  : [x/Lx, y/Ly, phi_hat(x, y)] per point    (B, Q, 3)
#     FNO             : channels [phi_hat, x/Lx, y/Ly]           (B, 3, Ny, Nx)
#
# LOSS: masked mean squared error of the scaled temperature over the solid
# points of the whole batch,
#     L = sum_i sum_j m_ij (T^pred_ij - T_ij)^2  /  sum_i sum_j m_ij
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   coordinates   : grid points scaled to [0,1]^2, tensor (Ny*Nx, 2)
#   normalization : {"T_min", "T_max", "sdf_mean", "sdf_std"}
#   point_chunk   : DeepONet trunk points evaluated per forward/backward piece
#   valid_points  : number of solid points in the batch (sum of the mask)
# =============================================================================

import os
import random
from pathlib import Path

import numpy as np
import torch

from ..data.protocol import sha256_file
from ..models.build import model_from_checkpoint


# =============================================================================
# DEVICE AND REPRODUCIBILITY
#
# FP32 everywhere, TF32 disabled, deterministic algorithms required, so a
# run repeated with the same seed on the same machine gives the same numbers.
# CUDA is required unless the CPU is asked for explicitly (no silent fallback).
# =============================================================================
def configure_device(cfg):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch; no CPU fallback.")
    if cfg.precision != "fp32":
        raise ValueError("This comparison uses FP32 for both operators.")

    device = torch.device(cfg.device)
    torch.set_num_threads(cfg.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if device.type == "cuda":
        test = torch.ones((16, 16), device=device)
        (test @ test).sum().item()
        print("Using CUDA:", torch.cuda.get_device_name(0), flush=True)
    else:
        print("Using CPU (explicit request)", flush=True)

    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# OPERATOR INPUTS
# =============================================================================
def normalize_trunk_coordinates(coords, Lx, Ly):
    coordinates = np.array(coords, dtype=np.float32, copy=True)
    coordinates[:, 0] = coordinates[:, 0] / Lx
    coordinates[:, 1] = coordinates[:, 1] / Ly
    return coordinates


def build_case_dependent_trunk_input(coordinates, sdf):
    # (Q, 2) coordinates and (B, Q) SDF  ->  (B, Q, 3) = [x/Lx, y/Ly, phi_hat]
    batch_size = sdf.shape[0]
    coordinates = coordinates.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat((coordinates, sdf.unsqueeze(-1)), dim=-1)


def fno_field(sdf, coordinates, architecture):
    # (B, Ny*Nx) SDF and (Ny*Nx, 2) coordinates  ->  (B, 3, Ny, Nx) image
    ny = architecture["grid_ny"]
    nx = architecture["grid_nx"]
    batch_size = len(sdf)

    coordinate_channels = coordinates.T.reshape(2, ny, nx)
    coordinate_channels = coordinate_channels.unsqueeze(0).expand(batch_size, -1, -1, -1)
    sdf_channel = sdf.reshape(batch_size, 1, ny, nx)

    return torch.cat((sdf_channel, coordinate_channels), dim=1)


def scale_sdf(sdf, normalization):
    return (sdf - normalization["sdf_mean"]) / normalization["sdf_std"]


# =============================================================================
# ONE TRAINING EPOCH
#
# DeepONet: the branch is evaluated once per batch; the trunk and the loss
# are evaluated on chunks of point_chunk points, and backward() is called
# per chunk. Each chunk's loss is divided by the number of solid points of
# the WHOLE batch, so the gradients of all chunks add up to exactly the
# gradient of the full-batch loss L. One optimizer step per batch.
# Returns the point-weighted training MSE of the epoch.
# =============================================================================
def train_epoch(model, loader, coordinates, optimizer, cfg, normalization, device):
    model.train()
    total_squared_error = torch.zeros((), device=device)
    total_valid_points = torch.zeros((), device=device)

    for sdf, mask, temperature in loader:
        sdf = sdf.to(device, non_blocking=True)
        sdf = scale_sdf(sdf, normalization)
        mask = mask.to(device, non_blocking=True)
        temperature = temperature.to(device, non_blocking=True)
        valid_points = mask.sum().clamp_min(1)

        optimizer.zero_grad(set_to_none=True)

        if model.model_kind == "fno":
            field = fno_field(sdf, coordinates, model.architecture)
            prediction = model(field).flatten(1)
            squared_error = ((prediction - temperature) ** 2 * mask).sum()
            loss = squared_error / valid_points
            loss.backward()
            total_squared_error += squared_error.detach()

        else:
            branch_features = model.branch(sdf)
            n_points = sdf.shape[1]

            for first in range(0, n_points, cfg.point_chunk):
                last = min(first + cfg.point_chunk, n_points)

                trunk_input = build_case_dependent_trunk_input(coordinates[first:last],
                                                               sdf[:, first:last])
                prediction = model.forward_from_branch(branch_features, trunk_input)

                error = prediction - temperature[:, first:last]
                squared_error = (error ** 2 * mask[:, first:last]).sum()
                loss = squared_error / valid_points

                # KEEP THE BRANCH GRAPH UNTIL THE LAST CHUNK
                loss.backward(retain_graph=(last < n_points))
                total_squared_error += squared_error.detach()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                           error_if_nonfinite=True)
        optimizer.step()
        total_valid_points += valid_points

    loss_value = float((total_squared_error / total_valid_points).item())
    if not np.isfinite(loss_value):
        raise FloatingPointError("Nonfinite training loss")
    return loss_value


# =============================================================================
# PREDICTION OF THE SCALED TEMPERATURE FOR A BATCH OF RAW SDF FIELDS
# Returns a tensor of shape (B, Ny*Nx).
# =============================================================================
@torch.inference_mode()
def predict_tensor(model, raw_sdf, coordinates, normalization, chunk=2048):
    model.eval()
    sdf = scale_sdf(raw_sdf, normalization)

    if model.model_kind == "fno":
        field = fno_field(sdf, coordinates, model.architecture)
        return model(field).flatten(1)

    branch_features = model.branch(sdf)
    pieces = []
    for first in range(0, sdf.shape[1], chunk):
        last = min(first + chunk, sdf.shape[1])
        trunk_input = build_case_dependent_trunk_input(coordinates[first:last], sdf[:, first:last])
        pieces.append(model.forward_from_branch(branch_features, trunk_input))

    return torch.cat(pieces, dim=1)


# =============================================================================
# VALIDATION LOSS (same masked MSE as training, no gradients)
# =============================================================================
@torch.inference_mode()
def validate(model, loader, coordinates, cfg, normalization, device):
    model.eval()
    total_squared_error = torch.zeros((), device=device)
    total_valid_points = torch.zeros((), device=device)

    for sdf, mask, temperature in loader:
        sdf = sdf.to(device, non_blocking=True)
        prediction = predict_tensor(model, sdf, coordinates, normalization, cfg.point_chunk)
        mask = mask.to(device, non_blocking=True)
        temperature = temperature.to(device, non_blocking=True)

        total_squared_error += ((prediction - temperature) ** 2 * mask).sum()
        total_valid_points += mask.sum()

    loss_value = float((total_squared_error / total_valid_points).item())
    if not np.isfinite(loss_value):
        raise FloatingPointError("Nonfinite validation loss")
    return loss_value


# =============================================================================
# CHECKPOINTS
# =============================================================================
def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


# =============================================================================
# LOAD THE BEST CHECKPOINT OF cfg.run_name AND CHECK IT AGAINST THE DATA
#
# The checkpoint must belong to this dataset (SHA-256), grid, temperature
# scale and split; otherwise the comparison would be meaningless.
# external = True : rounded-rectangle set, checked against the training
#                   dataset it was scaled with.
# Returns: model (eval mode), checkpoint dictionary, coordinates tensor
# =============================================================================
def load_checkpoint(cfg, data, device, external=False):
    checkpoint = torch.load(cfg.model_path, map_location=device, weights_only=True)

    if checkpoint.get("code_protocol") != "classic_v1":
        raise ValueError("Use a checkpoint produced by this package; preserve old runs separately.")
    if checkpoint["model_kind"] != cfg.operator:
        raise ValueError("Checkpoint operator differs from requested operator")

    if external:
        dataset_hash = str(data["source_training_sha256"][0])
    else:
        dataset_hash = sha256_file(cfg.dataset_path)
    if checkpoint["dataset_sha256"] != dataset_hash:
        raise ValueError("Checkpoint and dataset do not match")

    for key in ("Nx", "Ny", "Lx", "Ly"):
        if not np.isclose(data[key][0], checkpoint["grid"][key], rtol=0, atol=1e-10):
            raise ValueError("Grid mismatch: " + key)
    for key in ("T_min", "T_max"):
        if not np.isclose(data[key][0], checkpoint["normalization"][key], rtol=0, atol=1e-6):
            raise ValueError("Temperature scaling mismatch: " + key)
    if not external:
        for key in ("val_idx", "test_idx"):
            if not np.array_equal(data[key], checkpoint["splits"][key]):
                raise ValueError("Split mismatch: " + key)

        # THE RUN MAY HAVE USED ONLY THE FIRST N TRAINING CASES (train_size)
        trained_on = np.asarray(checkpoint["splits"]["train_idx"])
        if not np.array_equal(data["train_idx"][:len(trained_on)], trained_on):
            raise ValueError("Split mismatch: train_idx")

    model = model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    coordinates = normalize_trunk_coordinates(data["coords"], checkpoint["grid"]["Lx"],
                                              checkpoint["grid"]["Ly"])
    coordinates = torch.as_tensor(coordinates, device=device)

    return model, checkpoint, coordinates
