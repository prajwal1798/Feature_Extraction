# =============================================================================
# TRAINER: FIT ONE DEEPONET OR FNO ON THE TRAINING SPLIT
#
#   1. load dataset.npz and the frozen split
#   2. build the model from cfg (named variant or custom architecture)
#   3. Adam + ReduceLROnPlateau on the validation loss, gradient clipping
#   4. keep the weights with the lowest validation loss  (best_model.pt)
#      stop early after early_stop_patience epochs without improvement
#   5. write loss_history.csv, epoch_timing.csv, training_summary.json
#   6. score the best weights on the VALIDATION split (val metrics)
#
# The test split is never touched here: model selection uses validation
# numbers only. A recovery checkpoint (last_checkpoint.pt) every
# checkpoint_every epochs allows an interrupted run to resume (--resume)
# with the same optimizer, scheduler and random-number state.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   data            : arrays of dataset.npz (see data/dataset.py)
#   normalization   : {"T_min","T_max","sdf_mean","sdf_std"} from training cases
#   history         : rows [epoch, train MSE, val MSE, learning rate]
#   epoch_timing    : rows [epoch, train s, val s, cumulative s, val MSE]
#   best_loss       : lowest validation MSE so far (scaled temperature)
#   best_epoch      : epoch at which best_loss was reached
#   temperature_range : T_max - T_min; RMSE in K = sqrt(MSE) * temperature_range
# =============================================================================

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..data.protocol import sha256_file
from ..data.dataset import load_bundle, make_loaders
from ..models.build import apply_variant, build_model, parameter_count
from ..analysis.metrics import evaluate
from .runtime import (configure_device, seed_everything, normalize_trunk_coordinates,
                      train_epoch, validate, atomic_torch_save)

# SETTINGS THAT MUST NOT CHANGE WHEN A RUN IS RESUMED
RESUME_SETTINGS = ("lr", "batch_size", "precision", "train_seed", "point_chunk", "grad_clip",
                   "early_stop_patience", "scheduler_patience", "scheduler_factor",
                   "checkpoint_every")


# =============================================================================
# INPUT SCALING FROM THE TRAINING CASES ONLY
# =============================================================================
def training_normalization(data):
    training_sdf = data["sdf"][data["train_idx"]]
    normalization = {
        "T_min": float(data["T_min"][0]),
        "T_max": float(data["T_max"][0]),
        "sdf_mean": float(training_sdf.mean()),
        "sdf_std": max(float(training_sdf.std()), 1e-8),
    }
    return normalization


# =============================================================================
# DEEPONET OUTPUT BIAS STARTS AT THE MEAN TRAINING TEMPERATURE (solid points)
# =============================================================================
def initialise_deeponet_bias(model, data):
    temperature = data["temp"][data["train_idx"]]
    mask = data["mask"][data["train_idx"]]
    mean_temperature = (temperature * mask).sum() / mask.sum()
    with torch.no_grad():
        model.bias.fill_(float(mean_temperature))


# =============================================================================
# EVERYTHING NEEDED TO REBUILD, CHECK OR RESUME THE RUN
# =============================================================================
def make_checkpoint(cfg, state, model, optimizer, scheduler, train_loader, device, epoch):
    if device.type == "cuda":
        cuda_rng = torch.cuda.get_rng_state_all()
    else:
        cuda_rng = []

    splits = {}
    for key in ("train_idx", "val_idx", "test_idx"):
        splits[key] = state["data"][key].tolist()

    grid = {}
    for key in ("Nx", "Ny", "Lx", "Ly"):
        grid[key] = getattr(cfg, key)

    return {
        "code_protocol": "classic_v1",
        "protocol_version": 2,
        "model_kind": cfg.operator,
        "architecture": model.architecture,
        "model_state": model.state_dict(),
        "normalization": state["normalization"],
        "dataset_sha256": state["dataset_hash"],
        "splits": splits,
        "grid": grid,
        "settings": state["settings"],
        "epoch": epoch,
        "best_epoch": state["best_epoch"],
        "best_val_loss": state["best_loss"],
        "history": state["history"],
        "epoch_timing": state["epoch_timing"],
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": cuda_rng,
        "loader_rng": train_loader.generator.get_state(),
        "pod_metadata": None,  # field kept so checkpoints match the original package
    }


# =============================================================================
# RESUME FROM last_checkpoint.pt (or best_model.pt if that is all there is)
# =============================================================================
def resume_state(cfg, state, model, optimizer, scheduler, train_loader, device):
    recovery_file = Path(cfg.checkpoint_outdir) / "last_checkpoint.pt"
    if not recovery_file.exists():
        recovery_file = Path(cfg.model_path)

    saved = torch.load(recovery_file, map_location=device, weights_only=True)

    if saved.get("code_protocol") != "classic_v1":
        raise ValueError("Old-package checkpoints cannot resume this study")
    if saved["dataset_sha256"] != state["dataset_hash"] or saved["architecture"] != model.architecture:
        raise ValueError("Resume dataset or architecture mismatch")
    for setting in RESUME_SETTINGS:
        if saved["settings"][setting] != getattr(cfg, setting):
            raise ValueError("Resume setting mismatch: " + setting)
    if saved["settings"].get("train_size", 0) != cfg.train_size:
        raise ValueError("Resume setting mismatch: train_size")

    model.load_state_dict(saved["model_state"])
    optimizer.load_state_dict(saved["optimizer_state"])
    scheduler.load_state_dict(saved["scheduler_state"])
    torch.set_rng_state(saved["torch_rng"].cpu())
    if device.type == "cuda" and saved["cuda_rng"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
    train_loader.generator.set_state(saved["loader_rng"].cpu())

    state["start_epoch"] = saved["epoch"] + 1
    state["best_loss"] = saved["best_val_loss"]
    state["best_epoch"] = saved["best_epoch"]
    state["history"] = saved["history"]
    state["epoch_timing"] = saved.get("epoch_timing", [])
    state["normalization"] = saved["normalization"]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# =============================================================================
# LOG FILES OF A FINISHED RUN
# =============================================================================
def write_logs(cfg, state, model):
    log_dir = Path(cfg.log_outdir)
    temperature_range = state["normalization"]["T_max"] - state["normalization"]["T_min"]

    with open(log_dir / "loss_history.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_point_weighted_mse", "val_point_weighted_mse", "lr"])
        writer.writerows(state["history"])

    with open(log_dir / "epoch_timing.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "training_s", "validation_s", "elapsed_training_s",
                         "val_point_weighted_mse"])
        writer.writerows(state["epoch_timing"])

    timed_epochs = [row[0] for row in state["epoch_timing"]]
    history_epochs = [row[0] for row in state["history"]]

    summary = {
        "run_name": cfg.run_name,
        "operator": cfg.operator,
        "variant": cfg.variant,
        "seed": cfg.train_seed,
        "train_cases": int(len(state["data"]["train_idx"])),
        "best_epoch": state["best_epoch"],
        "epochs_completed": state["history"][-1][0],
        "best_val_loss": state["best_loss"],
        "best_val_rmse_K": float(np.sqrt(state["best_loss"]) * temperature_range),
        "parameters": parameter_count(model),
        "dataset_sha256": state["dataset_hash"],
        "test_used_for_selection": False,
        "timing_scope": "Synchronized training plus validation; excludes checkpointing, setup and interruptions",
        "timing_covers_full_history": timed_epochs == history_epochs,
        "architecture": model.architecture,
        "normalization": state["normalization"],
        "settings": state["settings"],
    }
    (log_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# =============================================================================
# MAIN ROUTINE
# =============================================================================
def train_model(cfg, score_validation=True):
    apply_variant(cfg)
    cfg.ensure_directories()
    device = configure_device(cfg)
    seed_everything(cfg.train_seed)

    # STEP 1: DATA
    data = load_bundle(cfg)
    train_loader, val_loader, test_loader = make_loaders(data, cfg)

    state = {
        "data": data,
        "normalization": training_normalization(data),
        "dataset_hash": sha256_file(cfg.dataset_path),
        "settings": asdict(cfg),
        "start_epoch": 1,
        "best_loss": float("inf"),
        "best_epoch": 0,
        "history": [],
        "epoch_timing": [],
    }

    # STEP 2: MODEL
    model = build_model(cfg, cfg.Ny, cfg.Nx)
    if cfg.operator == "deeponet":
        initialise_deeponet_bias(model, data)
    model = model.to(device)

    # STEP 3: OPTIMISER AND SCHEDULER
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)

    coordinates = normalize_trunk_coordinates(data["coords"], cfg.Lx, cfg.Ly)
    coordinates = torch.as_tensor(coordinates, device=device)
    recovery_path = Path(cfg.checkpoint_outdir) / "last_checkpoint.pt"

    if Path(cfg.model_path).exists() and not cfg.resume:
        raise FileExistsError("Run exists: use --resume or choose a new --run-name")
    if cfg.resume:
        resume_state(cfg, state, model, optimizer, scheduler, train_loader, device)

    print("%s: %s; seed %d; real scalar parameters %s"
          % (cfg.operator, cfg.variant, cfg.train_seed, format(parameter_count(model), ",")),
          flush=True)

    temperature_range = state["normalization"]["T_max"] - state["normalization"]["T_min"]

    # STEP 4: EPOCH LOOP
    for epoch in range(state["start_epoch"], cfg.epochs + 1):
        # A RESUMED RUN THAT HAD ALREADY STOPPED EARLY DOES NOT CONTINUE
        if state["history"] and state["history"][-1][0] - state["best_epoch"] >= cfg.early_stop_patience:
            break

        synchronize(device)
        start = time.perf_counter()
        training_loss = train_epoch(model, train_loader, coordinates, optimizer, cfg,
                                    state["normalization"], device)
        synchronize(device)
        training_seconds = time.perf_counter() - start

        start = time.perf_counter()
        validation_loss = validate(model, val_loader, coordinates, cfg, state["normalization"], device)
        synchronize(device)
        validation_seconds = time.perf_counter() - start

        if state["epoch_timing"]:
            elapsed = state["epoch_timing"][-1][3]
        else:
            elapsed = 0.0
        elapsed = elapsed + training_seconds + validation_seconds
        state["epoch_timing"].append([epoch, training_seconds, validation_seconds, elapsed,
                                      validation_loss])

        scheduler.step(validation_loss)
        state["history"].append([epoch, training_loss, validation_loss,
                                 optimizer.param_groups[0]["lr"]])

        # NEW BEST VALIDATION LOSS -> SAVE THESE WEIGHTS
        if validation_loss < state["best_loss"]:
            state["best_loss"] = validation_loss
            state["best_epoch"] = epoch
            atomic_torch_save(make_checkpoint(cfg, state, model, optimizer, scheduler,
                                              train_loader, device, epoch), cfg.model_path)

        stop = epoch - state["best_epoch"] >= cfg.early_stop_patience
        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs or stop:
            atomic_torch_save(make_checkpoint(cfg, state, model, optimizer, scheduler,
                                              train_loader, device, epoch), recovery_path)

        if epoch == 1 or epoch % 20 == 0:
            rmse = np.sqrt(validation_loss) * temperature_range
            print("Epoch %5d | training MSE %.6e | validation MSE %.6e | RMSE %.4f K"
                  % (epoch, training_loss, validation_loss, rmse), flush=True)

        if stop:
            break

    if not state["history"]:
        raise ValueError("No training epochs completed")

    # STEP 5: LOGS
    summary = write_logs(cfg, state, model)
    print("Best validation RMSE: %.4f K (epoch %d). Test set unused."
          % (summary["best_val_rmse_K"], summary["best_epoch"]), flush=True)

    # STEP 6: VALIDATION METRICS OF THE BEST WEIGHTS (used for model selection)
    if score_validation:
        evaluate(cfg, split="val")

    return summary
