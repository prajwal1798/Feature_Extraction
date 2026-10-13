# =============================================================================
# DATASET: dataset.npz -> PYTORCH BATCHES
#
# dataset.npz (written by data/protocol.py) holds, for every case i:
#     sdf[i]  : scaled SDF on the grid, flattened to Ny*Nx = 12288 values
#     mask[i] : 1 = solid point, 0 = cavity point
#     temp[i] : scaled temperature (0 in the cavity)
# and the frozen split train_idx / val_idx / test_idx.
#
# A batch is the tuple (sdf, mask, temp), each of shape (batch, 12288).
# How the SDF is turned into each operator's input (branch/trunk for
# DeepONet, a 3-channel image for FNO) is done in training/runtime.py.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   data        : dictionary of all arrays stored in dataset.npz
#   ids         : case indices of one split
#   batch_size  : cases per batch (cfg.batch_size)
#   generator   : random generator that fixes the shuffling order of the
#                 training batches (seeded with cfg.train_seed)
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CASES OF ONE SPLIT, HELD AS FLOAT32 TENSORS
# =============================================================================
class FieldDataset(Dataset):
    def __init__(self, sdf_cases, mask_cases, temp_cases):
        self.sdf_cases = torch.as_tensor(sdf_cases, dtype=torch.float32)
        self.mask_cases = torch.as_tensor(mask_cases, dtype=torch.float32)
        self.temp_cases = torch.as_tensor(temp_cases, dtype=torch.float32)

    def __len__(self):
        return len(self.sdf_cases)

    def __getitem__(self, i):
        return self.sdf_cases[i], self.mask_cases[i], self.temp_cases[i]


# =============================================================================
# READ dataset.npz AND CHECK IT
#
# Refuses old (pre-protocol-2) files and splits that overlap or miss cases.
# The grid size is copied from the file into cfg, so every later routine
# uses the grid the data was actually made on.
# =============================================================================
def load_bundle(cfg):
    data = {}
    with np.load(cfg.dataset_path, allow_pickle=False) as saved:
        for key in saved.files:
            data[key] = saved[key]

    version = int(data.get("protocol_version", [0])[0])
    if version != 2:
        raise ValueError("Legacy dataset: run prepare-legacy before training")

    # EVERY CASE MUST APPEAR IN EXACTLY ONE SPLIT
    all_ids = np.concatenate([data["train_idx"], data["val_idx"], data["test_idx"]])
    n_cases = len(data["sdf"])
    if not np.array_equal(np.sort(all_ids), np.arange(n_cases)):
        raise ValueError("Corrupt or overlapping saved splits")

    for key in ("Nx", "Ny", "Lx", "Ly"):
        setattr(cfg, key, data[key][0].item())

    return data


# =============================================================================
# ONE DATALOADER PER SPLIT
#
# Only the training loader shuffles; its order is fixed by the seeded
# generator, so a run with the same seed sees the same batches.
# Returns: train_loader, val_loader, test_loader
# =============================================================================
def make_loaders(data, cfg):
    generator = torch.Generator().manual_seed(cfg.train_seed)
    pin = cfg.device == "cuda"

    loaders = []
    for split in ("train_idx", "val_idx", "test_idx"):
        ids = data[split]
        dataset = FieldDataset(data["sdf"][ids], data["mask"][ids], data["temp"][ids])

        if split == "train_idx":
            loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                                generator=generator, pin_memory=pin, num_workers=0)
        else:
            loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                                pin_memory=pin, num_workers=0)
        loaders.append(loader)

    return loaders[0], loaders[1], loaders[2]
