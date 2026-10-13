# =============================================================================
# DATA PROTOCOL: FIXED TRAIN / VALIDATION / TEST SPLIT AND TEMPERATURE SCALING
#
# Takes the cases produced by data/generate.py and writes ONE file,
# dataset.npz, that every model (DeepONet, FNO, POD-DeepONet) reads.
# Because the split and the scaling are frozen in this file, all models are
# trained and tested on exactly the same cases with exactly the same numbers.
#
# 1. SPLIT, stratified by family (circle / rectangle):
#      each family is split 80 / 10 / 10 on its own, so every split has the
#      same circle : rectangle ratio.
#      500 combined cases -> 250 circles  -> 200 / 25 / 25
#                            250 rects    -> 200 / 25 / 25
#                            total           400 / 50 / 50
#
# 2. SCALING (min-max), from TRAINING solid points only:
#      T_min, T_max = smallest and largest T over solid points of train cases
#      temp = (T - T_min) / (T_max - T_min) * mask
#    Validation and test data are never used to choose the scale (no
#    information leaks from the test set). Their scaled values may therefore
#    lie slightly outside [0, 1]. Cavity points are set to 0 by the mask.
#
# 3. FINGERPRINT: the SHA-256 of dataset.npz is written to
#    dataset_manifest.json; later scripts check it to be sure they all use
#    the same file.
#
# This module does not use PyTorch.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   labels          : family of each case (0 = circle, 1 = rectangle)
#   train_fraction  : share of each family used for training     (0.8)
#   val_fraction    : share used for validation                  (0.1)
#   seed            : random seed of the split (cfg.split_seed = 123)
#   train, val, test: case indices of the three splits
#   sdf             : scaled SDF per case, shape (n_cases, 12288)
#   mask            : 1 solid / 0 cavity, shape (n_cases, 12288)
#   physical_temp   : temperature in kelvin, 0 in cavity, (n_cases, 12288)
#   temp            : scaled temperature, the training target
#   meta            : per case [shape_code, p1, p2, cx, cy]
#   fe_paths        : path of each case's fe_case_XXXXXX.npz
#   coords          : grid points, shape (12288, 2)
#   T_min, T_max    : scaling constants                          [K]
# =============================================================================

import json
import hashlib
from pathlib import Path, PureWindowsPath

import numpy as np


# =============================================================================
# SHA-256 FINGERPRINT OF A FILE
#
# The file is read in 1 MB pieces so large files never fill the memory.
# Any change to the file, even one byte, gives a completely different hash.
# =============================================================================
def sha256_file(path):
    hasher = hashlib.sha256()
    piece_size = 1024 * 1024

    with open(path, "rb") as f:
        while True:
            piece = f.read(piece_size)
            if len(piece) == 0:
                break
            hasher.update(piece)

    return hasher.hexdigest()


# =============================================================================
# STRATIFIED SPLIT
#
# For every family separately:
#   shuffle its case indices, then cut  [ first 80% | next 10% | rest ]
#   n_train = int(n * 0.8),  n_val = int(n * 0.1),  n_test = the remainder
# The three lists are finally shuffled once more so families are mixed.
# Same seed -> same split, every time.
# =============================================================================
def stratified_indices(labels, train_fraction=0.8, val_fraction=0.1, seed=123):
    fractions_ok = (
        train_fraction > 0
        and val_fraction > 0
        and train_fraction + val_fraction < 1
    )
    if not fractions_ok:
        raise ValueError("Invalid split fractions")

    rng = np.random.default_rng(seed)

    train = []
    val = []
    test = []

    for family in np.unique(labels):
        # INDICES OF THIS FAMILY, IN RANDOM ORDER
        family_ids = np.flatnonzero(labels == family)
        family_ids = rng.permutation(family_ids)

        n = len(family_ids)
        n_train = int(n * train_fraction)
        n_val = int(n * val_fraction)
        n_test = n - n_train - n_val

        if min(n_train, n_val, n_test) < 1:
            raise ValueError(
                "Each family needs enough cases for all three splits (at least 10 at 80/10/10)."
            )

        train.extend(family_ids[:n_train].tolist())
        val.extend(family_ids[n_train:n_train + n_val].tolist())
        test.extend(family_ids[n_train + n_val:].tolist())

    # MIX THE FAMILIES INSIDE EACH SPLIT
    train = rng.permutation(np.asarray(train, dtype=np.int64))
    val = rng.permutation(np.asarray(val, dtype=np.int64))
    test = rng.permutation(np.asarray(test, dtype=np.int64))

    return train, val, test


# =============================================================================
# WRITE dataset.npz AND dataset_manifest.json
#
# extra : optional dictionary of additional arrays stored in dataset.npz
#         (e.g. the generation_id from generate_data.py)
# =============================================================================
def finalize_dataset(cfg, sdf, mask, physical_temp, meta, fe_paths, coords, extra=None):
    sdf = np.asarray(sdf, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    physical_temp = np.asarray(physical_temp, dtype=np.float32)
    meta = np.asarray(meta, dtype=np.float32)

    # SANITY CHECK: NO NaN / inf IN THE SDF OR IN SOLID TEMPERATURES
    solid = mask > 0.5
    if not np.isfinite(sdf).all() or not np.isfinite(physical_temp[solid]).all():
        raise ValueError("Dataset contains invalid solid temperatures or SDF values")

    # STEP 1: SPLIT
    shape_codes = meta[:, 0]
    train, val, test = stratified_indices(
        shape_codes, cfg.train_fraction, cfg.val_fraction, cfg.split_seed
    )

    # STEP 2: SCALE FROM TRAINING SOLID POINTS ONLY
    train_temp = physical_temp[train]
    train_solid = mask[train] > 0.5
    train_values = train_temp[train_solid]

    T_min = float(train_values.min())
    T_max = float(train_values.max())
    if T_max <= T_min:
        raise ValueError("Degenerate temperature range")

    temp = (physical_temp - T_min) / (T_max - T_min) * mask

    # EVERYTHING THAT GOES INTO dataset.npz
    fields = {}
    fields["sdf"] = sdf
    fields["mask"] = mask
    fields["temp"] = temp
    fields["physical_temp"] = physical_temp
    fields["meta"] = meta
    fields["fe_paths"] = np.asarray(fe_paths, dtype=str)
    fields["coords"] = np.asarray(coords, dtype=np.float32)
    fields["T_min"] = np.array([T_min], dtype=np.float64)
    fields["T_max"] = np.array([T_max], dtype=np.float64)
    fields["train_idx"] = train
    fields["val_idx"] = val
    fields["test_idx"] = test

    # SETTINGS USED TO MAKE THE DATA (FOR LATER CHECKS)
    fields["Nx"] = np.array([cfg.Nx])
    fields["Ny"] = np.array([cfg.Ny])
    fields["Lx"] = np.array([cfg.Lx])
    fields["Ly"] = np.array([cfg.Ly])
    fields["maxh"] = np.array([cfg.maxh])
    fields["order"] = np.array([cfg.order])
    fields["kappa"] = np.array([cfg.kappa])
    fields["h_hole"] = np.array([cfg.h_hole])
    fields["T_inf"] = np.array([cfg.T_inf])
    fields["q_top"] = np.array([cfg.q_top])
    fields["dataset_family"] = np.array([cfg.dataset_family])
    fields["n_cases"] = np.array([len(sdf)])
    fields["protocol_version"] = np.array([2])
    fields["split_seed"] = np.array([cfg.split_seed])

    if extra is not None:
        for key in extra:
            fields[key] = extra[key]

    # SAFE WRITE: TEMPORARY NAME FIRST, RENAME WHEN COMPLETE
    path = Path(cfg.dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **fields)
    temporary.replace(path)

    # MANIFEST: HUMAN-READABLE SUMMARY + FINGERPRINT
    family_counts = {}
    for code in np.unique(meta[:, 0]):
        family_counts[str(int(code))] = int((meta[:, 0] == code).sum())

    manifest = {
        "protocol_version": 2,
        "dataset_sha256": sha256_file(path),
        "temperature_scale_source": "training solid points only",
        "T_min": T_min,
        "T_max": T_max,
        "family_counts": family_counts,
        "splits": {
            "train": train.tolist(),
            "validation": val.tolist(),
            "test": test.tolist(),
        },
    }

    manifest_path = path.with_name("dataset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("Saved protocol-v2 dataset: %s" % path, flush=True)
    return str(path)


# =============================================================================
# LEGACY CONVERSION (run.py stage "prepare-legacy")
#
# Older runs saved data/processed/<case_tag>/dataset.npz without the frozen
# split. This re-uses those FE results (no new solves) and passes them
# through finalize_dataset, so they follow the same protocol as new data.
# =============================================================================
def find_legacy_fe_file(cfg, stored_path):
    from .config import PROJECT_ROOT

    candidates = []
    candidates.append(PROJECT_ROOT / Path(stored_path))
    candidates.append(Path(stored_path))
    candidates.append(
        cfg._root(cfg.data_root) / "raw" / cfg.case_tag / "fe_cases"
        / PureWindowsPath(stored_path).name
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError("Missing original FE file: %s" % stored_path)


def prepare_legacy(cfg):
    if Path(cfg.dataset_path).exists():
        raise FileExistsError("Prepared dataset already exists. Reuse it without reconversion.")

    old_path = cfg._root(cfg.data_root) / "processed" / cfg.case_tag / cfg.dataset_name
    if not old_path.exists():
        raise FileNotFoundError("No legacy dataset at %s; run generate instead" % old_path)

    old = {}
    with np.load(old_path, allow_pickle=True) as saved:
        for key in saved.files:
            old[key] = saved[key]

    # GRID / MESH SETTINGS FOLLOW THE OLD FILE
    for key in ("Nx", "Ny", "Lx", "Ly", "maxh"):
        if key in old:
            setattr(cfg, key, old[key][0].item())

    # PHYSICAL TEMPERATURE: STORED DIRECTLY, OR UNDONE FROM THE OLD SCALING
    if "physical_temp" in old:
        physical = old["physical_temp"]
    else:
        old_min = float(old["T_min"][0])
        old_max = float(old["T_max"][0])
        physical = old["temp"] * (old_max - old_min) + old_min
        physical = np.where(old["mask"] > 0.5, physical, 0).astype(np.float32)

    fe_paths = []
    for stored_path in old["fe_paths"]:
        found = find_legacy_fe_file(cfg, str(stored_path))
        fe_paths.append(cfg.relative_project_path(found))

    extra = {"legacy_source_sha256": np.array([sha256_file(old_path)])}

    return finalize_dataset(cfg, old["sdf"], old["mask"], physical, old["meta"],
                            fe_paths, old["coords"], extra)
