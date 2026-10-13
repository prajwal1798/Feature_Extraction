# =============================================================================
# DATA-SIZE STUDY: ONE DATASET FOR EVERY TRAINING-SET SIZE
#
# Question: how does the test error fall as the number of training cases N
# grows? Expected behaviour: a power law,  error ~ C N^(-alpha), i.e. a
# straight line of slope -alpha on a log-log plot.
#
# Design (all cases solved ONCE):
#   training pool : n_pool cases from the HALTON sequence, stored in Halton
#                   order and alternating circle / rectangle.
#                   Training set N = the first N cases of the pool, so the
#                   sets are nested (200 is inside 400 is inside 800 ...)
#                   and every one of them is evenly spread.
#   validation    : scaling_val_cases  cases, Latin hypercube   } the SAME for
#   test          : scaling_test_cases cases, Latin hypercube   } every N
#
# Why Halton and not Latin hypercube for the pool: a Latin hypercube is
# evenly spread only as a whole; its first N points are just a random
# sample. The first N points of a Halton sequence are evenly spread for
# every N.
#
# The temperature scale (T_min, T_max) is taken from the whole training pool
# (never from validation or test), so it is the same for every N.
# =============================================================================

# =============================================================================
# VARIABLES (listed once; see also data/generate.py)
#
#   n_pool       : training-pool size (cfg.selected_dataset_size, e.g. 3200)
#   n_val, n_test: validation and test sizes (cfg.scaling_val_cases / _test_)
#   group        : one block of candidates, e.g. ("train", "circle")
#   quota        : number of accepted cases wanted in a group
#   candidates   : parameter list of a group, tried in order; a candidate whose
#                  mesh or solve fails is skipped and the next one is used
#   records      : accepted cases per group, in acceptance order
# =============================================================================

import json
import hashlib
from pathlib import Path

import numpy as np

from ..fem.sampling import make_cartesian_grid, sample_case
from .protocol import finalize_dataset
from .generate import (latin_hypercube_sampling, circle_parameters, rect_parameters,
                       make_record, save_record, fe_case_path)

HALTON_BASES = [2, 3, 5, 7]
GROUPS = [("train", "circle"), ("train", "rect"),
          ("val", "circle"), ("val", "rect"),
          ("test", "circle"), ("test", "rect")]


# =============================================================================
# HALTON SEQUENCE
#
# Radical inverse of the integer i in base b: write i in base b and mirror its
# digits behind the decimal point.
#     i = 6 = 110 (base 2)  ->  0.011 (base 2) = 0/2 + 1/4 + 1/8 = 0.375
# Dimension j of the Halton point i uses the j-th prime as base (2, 3, 5, 7).
# Index 0 would give the corner (0, 0, ...); the sequence starts at i = 1.
# =============================================================================
def radical_inverse(index, base):
    value = 0.0
    fraction = 1.0 / base

    while index > 0:
        digit = index % base
        value = value + digit * fraction
        index = index // base
        fraction = fraction / base

    return value


def halton_sequence(n_points, n_dim, start=1):
    points = np.zeros((n_points, n_dim), dtype=np.float64)

    for i in range(n_points):
        for j in range(n_dim):
            points[i, j] = radical_inverse(start + i, HALTON_BASES[j])

    return points


# =============================================================================
# QUOTAS AND CANDIDATE LISTS OF THE SIX GROUPS
#
# Training candidates: Halton, with 25% extra points at the end as reserve.
# Validation / test candidates: a Latin hypercube of exactly the quota, plus
# a separate reserve hypercube (different seeds for every group).
# =============================================================================
def scaling_quotas(cfg):
    n_pool = cfg.selected_dataset_size
    quotas = {
        ("train", "circle"): n_pool // 2,
        ("train", "rect"): n_pool - n_pool // 2,
        ("val", "circle"): cfg.scaling_val_cases // 2,
        ("val", "rect"): cfg.scaling_val_cases - cfg.scaling_val_cases // 2,
        ("test", "circle"): cfg.scaling_test_cases // 2,
        ("test", "rect"): cfg.scaling_test_cases - cfg.scaling_test_cases // 2,
    }
    return quotas


def scaling_candidates(cfg, quotas):
    candidates = {}

    for role, shape in GROUPS:
        quota = quotas[(role, shape)]
        n_reserve = max(20, quota // 4)

        if role == "train":
            n_dim = 3 if shape == "circle" else 4
            z = halton_sequence(quota + n_reserve, n_dim)
        else:
            n_dim = 3 if shape == "circle" else 4
            seed = cfg.seed + 2000 + 100 * GROUPS.index((role, shape))
            z_main = latin_hypercube_sampling(quota, n_dim, seed=seed)
            z_reserve = latin_hypercube_sampling(n_reserve, n_dim, seed=seed + 50)
            z = np.vstack([z_main, z_reserve])

        if shape == "circle":
            params = []
            for D, cx, cy in circle_parameters(cfg, z):
                params.append(("circle", D, 0.0, cx, cy))
        else:
            params = []
            for W, H, cx, cy in rect_parameters(cfg, z):
                params.append(("rect", W, H, cx, cy))

        candidates[(role, shape)] = params

    return candidates


# =============================================================================
# FINGERPRINT OF THE SETTINGS OF THIS DESIGN (marks every saved case)
# =============================================================================
def scaling_generation_id(cfg):
    keys = ("dataset_family", "selected_dataset_size", "scaling_val_cases",
            "scaling_test_cases", "seed", "Nx", "Ny", "Lx", "Ly", "maxh", "order",
            "kappa", "h_hole", "T_inf", "q_top",
            "D_min", "D_max", "W_min", "W_max", "H_min", "H_max")
    settings = {"design": "halton_train_lhs_val_test_v1"}
    for key in keys:
        settings[key] = getattr(cfg, key)

    text = json.dumps(settings, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


# =============================================================================
# RESTART: CASES ALREADY SAVED, SORTED INTO THEIR GROUPS
# Returns accepted records per group, the next candidate to try per group,
# and the number of files on disk.
# =============================================================================
def load_saved_scaling_cases(cfg, generation_id):
    accepted = {}
    next_candidate = {}
    for group in GROUPS:
        accepted[group] = []
        next_candidate[group] = 0

    n_saved = 0
    if cfg.force_generation:
        return accepted, next_candidate, n_saved

    while True:
        path = fe_case_path(cfg, n_saved)
        if not path.exists():
            break

        record = {}
        with np.load(path, allow_pickle=False) as saved:
            for key in saved.files:
                record[key] = saved[key]

        if str(record["generation_id"][0]) != generation_id:
            raise ValueError("FE cache belongs to a different configuration; use --force")

        role = str(record["role"][0])
        shape = "circle" if int(record["shape_code"][0]) == 0 else "rect"
        accepted[(role, shape)].append((record, cfg.relative_project_path(path)))
        next_candidate[(role, shape)] = int(record["candidate_id"][0]) + 1
        n_saved = n_saved + 1

    return accepted, next_candidate, n_saved


# =============================================================================
# MAIN ROUTINE: SOLVE ALL CASES, THEN WRITE dataset.npz WITH FIXED SPLITS
# =============================================================================
def generate_scaling_dataset(cfg):
    if cfg.dataset_family != "scaling":
        raise ValueError("Use --family scaling for the data-size study")
    cfg.ensure_directories()

    quotas = scaling_quotas(cfg)
    candidates = scaling_candidates(cfg, quotas)
    generation_id = scaling_generation_id(cfg)

    accepted, next_candidate, n_saved = load_saved_scaling_cases(cfg, generation_id)
    total = 0
    for group in GROUPS:
        total = total + quotas[group]

    # SOLVE GROUP BY GROUP UNTIL EVERY QUOTA IS FILLED
    for group in GROUPS:
        role, shape = group
        while len(accepted[group]) < quotas[group]:
            i = next_candidate[group]
            if i >= len(candidates[group]):
                raise RuntimeError("Not enough valid candidates in group %s / %s" % group)
            next_candidate[group] = i + 1

            shape_name, p1, p2, cx, cy = candidates[group][i]
            print("[%d/%d] %s %s: p1=%.6f, p2=%.6f, cx=%.6f, cy=%.6f"
                  % (n_saved + 1, total, role, shape, p1, p2, cx, cy), flush=True)

            try:
                _, _, sdf, mask, T, pts, triangles, T_fe = sample_case(cfg, shape_name, p1, p2, cx, cy)
                if not np.isfinite(T[mask > 0.5]).all() or not np.isfinite(T_fe).all():
                    raise ValueError("Nonfinite solid FE samples")
            except Exception as error:
                print("Rejected candidate %d of %s / %s: %s" % (i, role, shape, error), flush=True)
                continue

            record = make_record(cfg, shape_name, p1, p2, cx, cy, sdf, mask, T, pts, triangles,
                                 T_fe, generation_id, i)
            record["role"] = np.array([role])

            path = fe_case_path(cfg, n_saved)
            save_record(path, record)
            accepted[group].append((record, cfg.relative_project_path(path)))
            n_saved = n_saved + 1

    # ORDER: TRAINING POOL ALTERNATING CIRCLE / RECTANGLE, THEN VALIDATION, TEST
    ordered = []
    train_circles = accepted[("train", "circle")]
    train_rects = accepted[("train", "rect")]
    for k in range(max(len(train_circles), len(train_rects))):
        if k < len(train_circles):
            ordered.append(train_circles[k])
        if k < len(train_rects):
            ordered.append(train_rects[k])
    n_train = len(ordered)

    for group in [("val", "circle"), ("val", "rect")]:
        ordered.extend(accepted[group])
    n_val = len(ordered) - n_train

    for group in [("test", "circle"), ("test", "rect")]:
        ordered.extend(accepted[group])
    n_test = len(ordered) - n_train - n_val

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)

    # COLLECT THE ARRAYS
    sdf_list = []
    mask_list = []
    temperature_list = []
    meta = []
    fe_paths = []
    for record, path in ordered:
        sdf_list.append(record["sdf"])
        mask_list.append(record["mask"])
        temperature_list.append(record["grid_temperature"])
        meta.append([record["shape_code"][0], record["p1"][0], record["p2"][0],
                     record["cx"][0], record["cy"][0]])
        fe_paths.append(path)

    xgrid, ygrid, Xg, Yg, grid_points = make_cartesian_grid(cfg)
    extra = {
        "generation_id": np.array([generation_id]),
        "design": np.array(["halton_train_lhs_val_test_v1"]),
        "nested_train": np.array([1]),
    }

    print("Training pool %d, validation %d, test %d cases" % (n_train, n_val, n_test), flush=True)
    return finalize_dataset(cfg, sdf_list, mask_list, temperature_list, meta, fe_paths,
                            grid_points, extra, splits=(train_idx, val_idx, test_idx))
