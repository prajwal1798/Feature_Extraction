# =============================================================================
# DATA GENERATION: BUILD THE WHOLE DATASET OF FE CASES
#
# For a dataset of n_cases cavities this script:
#   1. draws cavity sizes and positions by Latin hypercube sampling (LHS)
#   2. for every cavity: mesh, solve, sample on the grid  (fem/sampling.py)
#   3. saves every accepted case as  fe_cases/fe_case_XXXXXX.npz
#   4. shuffles the cases and hands them to protocol.finalize_dataset,
#      which splits train/val/test, normalises and writes dataset.npz
#
# Dataset families:
#   "circle"   : all cases are circles
#   "rect"     : all cases are rectangles
#   "combined" : half circles, half rectangles
#
# Restart: cases already saved with the SAME settings are re-loaded instead of
# being solved again, so an interrupted run continues where it stopped.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   cfg              : Settings object (src/config.py)
#   n_samples, n_dim : number of LHS samples and number of parameters
#   z                : LHS sample in [0, 1)^n_dim, one row per candidate
#   D                : circle diameter, D_min .. D_max      (8 .. 14 mm)
#   W, H             : rectangle size,  W_min .. W_max      (8 .. 14 mm)
#                                       H_min .. H_max      (6 .. 12 mm)
#   cx, cy           : cavity centre                        [m]
#   margin           : minimum gap between cavity and plate edge [m]
#   plan             : list of candidates (shape, p1, p2, cx, cy)
#   quotas           : how many cases of each shape are wanted
#   counts           : how many cases of each shape are accepted so far
#   candidate_id     : position in plan of the next candidate to try
#   generation_id    : fingerprint of the settings (marks the saved files)
#   record           : dictionary of arrays for one accepted case
#   records          : list of (record, file path) for all accepted cases
#   meta             : per case [shape_code, p1, p2, cx, cy]
#                      shape_code: 0 = circle, 1 = rectangle
# =============================================================================

import os
import json
import hashlib
from pathlib import Path

import numpy as np

from ..fem.sampling import make_cartesian_grid, sample_case
from ..fem.solver import export_vtk
from ..analysis.plots import plot_preview
from .protocol import finalize_dataset


# =============================================================================
# LATIN HYPERCUBE SAMPLING
#
# For each parameter separately, [0, 1) is cut into n_samples equal bins and
# exactly ONE random value is drawn inside each bin. The bins are then put in
# random order. Every parameter is therefore spread evenly over its whole
# range, with no clusters and no gaps, using few samples.
#
#   example, n_samples = 4:  bins [0,.25) [.25,.5) [.5,.75) [.75,1)
#                            -> one value in each, order shuffled
# =============================================================================
def latin_hypercube_sampling(n_samples, n_dim, seed=42):
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, n_dim), dtype=np.float64)

    for j in range(n_dim):
        edges = np.linspace(0.0, 1.0, n_samples + 1)
        lower = edges[:-1]
        upper = edges[1:]

        u = rng.random(n_samples)                 # position inside each bin
        values = lower + u * (upper - lower)

        rng.shuffle(values)                       # random order of the bins
        samples[:, j] = values

    return samples


# =============================================================================
# CIRCLE CANDIDATES
#
# z = (zD, zx, zy) in [0, 1)^3 is mapped to physical values:
#   D  = D_min + zD (D_max - D_min)
#   cx, cy chosen so the circle stays at least margin = 1.5 maxh from every
#   plate edge:   r + margin <= cx <= Lx - r - margin   (same for cy with Ly)
# =============================================================================
def generate_lhs_circle_parameters(cfg, n_samples, seed_offset=0):
    z = latin_hypercube_sampling(n_samples, 3, seed=cfg.seed + seed_offset)
    margin = 1.5 * cfg.maxh

    params = []
    for zD, zx, zy in z:
        D = cfg.D_min + zD * (cfg.D_max - cfg.D_min)
        r = 0.5 * D

        x_min = r + margin
        x_max = cfg.Lx - r - margin
        y_min = r + margin
        y_max = cfg.Ly - r - margin

        # SKIP IF THE CIRCLE CANNOT FIT
        if x_max <= x_min or y_max <= y_min:
            continue

        cx = x_min + zx * (x_max - x_min)
        cy = y_min + zy * (y_max - y_min)

        params.append((D, cx, cy))

    return params


# =============================================================================
# RECTANGLE CANDIDATES
#
# z = (zW, zH, zx, zy) in [0, 1)^4, mapped as for the circle.
# NOTE: the edge margin here is a fixed 3 mm, whereas circles use
#       1.5 maxh = 1.2 mm. Kept as in the original to reproduce the dataset.
# =============================================================================
def generate_lhs_rect_parameters(cfg, n_samples, seed_offset=0):
    z = latin_hypercube_sampling(n_samples, 4, seed=cfg.seed + 1000 + seed_offset)
    margin = 0.003

    params = []
    for zW, zH, zx, zy in z:
        W = cfg.W_min + zW * (cfg.W_max - cfg.W_min)
        H = cfg.H_min + zH * (cfg.H_max - cfg.H_min)

        x_min = 0.5 * W + margin
        x_max = cfg.Lx - 0.5 * W - margin
        y_min = 0.5 * H + margin
        y_max = cfg.Ly - 0.5 * H - margin

        # SKIP IF THE RECTANGLE CANNOT FIT
        if x_max <= x_min or y_max <= y_min:
            continue

        cx = x_min + zx * (x_max - x_min)
        cy = y_min + zy * (y_max - y_min)

        params.append((W, H, cx, cy))

    return params


# =============================================================================
# CASE PLAN: THE ORDERED LIST OF CANDIDATES TO TRY
#
# quotas: combined -> n/2 circles and n - n/2 rectangles; otherwise all n of
#         the single family.
# For each family the plan holds the "primary" LHS set followed by a smaller
# "reserve" LHS set (different seed). Reserves are only used if some primary
# candidates are rejected by the FE step.
# Every entry has the same layout: (shape, p1, p2, cx, cy).
# =============================================================================
def build_case_plan(cfg):
    n = cfg.n_cases

    quotas = {}
    if cfg.dataset_family == "combined":
        quotas["circle"] = n // 2
        quotas["rect"] = n - n // 2
    else:
        quotas[cfg.dataset_family] = n

    plan = []
    for family in quotas:
        count = quotas[family]
        n_reserve = max(20, count // 2)

        if family == "circle":
            primary = generate_lhs_circle_parameters(cfg, count)
            reserves = generate_lhs_circle_parameters(cfg, n_reserve, seed_offset=100000)
            for D, cx, cy in primary + reserves:
                plan.append(("circle", D, 0.0, cx, cy))

        elif family == "rect":
            primary = generate_lhs_rect_parameters(cfg, count)
            reserves = generate_lhs_rect_parameters(cfg, n_reserve, seed_offset=100000)
            for W, H, cx, cy in primary + reserves:
                plan.append(("rect", W, H, cx, cy))

        else:
            raise ValueError("Invalid dataset family")

    return plan, quotas


# =============================================================================
# GENERATION ID: A FINGERPRINT OF THE SETTINGS
#
# All settings that change the physics, geometry or grid are written into a
# text string and hashed (SHA-256). Every saved case carries this id; on a
# restart, saved files with a different id are refused, so data made with
# old settings can never be mixed in by mistake.
# =============================================================================
def make_generation_id(cfg):
    keys = (
        "dataset_family", "n_cases", "seed", "Nx", "Ny", "Lx", "Ly", "maxh",
        "order", "kappa", "h_hole", "T_inf", "q_top",
        "D_min", "D_max", "W_min", "W_max", "H_min", "H_max",
    )

    settings = {}
    for key in keys:
        settings[key] = getattr(cfg, key)

    text = json.dumps(settings, sort_keys=True)
    generation_id = hashlib.sha256(text.encode()).hexdigest()

    return generation_id


def fe_case_path(cfg, case_number):
    file_name = "fe_case_%06d.npz" % case_number
    return Path(cfg.fe_outdir) / file_name


def shape_of(record):
    if int(record["shape_code"][0]) == 0:
        return "circle"
    return "rect"


# =============================================================================
# RESTART: RE-LOAD CASES THAT ARE ALREADY ON DISK
#
# Files fe_case_000000, 000001, ... are read in order until one is missing.
# Returns the loaded records, the per-shape counts, and the plan position to
# continue from (one after the last candidate that was used).
# =============================================================================
def load_saved_cases(cfg, generation_id, quotas):
    records = []
    counts = {}
    for family in quotas:
        counts[family] = 0
    candidate_id = 0

    if cfg.force_generation:
        return records, counts, candidate_id

    while len(records) < cfg.n_cases:
        path = fe_case_path(cfg, len(records))
        if not path.exists():
            break

        record = {}
        with np.load(path, allow_pickle=False) as saved:
            for key in saved.files:
                record[key] = saved[key]

        saved_id = str(record.get("generation_id", [""])[0])
        if saved_id != generation_id:
            raise ValueError(
                "FE cache belongs to a different configuration; use --force or another study size"
            )

        records.append((record, cfg.relative_project_path(path)))
        counts[shape_of(record)] += 1
        candidate_id = int(record["candidate_id"][0]) + 1

    return records, counts, candidate_id


# =============================================================================
# ONE ACCEPTED CASE AS A DICTIONARY OF ARRAYS
#
#   points, triangles, temperature : raw FE mesh data (vertices)
#   sdf               : grid SDF divided by max(Lx, Ly) = 36 mm, flattened
#                       -> dimensionless, roughly in [-0.2, 0.9]
#   mask              : 1 solid / 0 cavity, flattened
#   grid_temperature  : T on the grid, 0 (not NaN) in the cavity, flattened
#   generation_id, candidate_id : bookkeeping for restarts
# =============================================================================
def make_record(cfg, shape, p1, p2, cx, cy, sdf, mask, T, pts, triangles, T_fe,
                generation_id, candidate_id):
    if shape == "circle":
        shape_code = 0
    else:
        shape_code = 1

    sdf_scaled = sdf / max(cfg.Lx, cfg.Ly)
    T_grid = np.where(mask > 0.5, T, 0)

    record = {}
    record["points"] = pts.astype(np.float32)
    record["triangles"] = triangles.astype(np.int32)
    record["temperature"] = T_fe.astype(np.float32)
    record["shape_code"] = np.array([shape_code])
    record["p1"] = np.array([p1])
    record["p2"] = np.array([p2])
    record["cx"] = np.array([cx])
    record["cy"] = np.array([cy])
    record["sdf"] = sdf_scaled.ravel()
    record["mask"] = mask.ravel()
    record["grid_temperature"] = T_grid.ravel()
    record["generation_id"] = np.array([generation_id])
    record["candidate_id"] = np.array([candidate_id])

    return record


# =============================================================================
# SAVE ONE CASE SAFELY
#
# Written first to  fe_case_XXXXXX.partial.npz  and only then renamed.
# If the run is killed while writing, no half-written file carries the
# final name, so the restart logic never reads a broken case.
# =============================================================================
def save_record(path, record):
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **record)
    temporary.replace(path)


# =============================================================================
# MAIN ROUTINE: GENERATE THE DATASET
# =============================================================================
def generate_dataset(cfg):
    cfg.ensure_directories()
    if cfg.n_cases != cfg.selected_dataset_size:
        raise ValueError("n_cases must match selected_dataset_size")

    plan, quotas = build_case_plan(cfg)
    generation_id = make_generation_id(cfg)

    # RE-USE CASES ALREADY SOLVED WITH THE SAME SETTINGS
    records, counts, candidate_id = load_saved_cases(cfg, generation_id, quotas)

    # SOLVE NEW CASES UNTIL EVERY QUOTA IS FILLED
    while len(records) < cfg.n_cases:
        if candidate_id >= len(plan):
            raise RuntimeError("Not enough valid candidates; inspect generation failures")

        shape, p1, p2, cx, cy = plan[candidate_id]
        this_candidate = candidate_id
        candidate_id = candidate_id + 1

        # THIS SHAPE ALREADY HAS ENOUGH CASES
        if counts[shape] >= quotas[shape]:
            continue

        print(
            "[%d/%d] %s: p1=%.6f, p2=%.6f, cx=%.6f, cy=%.6f"
            % (len(records) + 1, cfg.n_cases, shape, p1, p2, cx, cy),
            flush=True,
        )

        # MESH + SOLVE + SAMPLE; REJECT THE CANDIDATE IF ANYTHING FAILS
        try:
            _, _, sdf, mask, T, pts, triangles, T_fe = sample_case(cfg, shape, p1, p2, cx, cy)

            solid_values = T[mask > 0.5]
            if not np.isfinite(solid_values).all() or not np.isfinite(T_fe).all():
                raise ValueError("Nonfinite solid FE samples")

        except Exception as error:
            print("Rejected candidate %d: %s" % (this_candidate, error), flush=True)
            continue

        record = make_record(cfg, shape, p1, p2, cx, cy, sdf, mask, T, pts, triangles,
                             T_fe, generation_id, this_candidate)

        path = fe_case_path(cfg, len(records))
        save_record(path, record)

        records.append((record, cfg.relative_project_path(path)))
        counts[shape] += 1

    assert counts == quotas

    # SHUFFLE SO CIRCLES AND RECTANGLES ARE MIXED (FIXED SEED -> REPRODUCIBLE)
    rng = np.random.default_rng(cfg.seed + 999)
    ordering = rng.permutation(len(records))

    shuffled = []
    for i in ordering:
        shuffled.append(records[i])
    records = shuffled

    # COLLECT THE ARRAYS FOR THE FINAL DATASET
    sdf_list = []
    mask_list = []
    temperature_list = []
    meta = []
    fe_paths = []

    for record, path in records:
        sdf_list.append(record["sdf"])
        mask_list.append(record["mask"])
        temperature_list.append(record["grid_temperature"])
        meta.append([
            record["shape_code"][0],
            record["p1"][0],
            record["p2"][0],
            record["cx"][0],
            record["cy"][0],
        ])
        fe_paths.append(path)

    xgrid, ygrid, Xg, Yg, grid_points = make_cartesian_grid(cfg)
    extra = {"generation_id": np.array([generation_id])}

    # SPLIT, NORMALISE AND WRITE dataset.npz  (data/protocol.py)
    return finalize_dataset(cfg, sdf_list, mask_list, temperature_list, meta,
                            fe_paths, grid_points, extra)


# =============================================================================
# PREVIEW: SOLVE AND PLOT ONE FIXED CASE (cfg.preview_*) TO CHECK THE SET-UP
# =============================================================================
def run_preview(cfg):
    cfg.ensure_directories()
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(cfg.fe_outdir, exist_ok=True)

    if cfg.preview_shape == "circle":
        shape = "circle"
        p1 = cfg.preview_D
        p2 = 0.0
    elif cfg.preview_shape == "rect":
        shape = "rect"
        p1 = cfg.preview_W
        p2 = cfg.preview_H
    else:
        raise ValueError("cfg.preview_shape must be 'circle' or 'rect'.")

    mesh, gfu, sdf, mask, T, pts, triangles, T_fe = sample_case(
        cfg, shape, p1, p2, cfg.preview_cx, cfg.preview_cy
    )

    if cfg.export_vtk:
        export_vtk(cfg, mesh, gfu, "preview_case")

    plot_preview(
        cfg=cfg, shape=shape, p1=p1, p2=p2, cx=cfg.preview_cx, cy=cfg.preview_cy,
        sdf=sdf, mask=mask, T=T, pts=pts, triangles=triangles, T_fe=T_fe,
    )
