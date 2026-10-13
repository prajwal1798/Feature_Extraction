# =============================================================================
# EXTERNAL TEST SET: ROUNDED RECTANGLES (OUT-OF-DISTRIBUTION GEOMETRY)
#
# The networks are trained on circles and sharp rectangles only. This set
# asks how they behave on a shape family they have never seen: rectangles
# with rounded corners of radius R.
#
# The temperature scale (T_min, T_max) is taken from the TRAINING dataset,
# so the networks see these cases exactly as they would see new data.
# Nothing here is used for training or model selection.
#
# Layout of meta for this set: [2, W, H, R, cx, cy]   (shape code 2)
# =============================================================================

# =============================================================================
# VARIABLES (listed once; see also data/generate.py)
#
#   R                 : corner radius, R_min .. R_max, at most 0.45 min(W,H) [m]
#   requested_cases   : number of rounded-rectangle cases wanted
#   T_min, T_max      : training temperature scale                     [K]
#   rounded_T_min/max : actual temperature range of this external set   [K]
# =============================================================================

import os

import numpy as np

from ..fem.sampling import make_cartesian_grid, sample_case
from .protocol import sha256_file
from .generate import latin_hypercube_sampling


# =============================================================================
# CANDIDATE GEOMETRIES (LHS in 5 parameters: W, H, R, cx, cy)
# =============================================================================
def generate_lhs_rounded_rect_parameters(cfg, n_samples):
    z = latin_hypercube_sampling(n_samples=n_samples, n_dim=5, seed=cfg.rounded_test_seed)
    margin = cfg.rounded_margin

    params = []
    for zW, zH, zR, zx, zy in z:
        W = cfg.rounded_W_min + zW * (cfg.rounded_W_max - cfg.rounded_W_min)
        H = cfg.rounded_H_min + zH * (cfg.rounded_H_max - cfg.rounded_H_min)

        R_requested = cfg.rounded_R_min + zR * (cfg.rounded_R_max - cfg.rounded_R_min)
        R = min(R_requested, 0.45 * min(W, H))

        x_min = 0.5 * W + margin
        x_max = cfg.Lx - 0.5 * W - margin
        y_min = 0.5 * H + margin
        y_max = cfg.Ly - 0.5 * H - margin

        if x_max <= x_min or y_max <= y_min:
            continue

        cx = x_min + zx * (x_max - x_min)
        cy = y_min + zy * (y_max - y_min)
        params.append((W, H, R, cx, cy))

    return params


# =============================================================================
# SAVE ONE ROUNDED-RECTANGLE FE CASE
# =============================================================================
def save_rounded_fe_case(cfg, case_id, pts, triangles, T_fe, W, H, R, cx, cy):
    os.makedirs(cfg.rounded_test_fe_outdir, exist_ok=True)

    file_name = "rounded_rect_fe_case_%06d.npz" % case_id
    fe_path = os.path.join(cfg.rounded_test_fe_outdir, file_name)

    np.savez(
        fe_path,
        points=pts.astype(np.float32),
        triangles=triangles.astype(np.int32),
        temperature=T_fe.astype(np.float32),
        shape_code=np.array([2.0], dtype=np.float32),
        p1=np.array([W], dtype=np.float32),
        p2=np.array([H], dtype=np.float32),
        p3=np.array([R], dtype=np.float32),
        cx=np.array([cx], dtype=np.float32),
        cy=np.array([cy], dtype=np.float32),
    )
    return fe_path


# =============================================================================
# MAIN ROUTINE: GENERATE THE EXTERNAL TEST DATASET
# =============================================================================
def generate_rounded_rect_test_dataset(cfg):
    os.makedirs(cfg.rounded_test_fe_outdir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.rounded_test_dataset_path), exist_ok=True)

    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(
            "Training dataset not found: %s\nGenerate the circle/rectangle training "
            "dataset first so its normalization scale can be reused." % cfg.dataset_path
        )

    # TRAINING SCALE AND GRID
    training_data = np.load(cfg.dataset_path, allow_pickle=True)
    T_min = float(training_data["T_min"][0])
    T_max = float(training_data["T_max"][0])
    cfg.Lx = float(training_data["Lx"][0])
    cfg.Ly = float(training_data["Ly"][0])
    cfg.Nx = int(training_data["Nx"][0])
    cfg.Ny = int(training_data["Ny"][0])

    xgrid, ygrid, Xg, Yg, grid_points = make_cartesian_grid(cfg)

    # OVERSAMPLE CANDIDATES IN CASE SOME FAIL TO MESH
    requested_cases = int(cfg.rounded_test_n_cases)
    candidate_count = max(int(np.ceil(1.8 * requested_cases)), requested_cases + 20)
    case_plan = generate_lhs_rounded_rect_parameters(cfg, candidate_count)

    sdf_all = []
    mask_all = []
    temp_all = []
    meta_all = []
    fe_paths = []
    solid_values = []

    accepted_cases = 0
    candidate_id = 0

    while accepted_cases < requested_cases:
        if candidate_id >= len(case_plan):
            raise RuntimeError(
                "Could not generate enough rounded-rectangle cases. Accepted %d / %d."
                % (accepted_cases, requested_cases)
            )

        W, H, R, cx, cy = case_plan[candidate_id]
        candidate_id = candidate_id + 1

        print(
            "[%d/%d] Shape=rounded_rect, W=%.6f, H=%.6f, R=%.6f, cx=%.6f, cy=%.6f"
            % (accepted_cases + 1, requested_cases, W, H, R, cx, cy),
            flush=True,
        )

        try:
            _, _, sdf, mask, T, pts, triangles, T_fe = sample_case(
                cfg, "rounded_rect", W, H, cx, cy, p3=R
            )
        except Exception as error:
            print("Meshing/solving failed, candidate skipped: %s" % error, flush=True)
            continue

        if not np.isfinite(T[mask > 0.5]).all() or not np.isfinite(T_fe).all():
            print("Rejected nonfinite rounded FE case", flush=True)
            continue

        fe_path = save_rounded_fe_case(cfg, accepted_cases, pts, triangles, T_fe,
                                       W, H, R, cx, cy)

        sdf_all.append((sdf / max(cfg.Lx, cfg.Ly)).ravel())
        mask_all.append(mask.ravel())
        temp_all.append(np.nan_to_num(T, nan=0.0).ravel())
        solid_values.append(T[mask > 0.5].ravel())
        meta_all.append([2.0, W, H, R, cx, cy])
        fe_paths.append(cfg.relative_project_path(fe_path))

        accepted_cases = accepted_cases + 1

    sdf_all = np.asarray(sdf_all, dtype=np.float32)
    mask_all = np.asarray(mask_all, dtype=np.float32)
    temp_all = np.asarray(temp_all, dtype=np.float32)
    meta_all = np.asarray(meta_all, dtype=np.float32)
    fe_paths = np.asarray(fe_paths, dtype=str)

    all_solid = np.concatenate(solid_values)
    rounded_T_min = float(np.min(all_solid))
    rounded_T_max = float(np.max(all_solid))

    # SCALE WITH THE TRAINING CONSTANTS (values may fall outside [0, 1])
    temp_scaled = (temp_all - T_min) / (T_max - T_min + 1e-12)
    temp_scaled = temp_scaled * mask_all

    np.savez(
        cfg.rounded_test_dataset_path,
        sdf=sdf_all,
        mask=mask_all,
        temp=temp_scaled,
        physical_temp=temp_all,
        protocol_version=np.array([2]),
        source_training_sha256=np.array([sha256_file(cfg.dataset_path)]),
        meta=meta_all,
        fe_paths=fe_paths,
        coords=grid_points,
        T_min=np.array([T_min], dtype=np.float64),
        T_max=np.array([T_max], dtype=np.float64),
        rounded_T_min=np.array([rounded_T_min], dtype=np.float32),
        rounded_T_max=np.array([rounded_T_max], dtype=np.float32),
        Lx=np.array([cfg.Lx], dtype=np.float64),
        Ly=np.array([cfg.Ly], dtype=np.float64),
        Nx=np.array([cfg.Nx], dtype=np.int32),
        Ny=np.array([cfg.Ny], dtype=np.int32),
        maxh=np.array([cfg.maxh], dtype=np.float32),
        dataset_family=np.array(["rounded_rect_external_test"], dtype=str),
        n_cases=np.array([accepted_cases], dtype=np.int32),
        source_training_dataset=np.array([cfg.relative_project_path(cfg.dataset_path)], dtype=str),
    )

    print("Training scale T_min = %.6f K, T_max = %.6f K" % (T_min, T_max))
    print("Rounded set range    = %.6f K .. %.6f K" % (rounded_T_min, rounded_T_max))
    print("Saved rounded-rectangle test dataset to: %s" % cfg.rounded_test_dataset_path)
    return cfg.rounded_test_dataset_path
