# =============================================================================
# METRICS: SCORE A TRAINED MODEL ON ONE SPLIT, IN KELVIN, ON SOLID POINTS
#
# Predictions are converted back to kelvin with the training scale,
#     T = T_min + T_hat (T_max - T_min),
# and compared with the FE temperature at the solid grid points only
# (cavity points carry no temperature). No interpolation is involved.
#
# Per case (e = T_pred - T_FE over the n solid points of that case):
#     RMSE        = sqrt( sum e^2 / n )                                  [K]
#     MAE         = sum |e| / n                                          [K]
#     max |e|                                                            [K]
#     rel. L2     = ||e||_2 / ||T_FE||_2                 (absolute kelvin)
#     rel. L2 rise= ||e||_2 / ||T_FE - T_inf||_2          (temperature rise)
#     R^2         = 1 - sum e^2 / sum (T_FE - mean T_FE)^2
#
# Over the whole split (all solid points of all cases pooled):
#     global RMSE, global relative L2, global R^2
#
# NOTE on relative L2: in absolute kelvin the 300 K coolant level sits in
# the denominator, so values are small (~1e-3). The "rise" version measures
# the error against the part of the field the operator actually has to learn.
# Both average the same case error norms ||e||; only the case weights differ.
# The absolute one is the primary metric, consistent with earlier results.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   split       : "train", "val" or "test"
#   external    : True -> rounded-rectangle set (data/external.py)
#   ids         : case indices scored
#   true, pred  : FE and predicted temperature of one case, (Ny*Nx,)     [K]
#   valid       : solid points of that case
#   rows        : one dictionary of metrics per case
# =============================================================================

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import load_bundle
from ..data.protocol import sha256_file
from ..training.runtime import configure_device, load_checkpoint, predict_tensor
from .plots import plot_case


# =============================================================================
# METRICS OF ONE CASE
# =============================================================================
def compute_metrics(true, pred, T_inf=300.0):
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    if not np.isfinite(pred).all() or len(true) == 0:
        raise ValueError("Invalid predictions or empty solid region")

    error = pred - true
    squared_deviation = np.sum((true - true.mean()) ** 2)

    metrics = {}
    metrics["rmse_K"] = float(np.sqrt(np.mean(error ** 2)))
    metrics["mae_K"] = float(np.mean(np.abs(error)))
    metrics["max_abs_K"] = float(np.max(np.abs(error)))
    metrics["relative_l2"] = float(np.linalg.norm(error) / np.linalg.norm(true))
    metrics["relative_l2_rise"] = float(np.linalg.norm(error) / np.linalg.norm(true - T_inf))
    metrics["r2"] = float(1.0 - np.sum(error ** 2) / squared_deviation)
    metrics["n_points"] = int(len(true))
    return metrics


# =============================================================================
# SCORE EVERY CASE OF ONE SPLIT AND WRITE <name>_grid_metrics.csv / _summary.json
#
# selected_only : score only case number cfg.test_case_number of the split
# make_plots    : also draw reference / prediction / error of that case
# =============================================================================
def evaluate(cfg, split="test", external=False, selected_only=False, make_plots=False):
    cfg.ensure_directories()
    device = configure_device(cfg)

    # WHICH CASES
    if external:
        data = {}
        with np.load(cfg.rounded_test_dataset_path, allow_pickle=False) as saved:
            for key in saved.files:
                data[key] = saved[key]
        ids = np.arange(len(data["sdf"]))
        folder = Path(cfg.rounded_test_outdir)
        selected = cfg.rounded_test_case_number
        name = "all_test_cases"
    else:
        data = load_bundle(cfg)
        ids = data[split + "_idx"]
        folder = Path(cfg.outdir)
        selected = cfg.test_case_number
        name = "all_%s_cases" % split

    if selected_only:
        name = "selected_case"
    if (selected_only or make_plots) and not 0 <= selected < len(ids):
        raise ValueError("Case number %d outside [0, %d]" % (selected, len(ids) - 1))

    model, checkpoint, coordinates = load_checkpoint(cfg, data, device, external=external)
    normalization = checkpoint["normalization"]
    T_range = normalization["T_max"] - normalization["T_min"]
    T_inf = cfg.T_inf

    rows = []
    sum_squared_error = 0.0
    sum_T = 0.0
    sum_T_squared = 0.0
    total_points = 0

    for number, case_id in enumerate(ids):
        if selected_only and number != selected:
            continue

        # PREDICT ONE CASE AND CONVERT TO KELVIN
        raw_sdf = torch.as_tensor(data["sdf"][case_id:case_id + 1], device=device)
        pred = predict_tensor(model, raw_sdf, coordinates, normalization, cfg.point_chunk)
        pred = pred.cpu().numpy()[0]
        pred = pred * T_range + normalization["T_min"]

        if "physical_temp" in data:
            true = data["physical_temp"][case_id]
        else:
            true = data["temp"][case_id] * T_range + normalization["T_min"]

        valid = data["mask"][case_id] > 0.5
        metrics = compute_metrics(true[valid], pred[valid], T_inf)

        # ACCUMULATE THE POOLED (GLOBAL) QUANTITIES IN DOUBLE PRECISION
        true_solid = true[valid].astype(np.float64)
        error_solid = pred[valid].astype(np.float64) - true_solid
        sum_squared_error += float(np.sum(error_solid ** 2))
        sum_T += float(np.sum(true_solid))
        sum_T_squared += float(np.sum(true_solid ** 2))
        total_points += int(valid.sum())

        row = {"case_id": int(case_id), "test_number": number,
               "shape_code": int(data["meta"][case_id, 0])}
        row.update(metrics)
        rows.append(row)

        if make_plots and number == selected:
            if external:
                plot_case(cfg, data, case_id, true, pred, valid, folder_name="rounded")
            else:
                plot_case(cfg, data, case_id, true, pred, valid, folder_name=split)

    # TABLE OF CASES
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / (name + "_grid_metrics.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # SUMMARY
    mean_T = sum_T / total_points
    total_squared_deviation = sum_T_squared - total_points * mean_T ** 2

    families = {}
    for code in sorted(set(row["shape_code"] for row in rows)):
        family_rmse = [row["rmse_K"] for row in rows if row["shape_code"] == code]
        families[str(code)] = float(np.mean(family_rmse))

    summary = {
        "operator": checkpoint.get("model_kind", "deeponet"),
        "run_name": cfg.run_name,
        "split": "external" if external else split,
        "checkpoint_sha256": sha256_file(cfg.model_path),
        "dataset_sha256": checkpoint["dataset_sha256"],
        "architecture": checkpoint["architecture"],
        "seed": checkpoint["settings"]["train_seed"],
        "mean_case_rmse_K": float(np.mean([row["rmse_K"] for row in rows])),
        "global_point_weighted_rmse_K": float(np.sqrt(sum_squared_error / total_points)),
        "mean_case_mae_K": float(np.mean([row["mae_K"] for row in rows])),
        "mean_case_relative_l2": float(np.mean([row["relative_l2"] for row in rows])),
        "mean_case_relative_l2_rise": float(np.mean([row["relative_l2_rise"] for row in rows])),
        "global_relative_l2": float(np.sqrt(sum_squared_error / sum_T_squared)),
        "global_r2": float(1.0 - sum_squared_error / total_squared_deviation),
        "cases": len(rows),
        "grid": [int(data["Ny"][0]), int(data["Nx"][0])],
        "metric_domain": "valid solid Cartesian points, no interpolation",
        "families": families,
    }
    (folder / (name + "_grid_summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return rows
