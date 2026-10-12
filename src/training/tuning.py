# =============================================================================
# TUNING: HYPERPARAMETER GRIDS FOR DEEPONET AND FNO
#
# Every grid cell is one independent training run with the SAME data split,
# optimiser, epoch budget, early stopping and scheduler; only the listed
# architecture fields (or the learning rate) change. Each finished cell is
# scored on the VALIDATION split; the test split is never used to choose.
#
# DEEPONET
#   stage "grid"      : hidden layers L in {2,4,6,8,10} x neurons N in {10,...,160}
#                       25 cells; L and N are applied to BOTH branch and trunk:
#                       branch 12288 -> N x L -> 128,   trunk 3 -> N x L -> 128
#                       latent p = 128, tanh, lr = cfg.lr  (all held fixed)
#   stage "grid_gelu" : the same 25 cells with GELU instead of tanh
#   stage "lr"        : learning rate in {3e-4, 1e-3, 3e-3} at the better of the
#                       two grid winners (lower validation relative L2)
#
# FNO  (three small stages instead of one large product grid)
#   stage "modes_width": modes m in {4,8,12,16} (m_y = m_x = m) x
#                        width in {16,32,64};  4 blocks, gelu      12 cells
#   stage "depth"      : blocks in {2,3,4,6} at the modes_width winner  4 cells
#   stage "lr"         : learning rate in {3e-4,1e-3,3e-3} at the depth winner
#
# Selection score: mean over seeds of the validation mean-case relative L2
# error (lower is better). RMSE and R^2 are recorded alongside.
#
# Files (under <results>/<case_tag>/tuning/):
#   <operator>_<stage>.csv            one row per cell and seed
#   <operator>_<stage>_selected.json  the winning cell of that stage
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   operator : "deeponet" or "fno"
#   stage    : name of a tuning stage (see above)
#   cell     : {"index", "label", "x", "y", "fields"}; fields are the cfg
#              values that define this cell; x, y are its grid coordinates
#   seeds    : training seeds of every cell (one seed first; more later to
#              measure the spread)
# =============================================================================

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..data.protocol import sha256_file
from ..analysis.metrics import evaluate
from .trainer import train_model

DEEPONET_LAYERS = [2, 4, 6, 8, 10]
DEEPONET_NEURONS = [10, 20, 40, 80, 160]
FNO_MODES = [4, 8, 12, 16]
FNO_WIDTHS = [16, 32, 64]
FNO_BLOCKS = [2, 3, 4, 6]
LEARNING_RATES = [3e-4, 1e-3, 3e-3]

STAGES = {"deeponet": ["grid", "grid_gelu", "lr"], "fno": ["modes_width", "depth", "lr"]}

# SETTINGS THAT MUST BE IDENTICAL FOR ALL CELLS OF A STAGE (fair comparison)
SHARED_SETTINGS = ("epochs", "batch_size", "precision", "point_chunk", "grad_clip",
                   "early_stop_patience", "scheduler_patience", "scheduler_factor",
                   "train_size")


def tuning_dir(cfg):
    folder = cfg.study_dir / "tuning"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def lr_label(lr):
    return "%.0e" % lr


# =============================================================================
# THE WINNER OF AN EARLIER STAGE (its cfg fields)
# =============================================================================
def selected_fields(cfg, operator, stage):
    path = tuning_dir(cfg) / ("%s_%s_selected.json" % (operator, stage))
    if not path.exists():
        raise FileNotFoundError("Stage '%s' of %s has not been selected yet: %s"
                                % (stage, operator, path))
    return json.loads(path.read_text())["winner"]["fields"]


# =============================================================================
# DEEPONET: THE BETTER OF THE TANH AND GELU GRID WINNERS
# (lower mean validation relative L2; a grid not yet selected is left out)
# =============================================================================
def best_grid_fields(cfg):
    best = None
    for stage in ("grid", "grid_gelu"):
        path = tuning_dir(cfg) / ("deeponet_%s_selected.json" % stage)
        if not path.exists():
            print("DeepONet stage '%s' not selected; left out of the comparison" % stage,
                  flush=True)
            continue
        winner = json.loads(path.read_text())["winner"]
        if best is None or winner["val_relative_l2"] < best[1]["val_relative_l2"]:
            best = (stage, winner)
    if best is None:
        raise FileNotFoundError("Neither DeepONet grid has been selected yet "
                                "(run tune-select --stage-name grid / grid_gelu)")
    stage, winner = best
    print("Learning-rate stage starts from the %s winner %s (val rel. L2 %.4e)"
          % (stage, winner["label"], winner["val_relative_l2"]), flush=True)
    return winner["fields"]


# =============================================================================
# CELLS OF ONE STAGE
# =============================================================================
def grid_cells(cfg, operator, stage):
    cells = []

    if operator == "deeponet" and stage in ("grid", "grid_gelu"):
        if stage == "grid":
            activation = "tanh"
        else:
            activation = "gelu"
        for L in DEEPONET_LAYERS:
            for N in DEEPONET_NEURONS:
                fields = {"branch_hidden": (N,) * L, "trunk_hidden": (N,) * L,
                          "latent_dim": 128, "deeponet_activation": activation, "lr": cfg.lr}
                cells.append({"label": "L%d_N%d" % (L, N), "x": L, "y": N, "fields": fields})

    elif operator == "fno" and stage == "modes_width":
        for m in FNO_MODES:
            for width in FNO_WIDTHS:
                fields = {"fno_modes_y": m, "fno_modes_x": m, "fno_width": width,
                          "fno_blocks": 4, "fno_activation": "gelu",
                          "fno_padding": cfg.fno_padding, "lr": cfg.lr}
                cells.append({"label": "m%d_w%d" % (m, width), "x": m, "y": width,
                              "fields": fields})

    elif operator == "fno" and stage == "depth":
        base = selected_fields(cfg, "fno", "modes_width")
        for blocks in FNO_BLOCKS:
            fields = dict(base)
            fields["fno_blocks"] = blocks
            cells.append({"label": "b%d" % blocks, "x": blocks, "y": None, "fields": fields})

    elif stage == "lr":
        if operator == "deeponet":
            base = best_grid_fields(cfg)
        else:
            base = selected_fields(cfg, "fno", "depth")
        for lr in LEARNING_RATES:
            fields = dict(base)
            fields["lr"] = lr
            cells.append({"label": "lr" + lr_label(lr), "x": lr, "y": None, "fields": fields})

    else:
        raise ValueError("Unknown tuning stage %r for %s" % (stage, operator))

    for index in range(len(cells)):
        cells[index]["index"] = index
    return cells


# =============================================================================
# SETTINGS OF ONE CELL AND SEED
# =============================================================================
def cell_settings(cfg, operator, stage, cell, seed):
    if stage == "lr":
        run_name = "tune_%s_%s_%s_seed%d" % (operator, stage, cell["label"], seed)
    else:
        run_name = "tune_%s_%s_%s_lr%s_seed%d" % (operator, stage, cell["label"],
                                                  lr_label(cell["fields"]["lr"]), seed)

    cell_cfg = replace(cfg, operator=operator, variant="custom", train_seed=seed,
                       run_name=run_name, resume=False)
    for name in cell["fields"]:
        value = cell["fields"][name]
        if isinstance(value, list):
            value = tuple(value)
        setattr(cell_cfg, name, value)
    return cell_cfg


def cell_is_finished(cell_cfg):
    return (Path(cell_cfg.outdir) / "all_val_cases_grid_summary.json").exists()


# =============================================================================
# TRAIN ONE CELL (or skip it if done; resume it if interrupted)
# Called once per SLURM array task, or in a loop for local runs.
# =============================================================================
def run_cell(cfg, operator, stage, index, seed):
    cells = grid_cells(cfg, operator, stage)
    if not 0 <= index < len(cells):
        raise ValueError("Cell index %d outside [0, %d]" % (index, len(cells) - 1))

    cell_cfg = cell_settings(cfg, operator, stage, cells[index], seed)

    if cell_is_finished(cell_cfg):
        print("Finished cell kept: %s" % cell_cfg.run_name, flush=True)
        return

    summary_path = Path(cell_cfg.log_outdir) / "training_summary.json"
    if summary_path.exists():
        # TRAINED BUT NOT YET SCORED ON VALIDATION
        evaluate(cell_cfg, split="val")
        return

    recovery = Path(cell_cfg.checkpoint_outdir) / "last_checkpoint.pt"
    if recovery.exists() or Path(cell_cfg.model_path).exists():
        cell_cfg.resume = True

    print("Tuning %s / %s cell %d of %d: %s"
          % (operator, stage, index + 1, len(cells), cell_cfg.run_name), flush=True)
    train_model(cell_cfg)


def run_stage(cfg, operator, stage, seeds, index=None):
    cells = grid_cells(cfg, operator, stage)
    if index is None:
        indices = range(len(cells))
    else:
        indices = [index]
    for i in indices:
        for seed in seeds:
            run_cell(cfg, operator, stage, i, seed)


# =============================================================================
# COLLECT THE FINISHED CELLS INTO <operator>_<stage>.csv
# =============================================================================
def collect_stage(cfg, operator, stage, seeds):
    dataset_hash = sha256_file(cfg.dataset_path)
    rows = []
    reference_settings = None

    for cell in grid_cells(cfg, operator, stage):
        for seed in seeds:
            cell_cfg = cell_settings(cfg, operator, stage, cell, seed)
            if not cell_is_finished(cell_cfg):
                raise FileNotFoundError("Cell not finished: %s" % cell_cfg.run_name)

            training = json.loads((Path(cell_cfg.log_outdir) / "training_summary.json").read_text())
            validation = json.loads((Path(cell_cfg.outdir) / "all_val_cases_grid_summary.json").read_text())

            if training["dataset_sha256"] != dataset_hash:
                raise ValueError("Dataset differs between tuning cells: %s" % cell_cfg.run_name)

            settings = training["settings"]
            shared = {}
            for name in SHARED_SETTINGS:
                shared[name] = settings.get(name)
            if reference_settings is None:
                reference_settings = shared
            if shared != reference_settings:
                raise ValueError("Unfair comparison, settings differ in %s" % cell_cfg.run_name)

            rows.append({
                "operator": operator,
                "stage": stage,
                "cell": cell["index"],
                "label": cell["label"],
                "x": cell["x"],
                "y": cell["y"],
                "lr": cell["fields"]["lr"],
                "seed": seed,
                "run_name": cell_cfg.run_name,
                "parameters": training["parameters"],
                "best_epoch": training["best_epoch"],
                "epochs_completed": training["epochs_completed"],
                "val_relative_l2": validation["mean_case_relative_l2"],
                "val_relative_l2_rise": validation["mean_case_relative_l2_rise"],
                "val_rmse_K": validation["global_point_weighted_rmse_K"],
                "val_mean_case_rmse_K": validation["mean_case_rmse_K"],
                "val_r2": validation["global_r2"],
            })

    path = tuning_dir(cfg) / ("%s_%s.csv" % (operator, stage))
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Tuning table: %s" % path, flush=True)
    return rows


# =============================================================================
# MEAN OVER SEEDS PER CELL (used by selection and by the heatmaps)
# =============================================================================
def mean_over_seeds(rows):
    cells = {}
    for row in rows:
        key = row["cell"]
        if key not in cells:
            cells[key] = {"cell": key, "label": row["label"], "x": row["x"], "y": row["y"],
                          "lr": row["lr"], "parameters": row["parameters"],
                          "val_relative_l2": [], "val_rmse_K": [], "val_r2": []}
        cells[key]["val_relative_l2"].append(row["val_relative_l2"])
        cells[key]["val_rmse_K"].append(row["val_rmse_K"])
        cells[key]["val_r2"].append(row["val_r2"])

    table = []
    for key in sorted(cells):
        cell = cells[key]
        cell["n_seeds"] = len(cell["val_r2"])
        for metric in ("val_relative_l2", "val_rmse_K", "val_r2"):
            values = np.array(cell[metric])
            cell[metric + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            cell[metric] = float(values.mean())
        table.append(cell)
    return table


# =============================================================================
# SELECT THE WINNER OF A STAGE (lowest mean validation relative L2)
# =============================================================================
def select_stage(cfg, operator, stage, seeds):
    rows = collect_stage(cfg, operator, stage, seeds)
    table = mean_over_seeds(rows)

    best = table[0]
    for cell in table:
        if cell["val_relative_l2"] < best["val_relative_l2"]:
            best = cell

    cells = grid_cells(cfg, operator, stage)
    winner = dict(best)
    winner["fields"] = cells[best["cell"]]["fields"]

    output = {
        "operator": operator,
        "stage": stage,
        "dataset_sha256": sha256_file(cfg.dataset_path),
        "seeds": list(seeds),
        "criterion": "lowest mean validation mean-case relative L2 (kelvin)",
        "test_used": False,
        "winner": winner,
        "cells": table,
    }
    path = tuning_dir(cfg) / ("%s_%s_selected.json" % (operator, stage))
    path.write_text(json.dumps(output, indent=2, default=list))
    print("Selected %s / %s: %s  (val rel. L2 %.4e, RMSE %.3f K, R2 %.4f)"
          % (operator, stage, best["label"], best["val_relative_l2"], best["val_rmse_K"],
             best["val_r2"]), flush=True)
    return output


# =============================================================================
# SETTINGS FOR THE FINAL RUNS: THE WINNER OF THE LAST STAGE
# =============================================================================
def final_settings(cfg, operator, seed):
    last_stage = STAGES[operator][-1]
    fields = selected_fields(cfg, operator, last_stage)

    final_cfg = replace(cfg, operator=operator, variant="custom", train_seed=seed,
                        run_name="final_%s_seed%d" % (operator, seed))
    for name in fields:
        value = fields[name]
        if isinstance(value, list):
            value = tuple(value)
        setattr(final_cfg, name, value)
    return final_cfg
