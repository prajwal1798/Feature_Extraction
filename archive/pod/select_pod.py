"""Select rank, activation and learning rate using completed validation runs."""

import csv
import json
import math
from pathlib import Path
from src.data_protocol import sha256_file


def pod_trials():
    return [(modes, variant, rate, label)
            for modes in (32, 64, 128)
            for variant in ("baseline", "activation_gelu")
            for rate, label in ((0.0003, "3e-4"), (0.0001, "1e-4"))]


def trial_name(modes, variant, label):
    return f"tune_pod_m{modes}_{variant}_lr{label}_seed123"


def select_pod(cfg, extra_modes=0, extra_basis=""):
    dataset_hash = sha256_file(cfg.dataset_path)
    if extra_modes and (extra_modes <= 128 or not extra_basis):
        raise ValueError("Extra rank must exceed 128 and needs --extra-pod-basis")
    def basis_reference(path):
        path = Path(path).resolve()
        from src.config import PROJECT_ROOT
        return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
    original_basis = basis_reference(cfg.pod_basis_path)
    trials = [(m, v, r, label, original_basis) for m, v, r, label in pod_trials()]
    if extra_modes:
        trials += [(extra_modes, v, r, label, basis_reference(cfg._root(extra_basis)))
                   for v in ("baseline", "activation_gelu")
                   for r, label in ((0.0003, "3e-4"), (0.0001, "1e-4"))]
    basis_hashes = {path: sha256_file(cfg._root(path)) for path in set(t[-1] for t in trials)}
    folder = cfg._root(cfg.logs_root) / cfg.case_tag
    rows, common_settings = [], None
    for modes, variant, rate, label, basis_file in trials:
        basis_hash = basis_hashes[basis_file]
        name = trial_name(modes, variant, label)
        result = json.loads((folder / name / "training_summary.json").read_text())
        settings = result["settings"]
        architecture = result["architecture"]
        if (result["operator"] != "pod_deeponet"
                or result["dataset_sha256"] != dataset_hash
                or result["pod_metadata"]["basis_sha256"] != basis_hash
                or result["variant"] != variant
                or settings["pod_modes"] != modes
                or settings["lr"] != rate or settings["train_seed"] != 123
                or architecture["latent"] != modes
                or architecture["branch_hidden"] != [256, 256, 256, 256]
                or architecture["activation"] != ("gelu" if variant == "activation_gelu" else "tanh")
                or result["test_used_for_selection"] is not False):
            raise ValueError("Inconsistent POD tuning run: " + name)
        shared = {key: settings[key] for key in (
            "epochs", "batch_size", "precision", "point_chunk", "grad_clip",
            "early_stop_patience", "scheduler_patience", "scheduler_factor",
        )}
        shared["branch_in"] = architecture["branch_in"]
        if common_settings is None:
            common_settings = shared
        elif shared != common_settings:
            raise ValueError("POD trials differ in uncontrolled training settings")
        if not math.isfinite(result["best_val_loss"]):
            raise ValueError("Nonfinite validation result: " + name)
        rows.append({"run_name": name, "variant": variant, "pod_modes": modes,
                     "basis_file": basis_file, "basis_sha256": basis_hash,
                     "lr": rate, "validation_mse": result["best_val_loss"],
                     "validation_rmse_K": result["best_val_rmse_K"],
                     "best_epoch": result["best_epoch"], "parameters": result["parameters"]})
    winner = min(rows, key=lambda row: row["validation_mse"])
    output = {"code_protocol": "classic_v1", "operator": "pod_deeponet",
              "dataset_sha256": dataset_hash, "basis_sha256": winner["basis_sha256"],
              "basis_file": winner["basis_file"],
              "initial_learning_rate": winner["lr"], "winner": winner,
              "selection_seed": 123, "trial_settings": common_settings,
              "trials": rows, "test_used": False}
    directory = cfg._root(cfg.results_root) / cfg.case_tag / "selection"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pod_deeponet_selected.json").write_text(json.dumps(output, indent=2))
    with (directory / "pod_deeponet_trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, indent=2), flush=True)
    return output
