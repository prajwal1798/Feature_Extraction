"""Fit the fixed POD trunk once, using only the saved training split."""

import json
import time
from pathlib import Path
import numpy as np
import torch
from src.data_protocol import sha256_file
from src.ml_dataset import load_bundle
from src.pod_basis import fit_pod, projection_errors
from src.runtime import atomic_torch_save


def load_pod_basis(cfg, data):
    pod = torch.load(cfg.pod_basis_path, map_location="cpu", weights_only=True)
    if pod.get("pod_protocol") != "mean_extension_v1":
        raise ValueError("Unsupported POD cache protocol")
    if pod["dataset_sha256"] != sha256_file(cfg.dataset_path):
        raise ValueError("POD cache belongs to a different dataset")
    if not np.array_equal(pod["train_idx"].numpy(), data["train_idx"]):
        raise ValueError("POD cache training split mismatch")
    if pod["basis"].shape[0] != data["sdf"].shape[1]:
        raise ValueError("POD grid mismatch")
    return pod


def prepare_pod(cfg):
    torch.set_num_threads(cfg.cpu_threads)
    data = load_bundle(cfg)
    path = Path(cfg.pod_basis_path)
    fit_seconds = None
    if path.exists() and not cfg.force_generation:
        pod = load_pod_basis(cfg, data)
        if pod["basis"].shape[1] < cfg.pod_modes:
            raise ValueError("Cached rank too small; use --force to refit before starting training")
        print("Reusing training-only POD cache:", path, flush=True)
        diagnostic_path = path.with_suffix(".json")
        if diagnostic_path.exists():
            diagnostics = json.loads(diagnostic_path.read_text())
            if (diagnostics["dataset_sha256"] == pod["dataset_sha256"]
                    and diagnostics["test_used"] is False
                    and any(row["split"] == "val" and row["modes"] == cfg.pod_modes
                            for row in diagnostics["projection"])):
                return pod
        print("Completing missing projection diagnostics without refitting the cached basis", flush=True)
        fitted = {key: value.numpy() if isinstance(value, torch.Tensor) else value
                  for key, value in pod.items()}
        fit_seconds = pod.get("basis_fit_cpu_seconds")
    else:
        start = time.perf_counter()
        print(f"Fitting up to {cfg.pod_modes} POD modes from {len(data['train_idx'])} training cases", flush=True)
        fitted = fit_pod(data["temp"], data["mask"], data["train_idx"], cfg.pod_modes)
        fit_seconds = time.perf_counter() - start
        pod = {key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
               for key, value in fitted.items()}
        pod.update(pod_protocol="mean_extension_v1", dataset_sha256=sha256_file(cfg.dataset_path),
                   basis_fit_cpu_seconds=fit_seconds,
                   grid={key: data[key][0].item() for key in ("Nx", "Ny", "Lx", "Ly")},
                   T_min=float(data["T_min"][0]), T_max=float(data["T_max"][0]))
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(pod, path)
    scale = float(data["T_max"][0] - data["T_min"][0])
    diagnostics = {
        "dataset_sha256": pod["dataset_sha256"], "fit_split": "train",
        "basis_sha256": sha256_file(path),
        "test_used": False, "extension": fitted["extension"],
        "mode_scaling": fitted["mode_scaling"],
        "numerical_rank": fitted["numerical_rank"],
        "retained_modes": fitted["retained_modes"],
        "basis_fit_cpu_seconds": fit_seconds,
        "cumulative_energy": fitted["cumulative_energy"].tolist(),
        "projection": [],
        "warning": "POD energy concerns mean-extended fields. Masked least-squares errors below are oracle representation bounds, not network prediction errors.",
    }
    ranks = sorted(set([min(p, fitted["retained_modes"]) for p in (16, 32, 64, 128, cfg.pod_modes)]))
    projection_start = time.perf_counter()
    for modes in ranks:
        for split in ("train", "val"):
            print(f"Oracle projection: {split}, {modes} modes", flush=True)
            rows = projection_errors(data["temp"], data["mask"], data[split + "_idx"],
                                     fitted["mean"], fitted["basis"][:, :modes],
                                     progress=lambda done, total: print(f"  {done}/{total} cases projected", flush=True))
            diagnostics["projection"].append({
                "split": split, "modes": modes,
                "mean_case_oracle_rmse_K": float(np.mean([r["rmse"] for r in rows]) * scale),
                "point_weighted_oracle_rmse_K": float(np.sqrt(sum(r["sse"] for r in rows) / sum(r["points"] for r in rows)) * scale),
            })
    diagnostics["projection_diagnostics_cpu_seconds"] = time.perf_counter() - projection_start
    diagnostic_path = path.with_suffix(".json")
    temporary_path = diagnostic_path.with_suffix(".partial.json")
    temporary_path.write_text(json.dumps(diagnostics, indent=2))
    temporary_path.replace(diagnostic_path)
    print(json.dumps(diagnostics, indent=2), flush=True)
    return pod
