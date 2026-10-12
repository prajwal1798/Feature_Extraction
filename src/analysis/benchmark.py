# =============================================================================
# BENCHMARK: HARDWARE RECORD AND SYNCHRONISED TIMINGS OF A TRAINED MODEL
#
# Run separately from training, on an otherwise idle device, after a run is
# finished. For one checkpoint it measures
#   training_epoch   one full training epoch replayed on a copy of the model
#                    (transfer, forward, masked loss, backward, clip, Adam)
#   validation_pass  one pass over the validation split
#   forward          batch-1 operator evaluation on prepared device inputs
#   end_to_end       batch-1 host SDF -> host temperature in kelvin
#   batch scaling    end-to-end time per batch for batch sizes 1..32
#
# Timing method: warm-up calls first; then for every repeat the device is
# synchronised, perf_counter_ns is read, the function runs "inner" times,
# the device is synchronised again. Reported: median and 5/95 percentiles
# of the wall time per call. CUDA event times are saved alongside.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   warmups   : untimed calls before measuring
#   repeats   : timed samples
#   inner     : calls per sample (short calls are repeated to beat timer noise)
#   rows      : one dictionary per sample (times, memory)
# =============================================================================

import copy
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import load_bundle, make_loaders
from ..data.protocol import sha256_file
from ..models.build import parameter_count
from ..training.runtime import (configure_device, load_checkpoint, train_epoch, validate,
                                predict_tensor, build_case_dependent_trunk_input, fno_field,
                                scale_sdf)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# =============================================================================
# HARDWARE AND SOFTWARE RECORD (saved next to every training run and benchmark)
# =============================================================================
def gpu_telemetry():
    query = ("index,name,driver_version,memory.total,clocks.current.sm,clocks.max.sm,"
             "temperature.gpu,power.draw,power.limit,utilization.gpu,pstate")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + query, "--format=csv"],
                                capture_output=True, text=True, timeout=5, check=True)
        return {"query": query, "csv": result.stdout.strip()}
    except Exception as error:
        return {"unavailable": str(error)}


def hardware_report():
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_model": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch_version": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "software": {},
        "slurm": {},
    }

    for name in ("numpy", "torch", "ngsolve", "netgen-mesher", "matplotlib"):
        try:
            report["software"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["software"][name] = None

    # READABLE CPU NAME ON LINUX
    if platform.system() == "Linux" and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                report["cpu_model"] = line.split(":", 1)[1].strip()
                break

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report["active_cuda_device"] = {
            "index": torch.cuda.current_device(),
            "name": properties.name,
            "vram_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "multiprocessors": properties.multi_processor_count,
        }

    for key in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
                "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_GPUS",
                "SLURM_ARRAY_TASK_ID"):
        report["slurm"][key] = os.environ.get(key)

    if hasattr(os, "sched_getaffinity"):
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))

    report["gpu_telemetry"] = gpu_telemetry()
    return report


# =============================================================================
# TIME ONE FUNCTION
# =============================================================================
def measure(function, device, warmups=5, repeats=30, inner=1):
    for w in range(warmups):
        function()
    synchronize(device)

    rows = []
    for repeat in range(repeats):
        synchronize(device)

        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        # NOTHING BUT THE MEASURED CALLS INSIDE THIS INTERVAL
        start = time.perf_counter_ns()
        if device.type == "cuda":
            start_event.record()
        for i in range(inner):
            function()
        if device.type == "cuda":
            end_event.record()
        synchronize(device)
        wall_seconds = (time.perf_counter_ns() - start) * 1e-9 / inner

        row = {"repeat": repeat, "wall_seconds": wall_seconds, "cuda_event_seconds": None,
               "peak_allocated_bytes": None, "additional_peak_allocated_bytes": None}
        if device.type == "cuda":
            row["cuda_event_seconds"] = float(start_event.elapsed_time(end_event)) * 1e-3 / inner
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            row["additional_peak_allocated_bytes"] = row["peak_allocated_bytes"] - baseline
        rows.append(row)

    values = np.array([row["wall_seconds"] for row in rows])
    summary = {
        "median_s": float(np.median(values)),
        "mean_s": float(values.mean()),
        "std_s": float(values.std(ddof=1)) if len(values) > 1 else None,
        "p05_s": float(np.quantile(values, 0.05)),
        "p95_s": float(np.quantile(values, 0.95)),
        "repeats": repeats,
        "warmups": warmups,
        "inner_repetitions": inner,
    }
    return rows, summary


def save_measurement(folder, name, measurement):
    rows, summary = measurement
    with open(folder / (name + "_samples.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


# =============================================================================
# FULL BENCHMARK OF ONE CHECKPOINT  ->  <outdir>/benchmarks/
# =============================================================================
def benchmark_model(cfg, repeats=30, warmups=5, epoch_repeats=3, inference_inner=20,
                    batch_sizes=(1, 2, 4, 8, 16, 32)):
    device = configure_device(cfg)
    data = load_bundle(cfg)
    model, checkpoint, coordinates = load_checkpoint(cfg, data, device)
    normalization = checkpoint["normalization"]

    # TIME THE EPOCH ON THE CASES THIS RUN WAS TRAINED ON (data-size study: first N)
    n_trained = len(checkpoint["splits"]["train_idx"])
    data["train_idx"] = data["train_idx"][:n_trained]
    train_loader, val_loader, test_loader = make_loaders(data, cfg)

    folder = Path(cfg.outdir) / "benchmarks"
    folder.mkdir(parents=True, exist_ok=True)
    report = hardware_report()
    (folder / "hardware.json").write_text(json.dumps(report, indent=2))

    summary = {
        "operator": model.model_kind,
        "checkpoint_sha256": sha256_file(cfg.model_path),
        "dataset_sha256": checkpoint["dataset_sha256"],
        "real_scalar_parameters": parameter_count(model),
        "batch_size": cfg.batch_size,
        "grid": [cfg.Ny, cfg.Nx],
        "point_chunk": cfg.point_chunk,
        "precision": cfg.precision,
        "architecture": checkpoint["architecture"],
        "hardware": {"os": report["os"], "cpu_model": report["cpu_model"],
                     "gpu": report.get("active_cuda_device"),
                     "torch_version": report["torch_version"]},
        "timing_backend": "perf_counter_ns with explicit CUDA synchronisation",
        "inference_cases": [],
    }

    # TRAINING EPOCH ON A SCRATCH COPY (the trained weights are not changed)
    scratch = copy.deepcopy(model)
    if device.type == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(scratch.parameters(), lr=checkpoint["settings"]["lr"])

    def replay_training_epoch():
        return train_epoch(scratch, train_loader, coordinates, optimizer, cfg, normalization, device)

    summary["training_epoch"] = save_measurement(
        folder, "training_epoch", measure(replay_training_epoch, device, 1, epoch_repeats))
    del scratch, optimizer
    model.to(device)

    # VALIDATION PASS
    def validation_pass():
        return validate(model, val_loader, coordinates, cfg, normalization, device)

    summary["validation_pass"] = save_measurement(
        folder, "validation_pass", measure(validation_pass, device, 1, epoch_repeats))

    # BATCH-1 INFERENCE ON THE FIRST FIVE TEST CASES
    T_range = normalization["T_max"] - normalization["T_min"]
    model.eval()
    with torch.inference_mode():
        for case_id in data["test_idx"][:5]:
            host_sdf = torch.from_numpy(data["sdf"][case_id:case_id + 1])
            sdf = scale_sdf(host_sdf.to(device), normalization)

            if model.model_kind == "fno":
                field = fno_field(sdf, coordinates, model.architecture)

                def forward():
                    return model(field)
            else:
                trunk_inputs = []
                for first in range(0, sdf.shape[1], cfg.point_chunk):
                    last = first + cfg.point_chunk
                    trunk_inputs.append(build_case_dependent_trunk_input(coordinates[first:last],
                                                                         sdf[:, first:last]))

                def forward():
                    branch_features = model.branch(sdf)
                    pieces = []
                    for trunk_input in trunk_inputs:
                        pieces.append(model.forward_from_branch(branch_features, trunk_input))
                    return torch.cat(pieces, dim=1)

            def end_to_end():
                prediction = predict_tensor(model, host_sdf.to(device), coordinates, normalization,
                                            cfg.point_chunk)
                return (prediction * T_range + normalization["T_min"]).cpu().numpy()

            prefix = "case_%06d" % int(case_id)
            forward_summary = save_measurement(
                folder, prefix + "_forward", measure(forward, device, warmups, repeats, inference_inner))
            end_summary = save_measurement(
                folder, prefix + "_end_to_end",
                measure(end_to_end, device, warmups, repeats, inference_inner))
            summary["inference_cases"].append({"case_id": int(case_id), "forward": forward_summary,
                                               "end_to_end": end_summary})

        # BATCH-SIZE SCALING OF END-TO-END INFERENCE
        summary["batch_scaling"] = []
        all_test_sdf = data["sdf"][data["test_idx"]]
        for batch in batch_sizes:
            if batch > len(all_test_sdf):
                break
            host_batch = torch.from_numpy(all_test_sdf[:batch])

            def batch_inference():
                prediction = predict_tensor(model, host_batch.to(device), coordinates,
                                            normalization, cfg.point_chunk)
                return (prediction * T_range + normalization["T_min"]).cpu().numpy()

            rows, result = measure(batch_inference, device, warmups, repeats, 10)
            summary["batch_scaling"].append({
                "batch_size": batch,
                "median_batch_ms": result["median_s"] * 1e3,
                "p05_batch_ms": result["p05_s"] * 1e3,
                "p95_batch_ms": result["p95_s"] * 1e3,
                "cases_per_second": batch / result["median_s"],
            })

    forward_medians = [case["forward"]["median_s"] for case in summary["inference_cases"]]
    end_medians = [case["end_to_end"]["median_s"] for case in summary["inference_cases"]]
    summary["inference_forward_median_s"] = float(np.median(forward_medians))
    summary["inference_end_to_end_median_s"] = float(np.median(end_medians))

    (folder / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print("Benchmark report: %s" % folder, flush=True)
    return summary
