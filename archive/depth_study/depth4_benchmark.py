"""Matched selected-checkpoint timing on one device, without modifying weights.

Writes to outputs/figure_benchmarks, preserving previous benchmark reports.
Run on an allocated GPU, not a SUNBIRD login node.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from src.config import Settings
from src.data_protocol import sha256_file
from src.ml_dataset import load_bundle
from src.runtime import configure_device, load_checkpoint, predict_tensor
from src.benchmarking import benchmark_model, hardware_report, measure, save_measurement
from scripts.paper_plots import load_final_records


def benchmark_scaling(cfg, seeds, batch_sizes, repeats=30, warmups=5, inner=10,
                      epoch_repeats=3, replay=True, benchmark_root="outputs/figure_benchmarks"):
    operators = ["deeponet"]
    from scripts.check_final_figures import check_inputs
    records = check_inputs(cfg, seeds)
    data = load_bundle(cfg)
    device = configure_device(cfg)
    root = cfg._root(benchmark_root) / cfg.case_tag
    root.mkdir(parents=True, exist_ok=True)
    hardware = hardware_report()
    (root / "hardware.json").write_text(json.dumps(hardware, indent=2))
    ids = np.sort(data["test_idx"])
    report = dict(protocol="figure_scaling_v1", dataset_sha256=sha256_file(cfg.dataset_path),
                  seed=seeds[0], batch_sizes=batch_sizes, grid=[cfg.Ny, cfg.Nx],
                  precision=cfg.precision, point_chunk=cfg.point_chunk,
                  hardware=hardware, operators={},
                  scope="In-memory host SDF to host full-grid Kelvin field. Includes transfers, "
                        "standardisation, model, reconstruction and denormalisation; excludes disk I/O.",
                  uncertainty="p05-p95 of repeated wall-clock measurements, not seed uncertainty.")
    previous = json.loads((root / "batch_scaling.json").read_text())
    report["operators"] = {op: previous["operators"][op] for op in ("fno", "pod_deeponet")}
    report["retained_timing_source"] = "Existing FNO/POD timings retained; DeepONet measured in a new session"
    report["retained_hardware"] = previous.get("hardware")
    for operator in operators:
        run = records[operator][0]
        current = replace(cfg, operator=operator, run_name=run["run"], train_seed=run["seed"])
        # Replay copies the checkpoint to a scratch model; scored parameters are untouched.
        if replay:
            benchmark_model(replace(current, results_root=benchmark_root),
                            repeats=repeats, warmups=warmups, epoch_repeats=epoch_repeats,
                            inference_inner=inner)
        print(f"Batch scaling: {operator}", flush=True)
        model, checkpoint, coords = load_checkpoint(current, data, device)
        checkpoint_hash = sha256_file(current.model_path)
        if checkpoint_hash != run["summary"]["checkpoint_sha256"]:
            raise ValueError("Benchmark checkpoint differs from evaluated checkpoint")
        model.eval()
        norm = checkpoint["normalization"]
        folder = root / run["run"] / "scaling"
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        with torch.inference_mode():
            for batch_size in batch_sizes:
                batch_ids = ids[np.arange(batch_size) % len(ids)]
                host = torch.from_numpy(np.ascontiguousarray(data["sdf"][batch_ids]))

                def operation():
                    raw = host.to(device)
                    y = predict_tensor(model, raw, coords, norm, cfg.point_chunk)
                    return (y * (norm["T_max"] - norm["T_min"]) + norm["T_min"]).cpu().numpy()

                try:
                    measurement = measure(operation, device, warmups, repeats, inner)
                    stats = save_measurement(folder, f"batch_{batch_size:03d}", measurement)
                    rows.append(dict(batch_size=batch_size, case_ids=batch_ids.tolist(),
                                     status="completed", **stats,
                                     peak_allocated_bytes=max((r["peak_allocated_bytes"] or 0)
                                                              for r in measurement[0])))
                    print(f"  B={batch_size}: {stats['median_s']*1000:.3f} ms/batch", flush=True)
                except torch.cuda.OutOfMemoryError:
                    rows.append(dict(batch_size=batch_size, case_ids=batch_ids.tolist(),
                                     status="out_of_memory"))
                    torch.cuda.empty_cache()
                    print(f"  B={batch_size}: out of GPU memory", flush=True)
        report["operators"][operator] = dict(run=run["run"], checkpoint_sha256=checkpoint_hash,
                                             architecture=checkpoint["architecture"], rows=rows)
        if sha256_file(current.model_path) != checkpoint_hash:
            raise RuntimeError("Checkpoint changed during benchmark")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        temporary = root / "batch_scaling.partial.json"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(root / "batch_scaling.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--case-tag", default="combined_0500")
    parser.add_argument("--seeds", default="321,777,2026,31415,27182")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--epoch-repeats", type=int, default=3)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--benchmark-root", default="outputs/figure_benchmarks")
    args = parser.parse_args()
    family, count = args.case_tag.rsplit("_", 1)
    cfg = Settings(dataset_family=family, selected_dataset_size=int(count),
                   n_cases=int(count), device=args.device)
    seeds = [int(x) for x in args.seeds.split(",")]
    sizes = sorted(set(int(x) for x in args.batch_sizes.split(",")))
    if not sizes or sizes[0] < 1 or args.repeats < 2 or args.warmups < 0 or args.inner < 1:
        raise ValueError("Invalid benchmark repetition or batch-size settings")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Supply at least two distinct training seeds")
    benchmark_scaling(cfg, seeds, sizes, args.repeats, args.warmups, args.inner,
                      args.epoch_repeats, not args.skip_replay, args.benchmark_root)


if __name__ == "__main__":
    main()
