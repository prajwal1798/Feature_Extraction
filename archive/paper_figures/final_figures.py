"""Complete final figure suite from selected checkpoints and frozen test cases.

Creates a full test-case atlas, selected detail plates, profiles, thermal
diagnostics and measured compute plots. No training or model selection occurs.
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch
from src.config import Settings
from src.data_protocol import sha256_file
from src.ml_dataset import load_bundle
from src.runtime import configure_device, load_checkpoint, predict_tensor
from src.evaluation import compute_metrics
from src.paper_style import COLORS, MODEL_NAMES, paper_style, save_paper_figure, panel_label
from scripts.paper_plots import (load_final_records, read_json, write_csv, plot_tuning,
    plot_oracle, plot_learning, plot_final_metrics, plot_basis, plot_timing, write_report)

OPERATORS = ("fno", "deeponet", "pod_deeponet")
FAMILIES = {0: "Circle", 1: "Rectangle"}


def geometry(data, case):
    ny, nx = int(data["Ny"][0]), int(data["Nx"][0])
    xy = data["coords"].reshape(ny, nx, 2).astype(float)
    x, y = xy[..., 0], xy[..., 1]
    shape, p1, p2, cx, cy = map(float, data["meta"][case, :5])
    if int(shape) == 0:
        distance = np.hypot(x-cx, y-cy)-p1/2
        area, top = np.pi*(p1/2)**2, cy+p1/2
    elif int(shape) == 1:
        dx, dy = np.abs(x-cx)-p1/2, np.abs(y-cy)-p2/2
        distance = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) + np.minimum(np.maximum(dx, dy), 0)
        area, top = p1*p2, cy+p2/2
    else:
        raise ValueError("This geometry analysis supports circle/rectangle only")
    step = np.hypot(float(np.diff(x[0]).mean()), float(np.diff(y[:, 0]).mean()))
    valid = data["mask"][case].reshape(ny, nx) > .5
    return dict(x=x, y=y, valid=valid, distance=distance, band=valid & (distance >= -1e-12)
                & (distance <= 1.5*step), band_width_m=1.5*step, shape=int(shape),
                cx=cx, cy=cy, area_mm2=area*1e6,
                heated_clearance_mm=(float(data["Ly"][0])-top)*1000)


def thermal_metrics(true, prediction, geom):
    valid = geom["valid"]
    if not valid.any():
        raise ValueError("Case has no solid grid points")
    true = np.asarray(true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(true[valid]).all() or not np.isfinite(prediction[valid]).all():
        raise ValueError("Nonfinite solid prediction")
    error = prediction-true
    maximum = float(true[valid].max())
    pred_maximum = float(prediction[valid].max())
    # Deterministic first predicted maximum; distance to nearest reference maximum.
    peak = np.unravel_index(np.argmax(np.where(valid, prediction, -np.inf)), valid.shape)
    reference_peaks = valid & np.isclose(true, maximum, rtol=0, atol=1e-6)
    distances = np.hypot(geom["x"][reference_peaks]-geom["x"][peak],
                         geom["y"][reference_peaks]-geom["y"][peak])
    band = geom["band"]
    heated = valid[-1]
    return dict(maximum_temperature_error_K=abs(pred_maximum-maximum),
                hotspot_displacement_mm=float(distances.min()*1000),
                near_cavity_mae_K=float(np.abs(error[band]).mean()) if band.any() else None,
                near_cavity_points=int(band.sum()),
                heated_wall_mae_K=float(np.abs(error[-1, heated]).mean()) if heated.any() else None)


def infer_all(cfg, data, records, output, force=False):
    ids = np.array([int(row["case_id"]) for row in records[OPERATORS[0]][0]["rows"]])
    if not np.array_equal(ids, np.sort(data["test_idx"])):
        raise ValueError("Scored case IDs differ from dataset test split")
    signature = dict(version=1, dataset_sha256=sha256_file(cfg.dataset_path),
                     ids=ids.tolist(), seeds=[r["seed"] for r in records[OPERATORS[0]]],
                     checkpoints={op: [r["summary"]["checkpoint_sha256"] for r in records[op]]
                                  for op in OPERATORS},
                     inference_sources={name: sha256_file(Path(__file__).resolve().parents[1]/"src"/name)
                                        for name in ("runtime.py", "model.py", "deeponet_model.py",
                                                     "pod_deeponet_model.py", "fno_model.py")},
                     device=cfg.device, point_chunk=cfg.point_chunk)
    cache = cfg._root(cfg.results_root)/cfg.case_tag/"figure_cache/test_predictions.npz"
    predictions = None
    if cache.exists() and not force:
        with np.load(cache, allow_pickle=False) as stored:
            if json.loads(str(stored["signature"].item())) == signature:
                predictions = {op: stored[op].copy() for op in OPERATORS}
                print("Validated prediction cache reused", flush=True)
    if predictions is None:
        device = configure_device(cfg)
        predictions = {}
        for operator in OPERATORS:
            all_seeds = []
            for run in records[operator]:
                current = replace(cfg, operator=operator, run_name=run["run"], train_seed=run["seed"])
                if sha256_file(current.model_path) != run["summary"]["checkpoint_sha256"]:
                    raise ValueError("Checkpoint differs from scored checkpoint: "+run["run"])
                model, checkpoint, coords = load_checkpoint(current, data, device)
                if checkpoint["settings"]["train_seed"] != run["seed"]:
                    raise ValueError("Checkpoint seed mismatch")
                norm = checkpoint["normalization"]
                values = []
                with torch.inference_mode():
                    for case in ids:
                        raw = torch.as_tensor(data["sdf"][case:case+1], device=device)
                        pred = predict_tensor(model, raw, coords, norm, cfg.point_chunk).cpu().numpy()[0]
                        values.append(pred*(norm["T_max"]-norm["T_min"])+norm["T_min"])
                all_seeds.append(np.asarray(values, dtype=np.float32))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("Inference complete: "+run["run"], flush=True)
            predictions[operator] = np.asarray(all_seeds)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name("test_predictions.partial.npz")
        np.savez_compressed(temporary, signature=np.array(json.dumps(signature, sort_keys=True)), **predictions)
        temporary.replace(cache)
    for operator in OPERATORS:
        expected = (len(records[operator]), len(ids), cfg.Nx*cfg.Ny)
        if predictions[operator].shape != expected:
            raise ValueError("Cached prediction dimensions mismatch")
        for si, run in enumerate(records[operator]):
            for ci, case in enumerate(ids):
                valid = data["mask"][case] > .5
                score = compute_metrics(data["physical_temp"][case, valid], predictions[operator][si, ci, valid])
                if not np.isclose(score["rmse_K"], float(run["rows"][ci]["rmse_K"]), rtol=5e-4, atol=2e-4):
                    raise ValueError(f"Reconstructed RMSE differs from saved evaluation: {operator}, {case}")
    return ids, predictions, signature


def select_main_cases(ids, records):
    """Fixed, documented reporting rule; never changes training/model selection."""
    pod = np.mean([r["values"] for r in records["pod_deeponet"]], axis=0)
    fno = np.mean([r["values"] for r in records["fno"]], axis=0)
    shapes = records["fno"][0]["shapes"]
    rows = []
    for family in sorted(set(shapes)):
        group = np.flatnonzero(shapes == family)
        ordered = sorted(group, key=lambda q: (pod[q], int(ids[q])))
        selections = [(ordered[0], "lowest_POD_RMSE"),
                      (ordered[(len(ordered)-1)//2], "median_POD_RMSE"),
                      (ordered[-1], "highest_POD_RMSE"),
                      (max(group, key=lambda q: (fno[q], -int(ids[q]))), "highest_FNO_RMSE")]
        for q, reason in selections:
            rows.append(dict(case_id=int(ids[q]), shape_code=int(family), reason=reason,
                             pod_seed_mean_rmse_K=float(pod[q]), fno_seed_mean_rmse_K=float(fno[q])))
    return sorted(set(r["case_id"] for r in rows)), rows


def plot_case_rmse(records, output):
    """Case means and sample SD across seeds; IDs retain their numeric spacing."""
    output = Path(output)
    operators = list(records)
    ids = np.array([int(r["case_id"]) for r in records[operators[0]][0]["rows"]])
    if np.any(np.diff(ids) <= 0):
        raise ValueError("Case IDs must be unique and sorted")
    seeds = [r["seed"] for r in records[operators[0]]]
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for sample SD")
    table = []
    with paper_style():
        fig, axes = plt.subplots(len(operators), 1, figsize=(10, 2.5*len(operators)),
                                 layout="constrained", sharex=True, squeeze=False)
        for oi, op in enumerate(operators):
            if [r["seed"] for r in records[op]] != seeds:
                raise ValueError("Case plots require the same ordered seeds")
            if any([int(row["case_id"]) for row in run["rows"]] != ids.tolist()
                   for run in records[op]):
                raise ValueError("Case IDs differ across models or seeds")
            ax = axes[oi, 0]
            values = np.asarray([r["values"] for r in records[op]])
            if values.shape != (len(seeds), len(ids)) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid case RMSE values")
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            color = COLORS[OPERATORS.index(op)]
            ax.plot(ids, mean, color=color, marker="o", markersize=2.5, label="Seed mean")
            ax.fill_between(ids, mean-sd, mean+sd, color=color, alpha=.16,
                            linewidth=0, label="Mean ± 1 SD")
            ax.set_title(f"{MODEL_NAMES[op]} · {len(seeds)} seeds")
            ax.set_ylabel("RMSE (K)")
            ax.legend(loc="upper right", ncol=2)
            panel_label(ax, chr(65+oi))
            # Linear axes preserve the complete symmetric band, even if its
            # descriptive lower limit is negative. No clipping or smoothing.
            ax.set_ylim(bottom=min(0., float(np.min(mean-sd))*1.05))
            for cid, m, s in zip(ids, mean, sd):
                table.append(dict(case_id=int(cid), operator=op, seeds=len(seeds),
                    mean_rmse_K=float(m), seed_sd_K=float(s),
                    lower_mean_minus_sd_K=float(m-s), upper_mean_plus_sd_K=float(m+s)))
        axes[-1, 0].set_xlabel("Test case ID")
        save_paper_figure(fig, output/"11_case_rmse")
    write_csv(output/"case_rmse.csv", table)


def case_metric_plots(records, output):
    plot_case_rmse(records, output)
    metrics = [("rmse_K", "RMSE (K)"), ("mae_K", "MAE (K)"), ("max_abs_K", "Maximum absolute error (K)")]
    with paper_style():
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
        for panel, (key, ylabel) in enumerate(metrics):
            values = [np.mean([[float(row[key]) for row in run["rows"]] for run in records[op]], axis=0)
                      for op in OPERATORS]
            boxes = axes[panel].boxplot(values, patch_artist=True, widths=.5,
                                        medianprops=dict(color="black"),
                                        flierprops=dict(marker="o", markersize=3))
            for i, box in enumerate(boxes["boxes"]):
                box.set(facecolor=COLORS[i], edgecolor=COLORS[i], alpha=.25)
            axes[panel].set_xticks([1, 2, 3], [MODEL_NAMES[o] for o in OPERATORS], rotation=15)
            if all(np.all(v > 0) for v in values):
                axes[panel].set_yscale("log")
            axes[panel].set_ylabel(ylabel)
            panel_label(axes[panel], chr(65+panel))
        save_paper_figure(fig, output/"12_error_boxplots")
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), layout="constrained")
        target = np.mean([r["values"] for r in records["pod_deeponet"]], axis=0)
        for ax, op in zip(axes, ("fno", "deeponet")):
            other = np.mean([r["values"] for r in records[op]], axis=0)
            color = COLORS[OPERATORS.index(op)]
            ax.scatter(other, target, color=color, s=18)
            positive = np.concatenate((other[other > 0], target[target > 0]))
            use_log = np.all(other > 0) and np.all(target > 0)
            lo, hi = (positive.min()*.7, positive.max()*1.4) if use_log else (0, max(float(other.max()), float(target.max()), 1e-6)*1.1)
            ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, label="Equal RMSE")
            ax.set(xscale="log" if use_log else "linear", yscale="log" if use_log else "linear", xlim=(lo, hi), ylim=(lo, hi),
                   xlabel=f"{MODEL_NAMES[op]} RMSE (K)", ylabel="POD-DeepONet RMSE (K)")
            ax.legend()
        save_paper_figure(fig, output/"13_paired_rmse")


def sensitivity_plots(data, ids, records, output):
    rows = []
    for q, case in enumerate(ids):
        geom = geometry(data, case)
        for op in OPERATORS:
            rows.append(dict(case_id=int(case), shape_code=geom["shape"], operator=op,
                             cavity_area_mm2=geom["area_mm2"], heated_clearance_mm=geom["heated_clearance_mm"],
                             rmse_K=float(np.mean([run["values"][q] for run in records[op]]))))
    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), layout="constrained", sharey=True)
        for fi, family in enumerate((0, 1)):
            for col, (key, xlabel) in enumerate((("cavity_area_mm2", "Cavity area (mm²)"),
                                               ("heated_clearance_mm", "Heated-wall clearance (mm)"))):
                ax = axes[fi, col]
                for oi, op in enumerate(OPERATORS):
                    selected = [r for r in rows if r["shape_code"] == family and r["operator"] == op]
                    ax.scatter([r[key] for r in selected], [r["rmse_K"] for r in selected],
                               color=COLORS[oi], marker=("o", "s", "^")[oi], s=18, label=MODEL_NAMES[op])
                ax.set(xlabel=xlabel, yscale="log", title=FAMILIES[family])
                if col == 0:
                    ax.set_ylabel("RMSE (K)")
                if fi == 0 and col == 1:
                    ax.legend()
                panel_label(ax, chr(65+fi*2+col))
        save_paper_figure(fig, output/"14_geometry_sensitivity")
    write_csv(output/"geometry_sensitivity.csv", rows)


def field_axes(ax, geom, field, cmap, lo, hi):
    image = ax.pcolormesh(geom["x"]*1000, geom["y"]*1000,
                         np.ma.masked_where(~geom["valid"], field),
                         shading="nearest", cmap=cmap, vmin=lo, vmax=hi, rasterized=True)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", aspect="equal",
           xlim=(geom["x"].min()*1000, geom["x"].max()*1000),
           ylim=(geom["y"].min()*1000, geom["y"].max()*1000))
    return image


def field_plate(data, case, fields, seed):
    geom = geometry(data, case)
    true = data["physical_temp"][case].reshape(geom["valid"].shape)
    all_fields = [true]+[fields[o] for o in OPERATORS]
    lo = min(float(a[geom["valid"]].min()) for a in all_fields)
    hi = max(float(a[geom["valid"]].max()) for a in all_fields)
    errors = [fields[o]-true for o in OPERATORS]
    bound = max(1e-9, max(float(np.abs(a[geom["valid"]]).max()) for a in errors))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    for i, (name, field) in enumerate(zip(["Reference"]+[MODEL_NAMES[o] for o in OPERATORS], all_fields)):
        im = field_axes(axes[0, i], geom, field, "viridis", lo, hi)
        axes[0, i].set_title(name)
        panel_label(axes[0, i], chr(65+i))
    axes[1, 0].pcolormesh(geom["x"]*1000, geom["y"]*1000, geom["valid"],
                         shading="nearest", cmap="Greys", vmin=0, vmax=1, rasterized=True)
    axes[1, 0].set(xlabel="x (mm)", ylabel="y (mm)", aspect="equal",
                   xlim=(0, geom["x"].max()*1000), ylim=(0, geom["y"].max()*1000), title="Geometry")
    panel_label(axes[1, 0], "E")
    for i, (op, error) in enumerate(zip(OPERATORS, errors), 1):
        field_axes(axes[1, i], geom, error, "RdBu_r", -bound, bound)
        rmse = np.sqrt(np.mean(error[geom["valid"]]**2))
        axes[1, i].set_title(f"RMSE = {rmse:.3f} K")
        panel_label(axes[1, i], chr(70+i-1))
    fig.colorbar(im, ax=axes[0].tolist(), label="Temperature (K)", shrink=.85)
    fig.colorbar(ScalarMappable(norm=Normalize(-bound, bound), cmap="RdBu_r"),
                 ax=axes[1, 1:].tolist(), label="Signed error (K)", shrink=.85)
    fig.suptitle(f"Case {case:06d} · {FAMILIES[geom['shape']]} · seed {seed}", fontsize=11)
    return fig


def detail_plate(data, case, fields, output):
    geom = geometry(data, case)
    true = data["physical_temp"][case].reshape(geom["valid"].shape)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6), layout="constrained")
    cavity = ~geom["valid"]
    if not cavity.any():
        raise ValueError("Expected a cavity")
    pad = 3*geom["band_width_m"]
    xlo, xhi = geom["x"][cavity].min()-pad, geom["x"][cavity].max()+pad
    ylo, yhi = geom["y"][cavity].min()-pad, geom["y"][cavity].max()+pad
    for oi, op in enumerate(OPERATORS):
        error = fields[op]-true
        bound = max(1e-9, float(np.abs(error[geom["valid"]]).max()))
        for row in range(2):
            im = field_axes(axes[row, oi], geom, error, "RdBu_r", -bound, bound)
            axes[row, oi].set_title(MODEL_NAMES[op] if row == 0 else "Cavity detail")
            if row:
                axes[row, oi].set(xlim=(max(0, xlo)*1000, min(geom["x"].max(), xhi)*1000),
                                  ylim=(max(0, ylo)*1000, min(geom["y"].max(), yhi)*1000))
            panel_label(axes[row, oi], chr(65+row*3+oi))
        fig.colorbar(im, ax=axes[:, oi].tolist(), label="Signed error (K)", shrink=.8)
    fig.suptitle("Separate error scales; compare colour bars", fontsize=11)
    save_paper_figure(fig, output/f"detail_case_{case:06d}")


def profile_plate(data, case, fields, output):
    geom = geometry(data, case)
    true = data["physical_temp"][case].reshape(geom["valid"].shape)
    row = int(np.argmin(np.abs(geom["y"][:, 0]-geom["cy"])))
    col = int(np.argmin(np.abs(geom["x"][0]-geom["cx"])))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    cuts = [(geom["x"][row]*1000, (row, slice(None)), "x (mm)", f"y = {geom['y'][row,0]*1000:.2f} mm"),
            (geom["y"][:, col]*1000, (slice(None), col), "y (mm)", f"x = {geom['x'][0,col]*1000:.2f} mm"),
            (geom["x"][-1]*1000, (-1, slice(None)), "x (mm)", "Heated wall")]
    csv_rows = []
    for panel, (coordinate, selection, xlabel, title) in enumerate(cuts):
        valid = geom["valid"][selection]
        for name, field, color, style in [("Reference", true, "black", "--")]+[
                (MODEL_NAMES[op], fields[op], COLORS[i], "-") for i, op in enumerate(OPERATORS)]:
            values = np.where(valid, field[selection], np.nan)
            axes[panel].plot(coordinate, values, color=color, linestyle=style, label=name)
            for q in np.flatnonzero(valid):
                csv_rows.append(dict(case_id=int(case), cut=title, model=name,
                                     coordinate_mm=float(coordinate[q]), temperature_K=float(values[q])))
        axes[panel].set(xlabel=xlabel, ylabel="Temperature (K)", title=title)
        panel_label(axes[panel], chr(65+panel))
    axes[0].legend(fontsize=8)
    save_paper_figure(fig, output/f"profiles_case_{case:06d}")
    write_csv(output/f"profiles_case_{case:06d}.csv", csv_rows)


def full_fields(data, ids, predictions, records, output):
    main_ids, selection = select_main_cases(ids, records)
    write_csv(output/"main_case_selection.csv", selection)
    atlas = output/"field_atlas"
    detail = output/"field_details"
    profiles = output/"profiles"
    for folder in (atlas, detail, profiles):
        folder.mkdir(parents=True, exist_ok=True)
    with paper_style(), PdfPages(atlas/"all_test_cases.pdf") as pages:
        for ci, case in enumerate(ids):
            shape = (int(data["Ny"][0]), int(data["Nx"][0]))
            fields = {op: predictions[op][0, ci].reshape(shape) for op in OPERATORS}
            fig = field_plate(data, int(case), fields, records["fno"][0]["seed"])
            pages.savefig(fig, bbox_inches="tight")
            save_paper_figure(fig, atlas/f"case_{case:06d}")
            if case in main_ids:
                detail_plate(data, int(case), fields, detail)
                profile_plate(data, int(case), fields, profiles)
            print(f"Field atlas: {ci+1}/{len(ids)} · case {case:06d}", flush=True)
    return main_ids


def thermal_plots(data, ids, predictions, records, output):
    rows = []
    shape = (int(data["Ny"][0]), int(data["Nx"][0]))
    for ci, case in enumerate(ids):
        geom = geometry(data, case)
        true = data["physical_temp"][case].reshape(shape)
        for op in OPERATORS:
            for si, run in enumerate(records[op]):
                rows.append(dict(case_id=int(case), operator=op, seed=run["seed"],
                                 band_width_mm=geom["band_width_m"]*1000,
                                 **thermal_metrics(true, predictions[op][si, ci].reshape(shape), geom)))
    write_csv(output/"thermal_metrics.csv", rows)
    metrics = [("maximum_temperature_error_K", "Maximum-temperature error (K)"),
               ("hotspot_displacement_mm", "Hotspot displacement (mm)"),
               ("near_cavity_mae_K", "Near-cavity MAE (K)"),
               ("heated_wall_mae_K", "Heated-wall MAE (K)")]
    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), layout="constrained")
        for ax, (key, ylabel), panel in zip(axes.flat, metrics, range(4)):
            for oi, op in enumerate(OPERATORS):
                values = np.array([np.mean([r[key] for r in rows if r["operator"] == op and
                                    r["seed"] == run["seed"] and r[key] is not None]) for run in records[op]])
                if not np.isfinite(values).all():
                    raise ValueError("Thermal diagnostic has no valid observations")
                ax.errorbar(oi, values.mean(), yerr=values.std(ddof=1), marker="o", color=COLORS[oi], capsize=4)
                ax.scatter(oi+np.linspace(-.10, .10, len(values)), values, s=15, color=COLORS[oi])
            ax.set_xticks(range(3), [MODEL_NAMES[o] for o in OPERATORS], rotation=10)
            ax.set_ylabel(ylabel)
            panel_label(ax, chr(65+panel))
        save_paper_figure(fig, output/"15_thermal_diagnostics")
    return rows


def compute_plots(cfg, records, output, benchmark_root="outputs/figure_benchmarks"):
    root = cfg._root(benchmark_root)/cfg.case_tag
    scaling = read_json(root/"batch_scaling.json")
    if scaling["dataset_sha256"] != records["fno"][0]["summary"]["dataset_sha256"]:
        raise ValueError("Timing dataset mismatch")
    if scaling["precision"] != cfg.precision or scaling["point_chunk"] != cfg.point_chunk:
        raise ValueError("Timing precision/chunk mismatch")
    for op in OPERATORS:
        if scaling["operators"][op]["checkpoint_sha256"] != records[op][0]["summary"]["checkpoint_sha256"]:
            raise ValueError("Timing checkpoint mismatch")
    plot_timing(replace(cfg, results_root=benchmark_root), records, output)
    table = []
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
        for oi, op in enumerate(OPERATORS):
            rows = [r for r in scaling["operators"][op]["rows"] if r["status"] == "completed"]
            if not rows:
                raise ValueError("No completed scaling timings: "+op)
            batch = np.array([r["batch_size"] for r in rows])
            median = np.array([r["median_s"] for r in rows])
            lower = np.array([r["p05_s"] for r in rows])
            upper = np.array([r["p95_s"] for r in rows])
            for panel, values, low, high in ((0, median*1000, lower*1000, upper*1000),
                                          (1, batch/median, batch/upper, batch/lower)):
                axes[panel].plot(batch, values, marker="o", color=COLORS[oi], label=MODEL_NAMES[op])
                axes[panel].fill_between(batch, low, high, color=COLORS[oi], alpha=.15)
            for r in rows:
                table.append(dict(operator=op, batch_size=r["batch_size"], median_batch_ms=r["median_s"]*1000,
                                  p05_batch_ms=r["p05_s"]*1000, p95_batch_ms=r["p95_s"]*1000,
                                  cases_per_second=r["batch_size"]/r["median_s"],
                                  peak_gpu_memory_MiB=r["peak_allocated_bytes"]/1024**2))
        for panel, ax in enumerate(axes):
            ax.set_xscale("log", base=2)
            ax.set_xticks(scaling["batch_sizes"], scaling["batch_sizes"])
            ax.set_xlabel("Batch size")
            ax.set_ylabel("Batch latency (ms)" if panel == 0 else "Throughput (cases/s)")
            ax.legend()
            panel_label(ax, chr(65+panel))
        save_paper_figure(fig, output/"16_batch_scaling")
    write_csv(output/"batch_scaling.csv", table)
    summaries = {op: read_json(root/records[op][0]["run"]/"benchmarks/benchmark_summary.json") for op in OPERATORS}
    with paper_style():
        fig, ax = plt.subplots(figsize=(5.3, 3.8), layout="constrained")
        for oi, op in enumerate(OPERATORS):
            benchmark = summaries[op]
            score = records[op][0]["summary"]["mean_case_rmse_K"]
            ax.scatter(benchmark["inference_end_to_end_median_s"]*1000, score,
                       color=COLORS[oi], s=50, label=MODEL_NAMES[op])
        ax.set(xlabel="Inference latency (ms/case)", ylabel="Test RMSE (K)", yscale="log")
        ax.legend()
        save_paper_figure(fig, output/"17_accuracy_latency")
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
        for oi, op in enumerate(OPERATORS):
            folder = root/records[op][0]["run"]/"benchmarks"
            for panel, filename in enumerate(("training_epoch_samples.csv", "validation_pass_samples.csv")):
                with (folder/filename).open() as f:
                    values = np.array([float(r["wall_seconds"]) for r in csv.DictReader(f)])
                axes[panel].scatter(oi+np.linspace(-.1, .1, len(values)), values, color=COLORS[oi], s=22)
                axes[panel].plot([oi-.18, oi+.18], [np.median(values)]*2, color=COLORS[oi])
            replay = summaries[op]["training_epoch"]
            count = int(read_json(cfg._root(cfg.logs_root)/cfg.case_tag/records[op][0]["run"]/"training_summary.json")["epochs_completed"])
            summaries[op]["estimated_training_compute_s"] = count*replay["median_s"]
        for panel, ax in enumerate(axes):
            ax.set_xticks(range(3), [MODEL_NAMES[o] for o in OPERATORS])
            ax.set_ylabel("Time per training epoch (s)" if panel == 0 else "Time per validation pass (s)")
            panel_label(ax, chr(65+panel))
        save_paper_figure(fig, output/"18_training_validation_timing")
    # Measured online work versus prediction count; no unsupported offline/FE intercept.
    with paper_style():
        fig, ax = plt.subplots(figsize=(5.4, 3.8), layout="constrained")
        counts = np.logspace(0, 6, 100)
        for oi, op in enumerate(OPERATORS):
            ax.loglog(counts, counts*summaries[op]["inference_end_to_end_median_s"],
                      color=COLORS[oi], label=MODEL_NAMES[op])
        ax.set(xlabel="Number of predictions", ylabel="Estimated inference time (s)")
        ax.legend()
        save_paper_figure(fig, output/"19_repeated_inference_cost")
    write_csv(output/"compute_cost.csv", [dict(operator=op,
        measured_epoch_median_s=b["training_epoch"]["median_s"],
        measured_validation_median_s=b["validation_pass"]["median_s"],
        estimated_training_compute_s=b["estimated_training_compute_s"],
        measured_inference_median_s=b["inference_end_to_end_median_s"],
        real_scalar_parameters=b["real_scalar_parameters"], fixed_buffer_bytes=b["fixed_buffer_bytes"])
        for op, b in summaries.items()])
    return scaling


def measured_learning_time(cfg, records, output):
    """Optional actual timing sidecars; never manufacture timestamps for old runs."""
    paths = [(op, run, cfg._root(cfg.logs_root)/cfg.case_tag/run["run"]/"epoch_timing.csv")
             for op in OPERATORS for run in records[op]]
    if not all(path.exists() for _, _, path in paths):
        return "unavailable: existing histories have no measured epoch timestamps"
    # Old resumed runs can have timings for only their final segment.
    for _, run, path in paths:
        with path.open() as f:
            timing_rows = list(csv.DictReader(f))
        history_path = path.with_name("loss_history.csv")
        with history_path.open() as f:
            history_rows = list(csv.DictReader(f))
        if [int(r["epoch"]) for r in timing_rows] != [int(r["epoch"]) for r in history_rows]:
            return "unavailable: measured timing covers only part of the training history"
        if not np.allclose([float(r["val_point_weighted_mse"]) for r in timing_rows],
                           [float(r["val_point_weighted_mse"]) for r in history_rows]):
            raise ValueError("Timing validation losses disagree with saved history")
    with paper_style():
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        for op, run, path in paths:
            with path.open() as f:
                rows = list(csv.DictReader(f))
            seconds = np.array([float(r["elapsed_training_s"]) for r in rows])
            losses = np.array([float(r["val_point_weighted_mse"]) for r in rows])
            if not np.isfinite(seconds).all() or np.any(np.diff(seconds) <= 0):
                raise ValueError("Invalid measured epoch timestamps")
            norm = read_json(cfg._root(cfg.logs_root)/cfg.case_tag/run["run"]/"training_summary.json")["normalization"]
            rmse = np.sqrt(losses)*(norm["T_max"]-norm["T_min"])
            first = run is records[op][0]
            if first:
                ax.semilogy(seconds/60, rmse, color=COLORS[OPERATORS.index(op)],
                            label=f"{MODEL_NAMES[op]} · seed {run['seed']}")
        ax.set(xlabel="Training + validation time (min)", ylabel="Validation RMSE (K)")
        ax.legend()
        save_paper_figure(fig, output/"20_measured_learning_time")
    return "produced from measured epoch_timing.csv sidecars"


def run_suite(cfg, seeds, skip_timing=False, skip_fields=False, force=False, benchmark_root="outputs/figure_benchmarks"):
    data = load_bundle(cfg)
    from scripts.check_final_figures import check_inputs
    records = check_inputs(cfg, seeds)
    selection = read_json(cfg._root(cfg.results_root)/cfg.case_tag/"selection/pod_deeponet_selected.json")
    basis = cfg._root(selection.get("basis_file", cfg.pod_basis_path))
    if sha256_file(basis) != selection["basis_sha256"]:
        raise ValueError("Selected POD basis changed")
    for run in records["pod_deeponet"]:
        if run["summary"]["pod_metadata"]["basis_sha256"] != selection["basis_sha256"] or \
           run["summary"]["architecture"]["latent"] != selection["winner"]["pod_modes"]:
            raise ValueError("Final POD checkpoint differs from selected rank/basis")
    diagnostic = cfg._root(max(selection["trials"], key=lambda r:r["pod_modes"]).get("basis_file", str(basis))).with_suffix(".json")
    diagnostics = read_json(diagnostic)
    if diagnostics["test_used"] or diagnostics["dataset_sha256"] != sha256_file(cfg.dataset_path):
        raise ValueError("POD diagnostics are not training/validation-only for this dataset")
    if diagnostics.get("basis_sha256") and diagnostics["basis_sha256"] != sha256_file(diagnostic.with_suffix(".pt")):
        raise ValueError("Oracle diagnostic basis mismatch")
    output = cfg._root(cfg.figures_root)/cfg.case_tag/"paper_final"
    output.mkdir(parents=True, exist_ok=True)
    print(f"Selected POD rank: {selection['winner']['pod_modes']}; output: {output}", flush=True)
    plot_tuning(selection["trials"], output)
    plot_oracle(diagnostics, selection["trials"], output)
    plot_learning(cfg, records, output)
    plot_final_metrics(records, output)
    plot_basis(cfg, basis, output)
    write_report(records, selection, diagnostics, output,
                 "The atlas includes every paired test case. Selected details show low/median/high seed-mean POD error "
                 "and highest seed-mean FNO error per geometry family, deduplicated; fields use the first declared seed.")
    write_csv(output/"pod_tuning.csv", selection["trials"])
    write_csv(output/"pod_oracle.csv", diagnostics["projection"])
    case_metric_plots(records, output)
    ids, predictions, signature = infer_all(cfg, data, records, output, force)
    sensitivity_plots(data, ids, records, output)
    thermal_plots(data, ids, predictions, records, output)
    main_ids = [] if skip_fields else full_fields(data, ids, predictions, records, output)
    scaling = None if skip_timing else compute_plots(cfg, records, output, benchmark_root)
    learning_time = measured_learning_time(cfg, records, output)
    notes = dict(selected_pod_modes=selection["winner"]["pod_modes"], operators=OPERATORS, seeds=seeds,
                 field_seed=seeds[0], all_test_ids=ids.tolist(), main_case_ids=main_ids,
                 provenance=signature, measured_learning_time=learning_time,
                 fields="Single declared seed; all models use identical cases and solid masks. Main cases chosen "
                        "by seed-mean POD low/median/high RMSE and FNO high RMSE per family; atlas includes every case.",
                 colour_scales="Shared within each field plate. Detail plates intentionally use separate symmetric "
                               "error colour bars; limits include all solid values, no hidden clipping.",
                 distributions="One scalar per case after averaging that scalar across seeds. "
                               "Not errors of an ensemble prediction; cases/seeds are not pooled as independent replicates.",
                 seed_error_bars="Sample SD across independent training seeds on the same fixed split; not a confidence interval.",
                 learning_curves="04 shows the declared first seed only; 04b shows every seed in a separate panel. "
                                 "Physical RMSE = sqrt(recorded point-weighted masked MSE) times the saved training temperature range. "
                                 "Training is the recorded minibatch aggregate during optimization, not a reevaluation of the final epoch model.",
                 case_rmse="11 uses separate linear-axis panels and numeric sorted case IDs. Lines show seed-mean case RMSE; "
                           "the complete shaded band is mean ± one sample SD across seeds. IDs are categorical identifiers; lines are guides. "
                           "Models have separate y ranges. Negative lower band limits, if present, are statistical limits, not negative RMSE observations.",
                 thermal="Near-cavity MAE uses solid grid points within 1.5 cell diagonals of the analytic cavity; "
                         "not exact boundary MAE. Heated-wall MAE uses the last grid row. Maximum/hotspot metrics are grid-based. "
                         "Hotspot displacement uses first predicted maximum and nearest tied reference maximum (1e-6 K tolerance).",
                 geometry_sensitivity="Descriptive correlations, not causal effects or an out-of-distribution test.",
                 timing="Same-job, synchronized FP32 timings; p05-p95 repeat variability. Accuracy-latency uses the "
                        "same first-seed checkpoints. Repeated inference is N times measured batch-1 median, an estimate. "
                        "Historical training compute estimates exclude checkpointing and are explicitly labelled. "
                        "No FE speedup, full amortisation or measured historical time curve is invented.")
    (output/"figure_notes.json").write_text(json.dumps(notes, indent=2))
    catalog = ["# Final figure catalogue", "", f"Selected POD rank: {notes['selected_pod_modes']}.",
               "Models: FNO, DeepONet, POD-DeepONet. Fields use seed "+str(seeds[0])+".", "",
               "## Summary figures", ""]
    catalog += [f"- `{p.name}`" for p in sorted(output.glob("*.pdf"))]
    catalog += ["", "## Field atlas", "", f"{len(ids)} paired test-case plates in `field_atlas/all_test_cases.pdf`.",
                "Individual plates: PDF/SVG/600-dpi PNG. IDs/selection reasons in main_case_selection.csv.",
                "Selected-case separate-scale cavity details: field_details/. Temperature line cuts: profiles/.",
                "", "## Definitions", "", *[f"- **{k}:** {v}" for k,v in notes.items() if k in
                ("fields", "colour_scales", "distributions", "seed_error_bars", "thermal", "geometry_sensitivity", "timing")],
                "", "Measured learning-time curve: "+learning_time+"."]
    (output/"figure_catalog.md").write_text("\n".join(catalog)+"\n")
    print("Final figure suite complete: "+str(output), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--case-tag", default="combined_0500")
    parser.add_argument("--seeds", default="321,777,2026,31415,27182")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--skip-fields", action="store_true")
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--benchmark-root", default="outputs/figure_benchmarks")
    args = parser.parse_args()
    family, count = args.case_tag.rsplit("_", 1)
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(seeds)<2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct seeds are required")
    cfg = Settings(dataset_family=family, selected_dataset_size=int(count), n_cases=int(count), device=args.device)
    run_suite(cfg, seeds, args.skip_timing, args.skip_fields, args.force_inference, args.benchmark_root)


if __name__ == "__main__":
    main()
