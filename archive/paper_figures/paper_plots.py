"""Publication figures from saved study records; no model selection on test data.

Run from project root: python -m scripts.paper_plots --device cuda
All uncertainty bars show sample SD across independent training seeds.
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.config import Settings, PROJECT_ROOT
from src.paper_style import COLORS, MODEL_NAMES, paper_style, save_paper_figure, panel_label


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_tuning(rows, output):
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained", sharey=True)
        for index, (ax, activation) in enumerate(zip(axes, ("tanh", "gelu"))):
            for color, rate, style in zip(COLORS, (0.0003, 0.0001), ("--", "-")):
                selected = sorted([r for r in rows if
                    ("gelu" if r["variant"] == "activation_gelu" else "tanh") == activation
                    and r["lr"] == rate], key=lambda r: r["pod_modes"])
                ax.plot([r["pod_modes"] for r in selected],
                        [r["validation_rmse_K"] for r in selected],
                        marker="o", color=color, linestyle=style,
                        label=f"LR = {rate:.0e}")
            ax.set_xticks(sorted(set(r["pod_modes"] for r in rows)))
            ax.set_xlabel("POD modes")
            ax.set_title("Tanh" if activation == "tanh" else "GELU")
            ax.legend()
            panel_label(ax, chr(65 + index))
        axes[0].set_ylabel("Validation RMSE (K)")
        save_paper_figure(fig, Path(output) / "01_pod_tuning")


def plot_oracle(diagnostics, trials, output):
    projection = [r for r in diagnostics["projection"] if r["split"] == "val"]
    projection.sort(key=lambda r: r["modes"])
    ranks = sorted(set(r["pod_modes"] for r in trials))
    with paper_style():
        fig, ax = plt.subplots(figsize=(5.3, 3.8), layout="constrained")
        ax.plot([r["modes"] for r in projection],
                [r["point_weighted_oracle_rmse_K"] for r in projection],
                color=COLORS[0], linestyle="--", marker="s", label="POD oracle")
        ax.plot(ranks, [min(r["validation_rmse_K"] for r in trials if r["pod_modes"] == p)
                       for p in ranks], color=COLORS[1], marker="o", label="POD-DeepONet")
        ax.set_xlabel("POD modes")
        ax.set_ylabel("Validation RMSE (K)")
        ax.set_xticks(sorted(set(ranks + [r["modes"] for r in projection])))
        ax.legend()
        save_paper_figure(fig, Path(output) / "02_pod_oracle")
    with paper_style():
        fig, ax = plt.subplots(figsize=(5.3, 3.8), layout="constrained")
        energy = np.asarray(diagnostics["cumulative_energy"])
        ax.plot(np.arange(1, len(energy)+1), energy*100, color=COLORS[0])
        ax.set_xlabel("POD modes")
        ax.set_ylabel("Cumulative energy (%)")
        ax.set_ylim(0, 100.5)
        save_paper_figure(fig, Path(output) / "03_pod_energy")


def load_final_records(cfg, operators, seeds):
    records, signatures = {}, []
    for operator in operators:
        records[operator] = []
        for seed in seeds:
            run = f"final_{operator}_seed{seed}"
            result = cfg._root(cfg.results_root) / cfg.case_tag / run
            summary = read_json(result / "all_test_cases_grid_summary.json")
            if summary["operator"] != operator or summary["seed"] != seed:
                raise ValueError("Final operator/seed mismatch: " + run)
            with (result / "all_test_cases_grid_metrics.csv").open() as f:
                rows = list(csv.DictReader(f))
            rows.sort(key=lambda r: int(r["case_id"]))
            ids = tuple(int(r["case_id"]) for r in rows)
            if len(ids) != len(set(ids)) or len(ids) != summary["cases"]:
                raise ValueError("Duplicate/missing case IDs: " + run)
            signature = (summary["dataset_sha256"], tuple(summary["grid"]),
                         summary["metric_domain"], ids,
                         tuple((int(r["shape_code"]), int(r["n_points"])) for r in rows))
            signatures.append(signature)
            values = np.array([float(r["rmse_K"]) for r in rows])
            points = np.array([int(r["n_points"]) for r in rows])
            if not np.isfinite(values).all() or (values < 0).any() or (points <= 0).any():
                raise ValueError("Invalid final metrics: " + run)
            if not np.isclose(values.mean(), summary["mean_case_rmse_K"]):
                raise ValueError("CSV/summary mean RMSE mismatch: " + run)
            if not np.isclose(np.sqrt(np.average(values**2, weights=points)),
                              summary["global_point_weighted_rmse_K"]):
                raise ValueError("CSV/summary point-weighted RMSE mismatch: " + run)
            records[operator].append(dict(seed=seed, run=run, summary=summary, rows=rows,
                                          values=values, shapes=np.array([int(r["shape_code"]) for r in rows])))
        first = records[operator][0]["summary"]
        if any(r["summary"]["architecture"] != first["architecture"]
               or (operator == "pod_deeponet" and r["summary"].get("pod_metadata") != first.get("pod_metadata"))
               for r in records[operator]):
            raise ValueError("Seed architectures or POD bases differ: " + operator)
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError("Final evaluations must share dataset, grid, masks and test IDs")
    return records


def plot_final_metrics(records, output):
    operators = list(records)
    table = []
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
        for panel, metric in enumerate(("mean_case_rmse_K", "global_point_weighted_rmse_K")):
            ax = axes[panel]
            for i, operator in enumerate(operators):
                values = np.array([r["summary"][metric] for r in records[operator]])
                sd = values.std(ddof=1) if len(values)>1 else 0.
                ax.bar(i, values.mean(), width=.55, facecolor=COLORS[i], alpha=.25,
                       edgecolor=COLORS[i], linewidth=1.8)
                # Separate outline keeps saturated edges like the reference figures.
                ax.bar(i, values.mean(), width=.55, fill=False, edgecolor=COLORS[i], linewidth=1.8)
                ax.errorbar(i, values.mean(), yerr=sd, color=COLORS[i], capsize=4, fmt="none")
                ax.scatter(i+np.linspace(-.12, .12, len(values)), values, color=COLORS[i], s=18, zorder=3)
                table.append(dict(operator=operator, metric=metric, seeds=len(values),
                                  mean=float(values.mean()), sample_sd=float(sd)))
            ax.set_xticks(range(len(operators)), [MODEL_NAMES[o] for o in operators])
            ax.set_yscale("log")
            ax.set_ylabel("Test RMSE (K)")
            ax.set_title("Mean over cases" if panel == 0 else "Point-weighted")
            panel_label(ax, chr(65+panel))
        save_paper_figure(fig, Path(output) / "05_test_rmse")
    write_csv(Path(output) / "test_rmse.csv", table)
    with paper_style():
        fig, ax = plt.subplots(figsize=(5.4, 3.8), layout="constrained")
        for i, operator in enumerate(operators):
            # One value per case, averaged across seeds. No pooling as independent replicates.
            values = np.sort(np.mean([r["values"] for r in records[operator]], axis=0))
            ax.step(values, np.arange(1, len(values)+1)/len(values), where="post",
                    color=COLORS[i], label=MODEL_NAMES[operator])
        ax.set_xscale("log")
        ax.set_xlabel("Test case RMSE (K)")
        ax.set_ylabel("Cumulative fraction")
        ax.set_ylim(0, 1.02)
        ax.legend()
        save_paper_figure(fig, Path(output) / "06_test_error_distribution")
    families = sorted(set(records[operators[0]][0]["shapes"]))
    with paper_style():
        fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
        width = .7/len(operators)
        family_table = []
        for i, operator in enumerate(operators):
            means, deviations = [], []
            for family in families:
                values = np.array([r["values"][r["shapes"] == family].mean() for r in records[operator]])
                means.append(values.mean())
                deviations.append(values.std(ddof=1) if len(values)>1 else 0.)
                family_table.append(dict(operator=operator, shape_code=int(family),
                    mean_case_rmse_K=float(values.mean()), sample_sd_K=float(deviations[-1])))
            x = np.arange(len(families)) + (i-(len(operators)-1)/2)*width
            ax.bar(x, means, width=width, fill=False, edgecolor=COLORS[i], linewidth=1.6,
                   yerr=deviations, error_kw=dict(ecolor=COLORS[i], capsize=3), label=MODEL_NAMES[operator])
        ax.set_xticks(range(len(families)), [{0:"Circle", 1:"Rectangle", 2:"Rounded rectangle"}.get(int(f), f"Family {f}") for f in families])
        ax.set_ylabel("Test RMSE (K)")
        ax.set_yscale("log")
        ax.legend()
        save_paper_figure(fig, Path(output) / "07_geometry_rmse")
    write_csv(Path(output) / "geometry_rmse.csv", family_table)


def learning_history(cfg, run):
    """Physical point-weighted RMSE from the recorded masked MSE and run scale."""
    folder = cfg._root(cfg.logs_root)/cfg.case_tag/run["run"]
    with (folder/"loss_history.csv").open() as file:
        rows = list(csv.DictReader(file))
    summary = read_json(folder/"training_summary.json")
    if int(summary["seed"]) != int(run["seed"]):
        raise ValueError("Learning-history seed mismatch: " + run["run"])
    norm = summary["normalization"]
    scale = float(norm["T_max"])-float(norm["T_min"])
    mse = np.array([[float(r["train_point_weighted_mse"]),
                     float(r["val_point_weighted_mse"])] for r in rows])
    epochs = np.array([int(r["epoch"]) for r in rows])
    if (not np.isfinite(scale) or scale <= 0 or len(rows) == 0 or not np.isfinite(mse).all()
            or np.any(mse < 0) or np.any(np.diff(epochs) <= 0)):
        raise ValueError("Invalid learning history: " + run["run"])
    return epochs, np.sqrt(mse)*scale


def plot_learning(cfg, records, output):
    """One declared primary seed, with separate fully opaque seed panels."""
    primary_seed = next(iter(records.values()))[0]["seed"]
    seeds = [r["seed"] for r in next(iter(records.values()))]
    if any([r["seed"] for r in runs] != seeds for runs in records.values()):
        raise ValueError("Learning figures require the same ordered seeds")
    table = []
    with paper_style():
        fig, axes = plt.subplots(1, len(records), figsize=(4.2*len(records), 3.6),
                                 layout="constrained", squeeze=False, sharey=True)
        for index, (operator, runs) in enumerate(records.items()):
            ax = axes[0, index]
            run = runs[0]
            epochs, rmse = learning_history(cfg, run)
            ax.semilogy(epochs, rmse[:, 0], color=COLORS[0], label="Training")
            ax.semilogy(epochs, rmse[:, 1], color=COLORS[1], linestyle="--", label="Validation")
            best = int(np.argmin(rmse[:, 1]))
            ax.scatter(epochs[best], rmse[best, 1], marker="*", s=65,
                       color="black", zorder=5, label="Best validation")
            ax.set_xlabel("Epoch")
            ax.set_title(f"{MODEL_NAMES[operator]} · seed {primary_seed}")
            ax.legend()
            panel_label(ax, chr(65+index))
        axes[0, 0].set_ylabel("RMSE (K)")
        save_paper_figure(fig, Path(output) / "04_learning_curves")
    with paper_style():
        fig, axes = plt.subplots(len(records), len(seeds),
            figsize=(3.0*len(seeds), 2.7*len(records)),
            layout="constrained", squeeze=False, sharey="row")
        for oi, (operator, runs) in enumerate(records.items()):
            for si, run in enumerate(runs):
                ax = axes[oi, si]
                epochs, rmse = learning_history(cfg, run)
                ax.semilogy(epochs, rmse[:, 0], color=COLORS[0], label="Training")
                ax.semilogy(epochs, rmse[:, 1], color=COLORS[1], linestyle="--", label="Validation")
                best = int(np.argmin(rmse[:, 1]))
                ax.scatter(epochs[best], rmse[best, 1], marker="*", color="black", s=40, zorder=5)
                ax.set_title(f"{MODEL_NAMES[operator]} · seed {run['seed']}", fontsize=10)
                if si == 0:
                    ax.set_ylabel("RMSE (K)")
                if oi == len(records)-1:
                    ax.set_xlabel("Epoch")
                if oi == 0 and si == 0:
                    ax.legend(fontsize=8)
                for epoch, (train, val) in zip(epochs, rmse):
                    table.append(dict(operator=operator, seed=run["seed"], epoch=int(epoch),
                        training_rmse_K=float(train), validation_rmse_K=float(val)))
        save_paper_figure(fig, Path(output) / "04b_learning_all_seeds")
    write_csv(Path(output)/"learning_rmse.csv", table)


def plot_fields(cfg, records, output):
    import torch
    from src.ml_dataset import load_bundle
    from src.runtime import configure_device, load_checkpoint, predict_tensor
    data = load_bundle(cfg)
    device = configure_device(cfg)
    ids = [int(v) for v in data["test_idx"]]
    # First sorted test ID per geometry family, independent of temperature/prediction errors.
    cases = [min(i for i in ids if int(data["meta"][i, 0]) == family)
             for family in sorted(set(int(data["meta"][i, 0]) for i in ids))]
    ny, nx = int(data["Ny"][0]), int(data["Nx"][0])
    extent = (0, float(data["Lx"][0]), 0, float(data["Ly"][0]))
    predictions = {i: {} for i in cases}
    for operator, runs in records.items():
        run = runs[0]
        model, checkpoint, coords = load_checkpoint(replace(cfg, operator=operator, run_name=run["run"]), data, device)
        if checkpoint["settings"]["train_seed"] != run["seed"]:
            raise ValueError("Field checkpoint seed mismatch")
        from src.data_protocol import sha256_file
        if sha256_file(replace(cfg, operator=operator, run_name=run["run"]).model_path) != run["summary"]["checkpoint_sha256"]:
            raise ValueError("Field checkpoint differs from scored checkpoint")
        norm = checkpoint["normalization"]
        for case in cases:
            raw = torch.as_tensor(data["sdf"][case:case+1], device=device)
            pred = predict_tensor(model, raw, coords, norm, cfg.point_chunk).cpu().numpy()[0]
            predictions[case][operator] = pred*(norm["T_max"]-norm["T_min"])+norm["T_min"]
        del model
    for case in cases:
        valid = data["mask"][case].reshape(ny, nx) > .5
        true = data["physical_temp"][case].reshape(ny, nx)
        temperatures = [true] + [predictions[case][o].reshape(ny, nx) for o in records]
        # Shared colour limits include all displayed predictions; no hidden clipping.
        lo = min(t[valid].min() for t in temperatures)
        hi = max(t[valid].max() for t in temperatures)
        errors = [t-true for t in temperatures[1:]]
        error_limit = max(max(np.abs(e[valid]).max() for e in errors), 1e-12)
        with paper_style():
            fig, axes = plt.subplots(2, len(temperatures), figsize=(3.2*len(temperatures)+1, 4.9), layout="constrained")
            for col, (name, field) in enumerate(zip(["Reference"]+[MODEL_NAMES[o] for o in records], temperatures)):
                ax = axes[0, col]
                im = ax.imshow(np.ma.masked_where(~valid, field), origin="lower", extent=extent,
                               cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
                ax.set_title(name)
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_aspect("equal")
                if col:
                    axes[1, col].imshow(np.ma.masked_where(~valid, errors[col-1]), origin="lower", extent=extent,
                        cmap="RdBu_r", vmin=-error_limit, vmax=error_limit, interpolation="nearest")
                    axes[1, col].set_xlabel("x (m)")
                    axes[1, col].set_ylabel("y (m)")
                    axes[1, col].set_aspect("equal")
            axes[1, 0].imshow(valid, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1,
                             interpolation="nearest")
            axes[1, 0].set_title("Geometry")
            axes[1, 0].set_xlabel("x (m)")
            axes[1, 0].set_ylabel("y (m)")
            fig.colorbar(im, ax=axes[0, :].tolist(), label="Temperature (K)", shrink=.8)
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            fig.colorbar(ScalarMappable(norm=Normalize(-error_limit, error_limit), cmap="RdBu_r"),
                         ax=axes[1, 1:].tolist(), label="Signed error (K)", shrink=.8)
            save_paper_figure(fig, Path(output)/f"08_fields_case_{case:06d}")
    return cases


def plot_basis(cfg, basis_path, output):
    import torch
    pod = torch.load(basis_path, map_location="cpu", weights_only=True)
    ny, nx = int(pod["grid"]["Ny"]), int(pod["grid"]["Nx"])
    extent = (0, pod["grid"]["Lx"], 0, pod["grid"]["Ly"])
    count = min(4, pod["basis"].shape[1])
    with paper_style():
        fig, axes = plt.subplots(1, count+1, figsize=(2.9*(count+1), 3), layout="constrained")
        mean = pod["mean"].numpy().reshape(ny, nx)*(pod["T_max"]-pod["T_min"])+pod["T_min"]
        im = axes[0].imshow(mean, origin="lower", extent=extent, cmap="viridis")
        axes[0].set_title("Mean")
        fig.colorbar(im, ax=axes[0], label="Temperature (K)", shrink=.8)
        basis = pod["basis"][:, :4].numpy()
        limit = float(np.abs(basis).max())
        for i in range(min(4, basis.shape[1])):
            im = axes[i+1].imshow(basis[:, i].reshape(ny, nx), origin="lower", extent=extent,
                                cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[i+1].set_title(f"Mode {i+1}")
        for ax in axes:
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=axes[1:].tolist(), label="Mode amplitude", shrink=.8)
        save_paper_figure(fig, Path(output)/"09_pod_modes")


def plot_timing(cfg, records, output):
    summaries = []
    for operator, runs in records.items():
        path = cfg._root(cfg.results_root)/cfg.case_tag/runs[0]["run"]/"benchmarks/benchmark_summary.json"
        b = read_json(path)
        scored = runs[0]["summary"]
        if b["checkpoint_sha256"] != scored["checkpoint_sha256"] or b["dataset_sha256"] != scored["dataset_sha256"]:
            raise ValueError("Stale benchmark: " + str(path))
        if not all(b["replay_training_settings_match"].values()):
            raise ValueError("Benchmark replay differs from training")
        summaries.append(b)
    first = summaries[0]
    for b in summaries[1:]:
        if any(b[k] != first[k] for k in ("hardware", "grid", "batch_size", "precision", "validation_precision", "compiled", "point_chunk")):
            raise ValueError("Timing comparison protocols differ")
        if [c["case_id"] for c in b["inference_cases"]] != [c["case_id"] for c in first["inference_cases"]]:
            raise ValueError("Timing cases differ")
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
        for panel, (key, factor, ylabel) in enumerate((("training_epoch", 1, "Time per epoch (s)"),
                                                       ("inference_end_to_end_median_s", 1000, "Inference time (ms)"))):
            for i, b in enumerate(summaries):
                value = b[key]["median_s"] if panel == 0 else b[key]
                axes[panel].bar(i, value*factor, width=.55, fill=False, edgecolor=COLORS[i], linewidth=1.8)
            axes[panel].set_xticks(range(len(records)), [MODEL_NAMES[o] for o in records])
            axes[panel].set_ylabel(ylabel)
            panel_label(axes[panel], chr(65+panel))
        save_paper_figure(fig, Path(output)/"10_timing")
    write_csv(Path(output)/"timing.csv", [dict(operator=b["operator"],
        train_epoch_s=b["training_epoch"]["median_s"], inference_ms=1000*b["inference_end_to_end_median_s"],
        parameters=b["real_scalar_parameters"], fixed_buffer_bytes=b.get("fixed_buffer_bytes", 0)) for b in summaries])


def write_report(records, selection, diagnostics, output, field_description=None):
    winner = selection["winner"]
    activation = "GELU" if winner["variant"] == "activation_gelu" else "Tanh"
    lines = ["# Operator learning study", "",
        f"Validation selection: {winner['pod_modes']} POD modes, {activation}, initial LR {selection['initial_learning_rate']:.0e}.",
        f"Best tuning validation RMSE: {winner['validation_rmse_K']:.4f} K (seed 123).",
        "Test data were excluded from basis fitting and configuration selection.", "",
        "| Model | Mean-case test RMSE (K) | Point-weighted test RMSE (K) |",
        "|---|---:|---:|"]
    for operator, runs in records.items():
        values = [np.array([r['summary'][key] for r in runs])
                  for key in ('mean_case_rmse_K', 'global_point_weighted_rmse_K')]
        scores = [f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" for v in values]
        lines.append(f"| {MODEL_NAMES[operator]} | {scores[0]} | {scores[1]} |")
    lines += ["", "Values are mean ± sample SD across independent training seeds scored on the same held-out cases.",
        "Mean-case RMSE weights each case equally. Point-weighted RMSE pools squared errors over all solid grid points.",
        "", "## POD representation check", "",
        "| Modes | Validation oracle RMSE (K) |", "|---:|---:|"]
    for row in diagnostics['projection']:
        if row['split'] == 'val':
            lines.append(f"| {row['modes']} | {row['point_weighted_oracle_rmse_K']:.4f} |")
    lines += ["", "Oracle coefficients use known targets. These errors describe the fixed basis's representation limit; they are not deployable predictions.",
        "Energy measures mean-extended training fields. It is not a measure of validation prediction accuracy.",
        "", "## Figure definitions", "",
        "Learning curves show normalized solid-point MSE against epochs. Validation curves are labelled Validation.",
        "Test error bars show seed SD, not confidence intervals. ECDFs use case errors averaged across seeds.",
        field_description or "Field cases are selected by the first sorted test ID per geometry family, using the first requested seed.",
        "Temperature and signed-error scales are shared across models within each displayed case. Cavity points are masked.",
        "The DeepONet baseline uses its saved geometry-conditioned architecture. No CNN-DeepONet was evaluated.",
        "", "All conclusions apply to this dataset, split, training budget and tested implementations."]
    (Path(output)/"study_summary.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--case-tag", default="combined_0500")
    parser.add_argument("--seeds", default="321,777,2026,31415,27182")
    parser.add_argument("--operators", default="fno,deeponet,pod_deeponet")
    parser.add_argument("--skip-fields", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--output-subdir", default="paper")
    args = parser.parse_args()
    family, count = args.case_tag.rsplit("_", 1)
    cfg = Settings(dataset_family=family, selected_dataset_size=int(count), n_cases=int(count), device=args.device)
    seeds = [int(v) for v in args.seeds.split(",")]
    operators = args.operators.split(",")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Supply at least two distinct seeds for final uncertainty plots")
    if not operators or len(set(operators)) != len(operators) or any(o not in MODEL_NAMES for o in operators):
        raise ValueError("Invalid operator list")
    selection = read_json(cfg._root(cfg.results_root)/cfg.case_tag/"selection/pod_deeponet_selected.json")
    if Path(args.output_subdir).name != args.output_subdir or args.output_subdir in ("", ".", ".."):
        raise ValueError("Output subdirectory must be one folder name")
    directory = cfg._root(cfg.figures_root)/cfg.case_tag/args.output_subdir
    directory.mkdir(parents=True, exist_ok=True)
    trials = selection["trials"]
    basis = cfg._root(selection.get("basis_file", cfg.pod_basis_path))
    from src.data_protocol import sha256_file
    if sha256_file(basis) != selection["basis_sha256"]:
        raise ValueError("Selected POD cache changed after selection")
    # Plot the largest fitted rank's diagnostic, even when rank 128 wins selection.
    largest_trial = max(trials, key=lambda r: r["pod_modes"])
    diagnostic_path = cfg._root(largest_trial.get("basis_file", str(basis))).with_suffix(".json")
    diagnostics = read_json(diagnostic_path)
    if diagnostics["dataset_sha256"] != selection["dataset_sha256"] or diagnostics["test_used"]:
        raise ValueError("Diagnostic dataset/protocol mismatch")
    if diagnostics.get("basis_sha256") and diagnostics["basis_sha256"] != sha256_file(diagnostic_path.with_suffix(".pt")):
        raise ValueError("Oracle diagnostic belongs to another POD cache")
    plot_tuning(trials, directory)
    plot_oracle(diagnostics, trials, directory)
    write_csv(directory/"pod_tuning.csv", trials)
    write_csv(directory/"pod_oracle.csv", diagnostics["projection"])
    manifest = dict(selection=selection["winner"], test_used_for_selection=False,
        uncertainty="Sample SD across training seeds; dots are individual seed scores.",
        loss="Normalized solid-point MSE; blue solid=train, red dashed=validation. Faint curves are all seeds; bold curve is the first requested seed.",
        oracle="Masked least-squares fit using known targets; representation diagnostic, not an inference model.",
        ecdf="One value per test case: RMSE averaged across training seeds.",
        basis="Mean-extended training snapshots; mode amplitudes are dimensionless and use full-grid RMS normalization.")
    if not args.diagnostics_only:
        records = load_final_records(cfg, operators, seeds)
        pod_runs = records.get("pod_deeponet", [])
        if any(r["summary"].get("pod_metadata", {}).get("basis_sha256") != selection["basis_sha256"]
               or r["summary"]["architecture"].get("latent") != selection["winner"]["pod_modes"] for r in pod_runs):
            raise ValueError("Final POD runs differ from selected basis/rank")
        plot_learning(cfg, records, directory)
        plot_final_metrics(records, directory)
        if not args.skip_fields:
            manifest["field_case_ids"] = plot_fields(cfg, records, directory)
            manifest["field_seed"] = seeds[0]
        if args.timing:
            plot_timing(cfg, records, directory)
        write_report(records, selection, diagnostics, directory)
        manifest.update(operators=operators, seeds=seeds)
    plot_basis(cfg, basis, directory)
    (directory/"figure_notes.json").write_text(json.dumps(manifest, indent=2))
    print("Paper figures saved:", directory, flush=True)


if __name__ == "__main__":
    main()
