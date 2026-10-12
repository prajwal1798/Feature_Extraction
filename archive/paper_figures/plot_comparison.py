"""Compare saved tuning and final results. No training or test-based selection."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OPERATORS = ["deeponet", "fno"]
NAMES = ["DeepONet", "FNO"]
COLORS = ["#2563ab", "#d16a22"]
VARIANTS = {
    "deeponet": ["baseline", "activation_gelu", "depth3", "latent256"],
    "fno": ["baseline", "activation_tanh", "depth3", "modes8"],
}
SEEDS = [321, 777, 2026, 31415, 27182]


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def save(figure, folder, name):
    figure.savefig(folder / (name + ".png"), dpi=220, bbox_inches="tight")
    figure.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    plt.close(figure)
    print("Saved:", name)


def loss_plot(run, folder, name):
    history = run["history"]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.semilogy(history[:, 0], history[:, 1], label="Training")
    axis.semilogy(history[:, 0], history[:, 2], label="Validation")
    axis.set(xlabel="Epoch", ylabel="Normalized solid-point MSE", title=name)
    axis.legend()
    save(figure, folder, "loss_history")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--case-tag", default="combined_0500")
    args = parser.parse_args()
    root = args.root.resolve()
    logs = root / "outputs/logs" / args.case_tag
    results = root / "outputs/study_results" / args.case_tag
    figures = root / "outputs/figures" / args.case_tag
    folder = figures / "comparison"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    tuning, finals, rows = {}, {}, []
    fingerprints = set()
    normalizations = []
    for operator in OPERATORS:
        tuning[operator] = []
        for variant in VARIANTS[operator]:
            name = f"tune_{operator}_{variant}_seed123"
            summary = read_json(logs / name / "training_summary.json")
            history = np.atleast_2d(np.loadtxt(logs / name / "loss_history.csv", delimiter=",", skiprows=1))
            fingerprints.add(summary["dataset_sha256"])
            normalizations.append(summary["normalization"])
            run = {"summary": summary, "history": history}
            tuning[operator].append(run)
            rows.append({"operator": operator, "variant": variant,
                         "architecture": json.dumps(summary["architecture"], sort_keys=True),
                         "parameters": summary["parameters"],
                         "best_validation_rmse_K": summary["best_val_rmse_K"],
                         "best_epoch": summary["best_epoch"],
                         "epochs_completed": summary["epochs_completed"], "seed": 123})
        finals[operator] = []
        for seed in SEEDS:
            name = f"final_{operator}_seed{seed}"
            summary = read_json(logs / name / "training_summary.json")
            metrics = read_json(results / name / "all_test_cases_grid_summary.json")
            benchmark = read_json(results / name / "benchmarks/benchmark_summary.json")
            if summary["seed"] != seed or metrics["seed"] != seed or metrics["operator"] != operator:
                raise ValueError("Run metadata mismatch: " + name)
            if metrics["architecture"] != summary["architecture"] or benchmark["architecture"] != summary["architecture"]:
                raise ValueError("Architecture mismatch: " + name)
            if metrics["checkpoint_sha256"] != benchmark["checkpoint_sha256"]:
                raise ValueError("Benchmark is for a different checkpoint: " + name)
            fingerprints.update([summary["dataset_sha256"], metrics["dataset_sha256"], benchmark["dataset_sha256"]])
            normalizations.append(summary["normalization"])
            normalizations.append(benchmark["normalization"])
            history = np.atleast_2d(np.loadtxt(logs / name / "loss_history.csv", delimiter=",", skiprows=1))
            cases = read_rows(results / name / "all_test_cases_grid_metrics.csv")
            finals[operator].append({"summary": summary, "metrics": metrics,
                                     "benchmark": benchmark, "history": history, "cases": cases})
    if len(fingerprints) != 1 or any(n != normalizations[0] for n in normalizations):
        raise ValueError("Datasets or training normalization differ")
    for operator in OPERATORS:
        if any(run["summary"]["architecture"] != finals[operator][0]["summary"]["architecture"] for run in finals[operator]):
            raise ValueError("Final seeds use different architectures")

    with (folder / "tuning_parameters.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Populate the previously empty individual tuning folders from saved CSVs.
    for operator in OPERATORS:
        for run in tuning[operator]:
            destination = figures / run["summary"]["run_name"]
            destination.mkdir(parents=True, exist_ok=True)
            loss_plot(run, destination, run["summary"]["run_name"])

    figure, axis = plt.subplots(figsize=(11, 5))
    values = [row["best_validation_rmse_K"] for row in rows]
    labels = [NAMES[OPERATORS.index(row["operator"])] + "\n" + row["variant"] for row in rows]
    bars = axis.bar(np.arange(8), values, color=[COLORS[OPERATORS.index(row["operator"])] for row in rows])
    axis.set_xticks(np.arange(8), labels, rotation=25, ha="right")
    axis.set_yscale("log")
    axis.set(ylabel="Best validation RMSE [K]", title="All eight tuning configurations: seed 123")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width()/2, value*1.07, f"{value:.4f}", ha="center", fontsize=9)
    axis.set_ylim(min(values)*0.65, max(values)*1.6)
    save(figure, folder, "01_all_tuning_validation")

    for metric, ylabel, filename in [(2, "Validation RMSE [K]", "02_tuning_validation_curves"),
                                       (1, "Training RMSE [K]", "03_tuning_training_curves")]:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for axis, operator, title in zip(axes, OPERATORS, NAMES):
            for run in tuning[operator]:
                h, s = run["history"], run["summary"]
                scale = s["normalization"]["T_max"] - s["normalization"]["T_min"]
                axis.semilogy(h[:, 0], np.sqrt(h[:, metric])*scale, label=s["variant"], linewidth=1)
            axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
            axis.legend(fontsize=8)
        figure.suptitle("Tuning: one seed per configuration; curves end when training stops")
        save(figure, folder, filename)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, setting in zip(axes, ["activation", "depth"]):
        for i, operator in enumerate(OPERATORS):
            runs = tuning[operator]
            if setting == "activation":
                pair = [runs[0], runs[1]] if operator == "deeponet" else [runs[1], runs[0]]
                labels = ["tanh", "GELU"]
            else:
                pair, labels = [runs[0], runs[2]], ["Baseline depth", "Depth 3"]
            values = [run["summary"]["best_val_rmse_K"] for run in pair]
            axis.bar(np.arange(2) + (i-0.5)*0.35, values, width=0.35, color=COLORS[i], label=NAMES[i])
        axis.set_xticks([0, 1], labels)
        axis.set_yscale("log")
        axis.set(ylabel="Best validation RMSE [K]", title=setting.capitalize() + " comparison; seed 123")
        axis.legend()
    figure.text(0.5, -0.03, "DeepONet depth counts hidden layers per branch/trunk; FNO depth counts operator blocks.", ha="center", fontsize=9)
    save(figure, folder, "04_activation_and_depth")

    figure, axis = plt.subplots(figsize=(8, 5))
    for i, operator in enumerate(OPERATORS):
        runs = tuning[operator]
        x = [run["summary"]["parameters"] / 1e6 for run in runs]
        y = [run["summary"]["best_val_rmse_K"] for run in runs]
        axis.scatter(x, y, color=COLORS[i], label=NAMES[i], s=55)
        for j, (xx, yy, run) in enumerate(zip(x, y, runs)):
            axis.annotate(run["summary"]["variant"], (xx, yy), xytext=(5, 5 if j % 2 == 0 else -14), textcoords="offset points", fontsize=8)
    axis.set_yscale("log")
    axis.set(xlabel="Real scalar parameters [millions]", ylabel="Best validation RMSE [K]", title="Accuracy versus model size; tuning seed 123")
    axis.legend()
    save(figure, folder, "05_parameters_vs_validation")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, operator in enumerate(OPERATORS):
        values = np.array([run["metrics"]["mean_case_rmse_K"] for run in finals[operator]])
        axes[0].bar(np.arange(5)+(i-0.5)*0.35, values, width=0.35, color=COLORS[i], label=NAMES[i])
        axes[1].bar(i, values.mean(), yerr=values.std(ddof=1), capsize=6, color=COLORS[i])
        axes[1].scatter(np.full(5, i)+np.linspace(-0.12, 0.12, 5), values, color="black", s=16, zorder=3)
    axes[0].set_xticks(np.arange(5), [str(seed) for seed in SEEDS])
    axes[0].set(xlabel="Training seed", ylabel="Mean test-case RMSE [K]", title="Paired seeds on the same held-out cases")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].set_xticks([0, 1], NAMES)
    axes[1].set(ylabel="Mean test-case RMSE [K]", title="Mean ± sample SD across five seeds")
    save(figure, folder, "06_final_test_seed_comparison")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, operator, title in zip(axes, OPERATORS, NAMES):
        for run in finals[operator]:
            h, s = run["history"], run["summary"]
            scale = s["normalization"]["T_max"] - s["normalization"]["T_min"]
            axis.semilogy(h[:, 0], np.sqrt(h[:, 2])*scale, label=str(s["seed"]), linewidth=1)
        axis.set(title=title, xlabel="Epoch", ylabel="Validation RMSE [K]")
        axis.legend(title="Seed", fontsize=8)
    save(figure, folder, "07_final_validation_curves")

    reference = sorted((int(row["case_id"]), int(row["shape_code"])) for row in finals["deeponet"][0]["cases"])
    case_values = {}
    for operator in OPERATORS:
        arrays = []
        for run in finals[operator]:
            ordered = sorted(run["cases"], key=lambda row: int(row["case_id"]))
            if [(int(row["case_id"]), int(row["shape_code"])) for row in ordered] != reference:
                raise ValueError("Test cases or shape families differ")
            arrays.append([float(row["rmse_K"]) for row in ordered])
        case_values[operator] = np.asarray(arrays)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, operator in enumerate(OPERATORS):
        values = case_values[operator]
        axes[0].plot([case[0] for case in reference], values.mean(axis=0), "o-", markersize=3, color=COLORS[i], label=NAMES[i])
        axes[0].fill_between([case[0] for case in reference], values.min(axis=0), values.max(axis=0), color=COLORS[i], alpha=0.15)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Held-out case ID", ylabel="RMSE [K]", title="Per-case seed mean; shading: seed min–max")
    axes[0].legend()
    x, y = case_values["deeponet"].mean(axis=0), case_values["fno"].mean(axis=0)
    axes[1].scatter(x, y, s=25)
    low, high = min(x.min(), y.min())*0.8, max(x.max(), y.max())*1.2
    axes[1].plot([low, high], [low, high], "k--", label="Equal error")
    axes[1].set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high), xlabel="DeepONet RMSE [K]", ylabel="FNO RMSE [K]", title="Each point: same test case, averaged over seeds")
    axes[1].legend()
    save(figure, folder, "08_paired_test_case_errors")

    benchmarks = [run["benchmark"] for operator in OPERATORS for run in finals[operator]]
    reference_benchmark = benchmarks[0]
    common = ["batch_size", "grid", "precision", "validation_precision", "compiled", "hardware"]
    compatible = all(all(b[key] == reference_benchmark[key] for key in common)
                     and all(b["replay_training_settings_match"].values()) for b in benchmarks)
    if compatible:
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        fields = [("training_epoch", "Training epoch replay [s]", 1),
                  ("inference_forward_median_s", "Prepared-input inference [ms]", 1000),
                  ("inference_end_to_end_median_s", "Host-to-host inference [ms]", 1000)]
        for axis, (key, title, scale) in zip(axes, fields):
            for i, operator in enumerate(OPERATORS):
                records = [run["benchmark"][key] for run in finals[operator]]
                values = np.array([r["median_s"] if isinstance(r, dict) else r for r in records])*scale
                axis.bar(i, values.mean(), yerr=values.std(ddof=1), capsize=5, color=COLORS[i])
                axis.scatter(np.full(5, i), values, color="black", s=12)
            axis.set_xticks([0, 1], NAMES)
            axis.set_title(title, fontsize=10)
            axis.set_ylim(bottom=0)
        figure.suptitle("Benchmark medians per seed; bars: mean ± seed SD (concurrent cluster jobs)")
        save(figure, folder, "09_measured_timing")
    else:
        print("Timing plot skipped: recorded hardware or benchmark protocols differ.")

    (folder / "README.txt").write_text(
        "Tuning uses validation only, one seed (123) per configuration.\n"
        "Activation and depth alternatives change one setting relative to each operator's baseline.\n"
        "Latent dimension and Fourier modes are different architecture-specific parameters.\n"
        "All 8 actual configurations are shown; no unrun combinations are inferred.\n"
        "Final tests use 5 selected-configuration seeds, with common frozen test cases.\n"
        "Seed SD is sample SD, not a confidence interval or per-point error.\n"
        "Timing compares recorded replay/inference benchmarks, not total training duration.\n"
        "Temperature-field images remain in each final run's test folder.\n"
    )
    print("Comparison plots:", folder)


if __name__ == "__main__":
    main()
