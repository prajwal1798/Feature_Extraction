# =============================================================================
# REPORT: STUDY-LEVEL TABLES AND FIGURES
#
#   plot_tuning      annotated heatmap (2-D stage) or curve (1-D stage) of
#                    the validation metrics of a tuning stage; best cell marked
#   summarize_seeds  final test metrics of one operator over its seeds
#   compare          DeepONet against FNO on the test split:
#                    learning curves, test RMSE over seeds, case-error
#                    distribution, error by shape family, paired case errors,
#                    timing (if benchmarks exist)
#
# Everything is read from files written by training, evaluation, tuning and
# benchmarking, so figures can be redrawn at any time without retraining.
# Seed spread is shown as the sample standard deviation across independently
# trained seeds (not a confidence interval).
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   operators : ("deeponet", "fno")
#   seeds     : seeds of the final runs, e.g. 321 777 2026 31415 27182
#   records   : per operator a list of per-seed dictionaries:
#               {"seed", "summary" (test summary json), "cases" (per-case rows),
#                "history" (loss_history rows), "T_range"}
# =============================================================================

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .style import paper_style, save_figure, COLORS, MODEL_NAMES, FAMILY_NAMES
from ..training.tuning import tuning_dir, mean_over_seeds

OPERATORS = ("deeponet", "fno")


def read_csv_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_dir(cfg):
    folder = cfg._root(cfg.figures_root) / cfg.case_tag / "report"
    if cfg.dataset_family == "scaling" and cfg.train_size > 0:
        folder = folder / ("N%04d" % cfg.train_size)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# =============================================================================
# TUNING FIGURE
#
# 2-D stage: three heatmaps (relative L2, RMSE, R^2); every cell shows its
#            value (mean over seeds); the selected cell has a thick frame.
# 1-D stage: three panels of metric against the stage variable.
# =============================================================================
TUNING_AXES = {
    ("deeponet", "grid"): ("Hidden layers L", "Neurons per layer N"),
    ("deeponet", "grid_gelu"): ("Hidden layers L", "Neurons per layer N"),
    ("fno", "modes_width"): ("Fourier modes m (m_y = m_x)", "Channel width"),
    ("fno", "depth"): ("Fourier blocks", None),
    ("deeponet", "lr"): ("Learning rate", None),
    ("fno", "lr"): ("Learning rate", None),
}

TUNING_METRICS = [
    ("val_relative_l2", "Validation relative $L_2$", "%.2e", "viridis_r"),
    ("val_rmse_K", "Validation RMSE (K)", "%.2f", "viridis_r"),
    ("val_r2", "Validation $R^2$", "%.4f", "viridis"),
]


def plot_tuning(cfg, operator, stage):
    rows = read_csv_rows(tuning_dir(cfg) / ("%s_%s.csv" % (operator, stage)))
    for row in rows:
        row["cell"] = int(row["cell"])
        row["x"] = float(row["x"])
        row["y"] = None if row["y"] in ("", "None") else float(row["y"])
        row["lr"] = float(row["lr"])
        row["parameters"] = int(row["parameters"])
        for metric, title, number_format, colormap in TUNING_METRICS:
            row[metric] = float(row[metric])
    table = mean_over_seeds(rows)

    selection = json.loads((tuning_dir(cfg) / ("%s_%s_selected.json" % (operator, stage))).read_text())
    best_cell = selection["winner"]["cell"]
    x_label, y_label = TUNING_AXES[(operator, stage)]
    n_seeds = table[0]["n_seeds"]

    with paper_style():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")

        for axis, (metric, title, number_format, colormap) in zip(axes, TUNING_METRICS):

            if y_label is not None:
                # 2-D HEATMAP: columns = x values, rows = y values
                x_values = sorted(set(cell["x"] for cell in table))
                y_values = sorted(set(cell["y"] for cell in table))
                grid = np.full((len(y_values), len(x_values)), np.nan)
                for cell in table:
                    i = y_values.index(cell["y"])
                    j = x_values.index(cell["x"])
                    grid[i, j] = cell[metric]

                image = axis.imshow(grid, origin="lower", cmap=colormap, aspect="auto")
                fig.colorbar(image, ax=axis, shrink=0.9)

                low = np.nanmin(grid)
                high = np.nanmax(grid)
                for cell in table:
                    i = y_values.index(cell["y"])
                    j = x_values.index(cell["x"])
                    # DARK TEXT ON LIGHT CELLS, LIGHT TEXT ON DARK CELLS
                    shade = (cell[metric] - low) / (high - low + 1e-30)
                    if colormap.endswith("_r"):
                        shade = 1.0 - shade
                    color = "black" if shade > 0.5 else "white"
                    axis.text(j, i, number_format % cell[metric], ha="center", va="center",
                              fontsize=8, color=color)
                    if cell["cell"] == best_cell:
                        axis.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                 edgecolor="red", linewidth=3))

                axis.set_xticks(range(len(x_values)))
                axis.set_xticklabels(["%g" % value for value in x_values])
                axis.set_yticks(range(len(y_values)))
                axis.set_yticklabels(["%g" % value for value in y_values])
                axis.set_ylabel(y_label)

            else:
                # 1-D CURVE
                x_values = [cell["x"] for cell in table]
                y_values = [cell[metric] for cell in table]
                axis.plot(x_values, y_values, "o-", color=COLORS[operator])
                for cell in table:
                    axis.annotate(number_format % cell[metric], (cell["x"], cell[metric]),
                                  textcoords="offset points", xytext=(0, 13), ha="center",
                                  fontsize=8)
                    if cell["cell"] == best_cell:
                        axis.plot(cell["x"], cell[metric], "s", markersize=12,
                                  markerfacecolor="none", markeredgecolor="red", markeredgewidth=2)
                axis.margins(x=0.12, y=0.18)
                if stage == "lr":
                    axis.set_xscale("log")
                axis.set_xticks(x_values)
                axis.set_xticklabels(["%g" % value for value in x_values])
                axis.set_ylabel(title)

            axis.set_xlabel(x_label)
            axis.set_title(title)

        fig.suptitle("%s tuning, stage '%s' (validation split, mean of %d seed%s; red = selected)"
                     % (MODEL_NAMES[operator], stage, n_seeds, "" if n_seeds == 1 else "s"))
        save_figure(fig, report_dir(cfg) / ("tuning_%s_%s" % (operator, stage)))


# =============================================================================
# LOAD THE FINAL RUNS OF ONE OPERATOR
# =============================================================================
# =============================================================================
# NAME OF THE FINAL RUN OF ONE OPERATOR AND SEED
#   main study      : final_<operator>_seed<seed>
#   data-size study : scale_<operator>_N<NNNN>_seed<seed>  (cfg.train_size = N)
# =============================================================================
def final_run_name(cfg, operator, seed):
    if cfg.dataset_family == "scaling":
        return "scale_%s_N%04d_seed%d" % (operator, cfg.train_size, seed)
    return "final_%s_seed%d" % (operator, seed)


def load_final_records(cfg, operator, seeds):
    records = []
    for seed in seeds:
        run_name = final_run_name(cfg, operator, seed)
        results = cfg.study_dir / run_name
        logs = cfg._root(cfg.logs_root) / cfg.case_tag / run_name

        summary = json.loads((results / "all_test_cases_grid_summary.json").read_text())
        if summary["operator"] != operator or summary["seed"] != seed:
            raise ValueError("Seed/operator metadata mismatch in %s" % run_name)

        cases = read_csv_rows(results / "all_test_cases_grid_metrics.csv")
        history = np.atleast_2d(np.loadtxt(logs / "loss_history.csv", delimiter=",", skiprows=1))
        training = json.loads((logs / "training_summary.json").read_text())
        T_range = training["normalization"]["T_max"] - training["normalization"]["T_min"]

        benchmark_path = results / "benchmarks" / "benchmark_summary.json"
        benchmark = None
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text())

        records.append({"seed": seed, "summary": summary, "cases": cases, "history": history,
                        "T_range": T_range, "training": training, "benchmark": benchmark})

    # ALL SEEDS MUST SHARE DATASET AND ARCHITECTURE
    for record in records:
        if record["summary"]["dataset_sha256"] != records[0]["summary"]["dataset_sha256"]:
            raise ValueError("Seed runs use different datasets")
        if record["summary"]["architecture"] != records[0]["summary"]["architecture"]:
            raise ValueError("Seed runs use different architectures")
    return records


def case_rmse_matrix(records):
    # rows = seeds, columns = test cases (same order for all seeds)
    case_ids = [int(row["case_id"]) for row in records[0]["cases"]]
    matrix = []
    for record in records:
        if [int(row["case_id"]) for row in record["cases"]] != case_ids:
            raise ValueError("Seeds were scored on different test cases")
        matrix.append([float(row["rmse_K"]) for row in record["cases"]])
    shape_codes = [int(row["shape_code"]) for row in records[0]["cases"]]
    return np.array(case_ids), np.array(shape_codes), np.array(matrix)


def mean_and_sd(values):
    values = np.asarray(values, dtype=float)
    if len(values) > 1:
        return float(values.mean()), float(values.std(ddof=1))
    return float(values.mean()), None


# =============================================================================
# SEED SUMMARY OF ONE OPERATOR  ->  <results>/<tag>/seed_summary/
# =============================================================================
def summarize_seeds(cfg, operator, seeds):
    records = load_final_records(cfg, operator, seeds)

    rows = []
    for record in records:
        summary = record["summary"]
        rows.append({"seed": record["seed"],
                     "mean_case_rmse_K": summary["mean_case_rmse_K"],
                     "global_point_weighted_rmse_K": summary["global_point_weighted_rmse_K"],
                     "mean_case_relative_l2": summary["mean_case_relative_l2"],
                     "global_r2": summary["global_r2"],
                     "best_epoch": record["training"]["best_epoch"]})

    folder = cfg.study_dir / "seed_summary"
    if cfg.dataset_family == "scaling" and cfg.train_size > 0:
        folder = folder / ("N%04d" % cfg.train_size)
    write_csv_rows(folder / ("%s_seeds.csv" % operator), rows)

    result = {"operator": operator, "seeds": list(seeds),
              "scope": "Mean and sample SD across independently trained seeds, same test cases."}
    for metric in ("mean_case_rmse_K", "global_point_weighted_rmse_K", "mean_case_relative_l2",
                   "global_r2"):
        mean, sd = mean_and_sd([row[metric] for row in rows])
        result[metric] = mean
        result[metric + "_seed_sd"] = sd

    (folder / ("%s_summary.json" % operator)).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


# =============================================================================
# DEEPONET AGAINST FNO ON THE TEST SPLIT
# =============================================================================
def compare(cfg, seeds):
    records = {}
    for operator in OPERATORS:
        records[operator] = load_final_records(cfg, operator, seeds)
        summarize_seeds(cfg, operator, seeds)

    folder = report_dir(cfg)

    with paper_style():
        # (1) LEARNING CURVES: validation RMSE in kelvin, every seed
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained", sharey=True)
        for axis, operator in zip(axes, OPERATORS):
            for record in records[operator]:
                history = record["history"]
                axis.semilogy(history[:, 0], np.sqrt(history[:, 2]) * record["T_range"],
                              color=COLORS[operator], alpha=0.6, lw=1.0)
            axis.set_title("%s (%d seeds)" % (MODEL_NAMES[operator], len(records[operator])))
            axis.set_xlabel("Epoch")
        axes[0].set_ylabel("Validation RMSE (K)")
        save_figure(fig, folder / "learning_curves")

        # (2) TEST RMSE: mean over seeds +- SD, single seeds as dots
        fig, axis = plt.subplots(figsize=(4.6, 3.8), layout="constrained")
        table = []
        for position, operator in enumerate(OPERATORS):
            values = [record["summary"]["mean_case_rmse_K"] for record in records[operator]]
            mean, sd = mean_and_sd(values)
            axis.bar(position, mean, color=COLORS[operator], alpha=0.35, width=0.6)
            if sd is not None:
                axis.errorbar(position, mean, yerr=sd, color=COLORS[operator], capsize=5)
            axis.plot([position] * len(values), values, "o", color=COLORS[operator])
            table.append({"operator": operator, "seeds": len(values), "mean_case_rmse_K": mean,
                          "seed_sd_K": sd})
        axis.set_xticks(range(len(OPERATORS)))
        axis.set_xticklabels([MODEL_NAMES[operator] for operator in OPERATORS])
        axis.set_ylabel("Test RMSE, mean over cases (K)")
        save_figure(fig, folder / "test_rmse")
        write_csv_rows(folder / "test_rmse.csv", table)

        # (3) DISTRIBUTION OF CASE ERRORS (seed-mean RMSE per test case)
        fig, axis = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
        case_tables = {}
        for operator in OPERATORS:
            case_ids, shape_codes, matrix = case_rmse_matrix(records[operator])
            case_tables[operator] = (case_ids, shape_codes, matrix.mean(axis=0))
            sorted_errors = np.sort(matrix.mean(axis=0))
            fraction = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            axis.step(sorted_errors, fraction, where="post", color=COLORS[operator],
                      label=MODEL_NAMES[operator])
        axis.set_xscale("log")
        axis.set_xlabel("Case RMSE, mean over seeds (K)")
        axis.set_ylabel("Fraction of test cases")
        axis.legend()
        save_figure(fig, folder / "case_error_distribution")

        # (4) ERROR BY SHAPE FAMILY
        fig, axis = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
        family_rows = []
        codes = sorted(set(case_tables[OPERATORS[0]][1].tolist()))
        width = 0.38
        for k, operator in enumerate(OPERATORS):
            case_ids, shape_codes, case_mean = case_tables[operator]
            for c, code in enumerate(codes):
                value = float(case_mean[shape_codes == code].mean())
                axis.bar(c + (k - 0.5) * width, value, width=width, color=COLORS[operator],
                         label=MODEL_NAMES[operator] if c == 0 else None)
                family_rows.append({"operator": operator, "shape": FAMILY_NAMES[code],
                                    "mean_case_rmse_K": value})
        axis.set_xticks(range(len(codes)))
        axis.set_xticklabels([FAMILY_NAMES[code] for code in codes])
        axis.set_ylabel("Test RMSE (K)")
        axis.legend()
        save_figure(fig, folder / "rmse_by_family")
        write_csv_rows(folder / "rmse_by_family.csv", family_rows)

        # (5) PAIRED CASE ERRORS: same test case, both operators
        fig, axis = plt.subplots(figsize=(4.6, 4.4), layout="constrained")
        ids_d, codes_d, mean_d = case_tables["deeponet"]
        ids_f, codes_f, mean_f = case_tables["fno"]
        if not np.array_equal(ids_d, ids_f):
            raise ValueError("Operators were scored on different test cases")
        for code in codes:
            chosen = codes_d == code
            axis.loglog(mean_f[chosen], mean_d[chosen], "o", label=FAMILY_NAMES[code], alpha=0.8)
        low = min(mean_d.min(), mean_f.min()) * 0.8
        high = max(mean_d.max(), mean_f.max()) * 1.25
        axis.plot([low, high], [low, high], "k--", lw=1)
        axis.set_xlabel("FNO case RMSE (K)")
        axis.set_ylabel("DeepONet case RMSE (K)")
        axis.legend()
        save_figure(fig, folder / "paired_case_rmse")
        paired = []
        for i in range(len(ids_d)):
            paired.append({"case_id": int(ids_d[i]), "shape": FAMILY_NAMES[int(codes_d[i])],
                           "deeponet_rmse_K": float(mean_d[i]), "fno_rmse_K": float(mean_f[i])})
        write_csv_rows(folder / "paired_case_rmse.csv", paired)

        # (6) TIMING, ONLY IF EVERY FINAL RUN WAS BENCHMARKED
        have_benchmarks = True
        for operator in OPERATORS:
            for record in records[operator]:
                if record["benchmark"] is None:
                    have_benchmarks = False
        if have_benchmarks:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
            timing_rows = []
            for position, operator in enumerate(OPERATORS):
                epoch = [r["benchmark"]["training_epoch"]["median_s"] for r in records[operator]]
                inference = [r["benchmark"]["inference_end_to_end_median_s"] * 1e3
                             for r in records[operator]]
                axes[0].bar(position, np.median(epoch), color=COLORS[operator])
                axes[1].bar(position, np.median(inference), color=COLORS[operator])
                timing_rows.append({"operator": operator, "train_epoch_s": float(np.median(epoch)),
                                    "inference_ms": float(np.median(inference)),
                                    "parameters": records[operator][0]["training"]["parameters"]})
            for axis in axes:
                axis.set_xticks(range(len(OPERATORS)))
                axis.set_xticklabels([MODEL_NAMES[operator] for operator in OPERATORS])
            axes[0].set_ylabel("Training epoch (s)")
            axes[1].set_ylabel("Batch-1 inference (ms)")
            save_figure(fig, folder / "timing")
            write_csv_rows(folder / "timing.csv", timing_rows)
        else:
            print("Timing figure skipped: run the benchmark stage on every final run first.")

    print("Comparison report: %s" % folder, flush=True)
