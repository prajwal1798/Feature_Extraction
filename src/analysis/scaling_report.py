# =============================================================================
# DATA-SIZE STUDY: LEARNING CURVES  error(N)  OF DEEPONET AND FNO
#
# For every operator, training-set size N and seed the test summary of run
# scale_<operator>_N<NNNN>_seed<seed> is read. Plotted on log-log axes:
#   mean over seeds of the test error, with the seed standard deviation.
# A straight line is fitted to log(error) against log(N):
#     log e = log C - alpha log N      ->      e = C N^(-alpha)
# alpha is the convergence rate in the number of training cases: a larger
# alpha means the operator profits more from extra data.
#
# Output: <figures>/<case_tag>/report/scaling.pdf/.png, scaling.csv,
#         scaling_rates.json
# =============================================================================

import json

import numpy as np
import matplotlib.pyplot as plt

from .style import paper_style, save_figure, COLORS, MODEL_NAMES
from .report import report_dir, write_csv_rows
from ..training.scaling_runs import scaling_run_name

OPERATORS = ("deeponet", "fno")
METRICS = [("mean_case_rmse_K", "Test RMSE, mean over cases (K)"),
           ("mean_case_relative_l2_rise", "Test relative $L_2$ of the temperature rise")]


# =============================================================================
# LEAST-SQUARES LINE THROUGH (log N, log e):  returns alpha and C
# =============================================================================
def fit_power_law(sizes, errors):
    x = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.asarray(errors, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    return float(-slope), float(np.exp(intercept))


def scaling_report(cfg, sizes, seeds):
    rows = []
    missing = []

    for operator in OPERATORS:
        for N in sizes:
            for seed in seeds:
                run_name = scaling_run_name(operator, N, seed)
                path = cfg.study_dir / run_name / "all_test_cases_grid_summary.json"
                if not path.exists():
                    missing.append(run_name)
                    continue
                summary = json.loads(path.read_text())
                row = {"operator": operator, "train_cases": N, "seed": seed}
                for metric, label in METRICS:
                    row[metric] = summary[metric]
                row["global_r2"] = summary["global_r2"]
                rows.append(row)

    if missing:
        print("Runs not finished yet (left out):", flush=True)
        for name in missing:
            print("   " + name, flush=True)
    if not rows:
        raise FileNotFoundError("No finished scaling runs found")

    folder = report_dir(cfg)
    write_csv_rows(folder / "scaling.csv", rows)

    rates = {}
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")

        for axis, (metric, label) in zip(axes, METRICS):
            for operator in OPERATORS:
                used_sizes = []
                means = []
                deviations = []
                for N in sizes:
                    values = [row[metric] for row in rows
                              if row["operator"] == operator and row["train_cases"] == N]
                    if not values:
                        continue
                    used_sizes.append(N)
                    means.append(float(np.mean(values)))
                    if len(values) > 1:
                        deviations.append(float(np.std(values, ddof=1)))
                    else:
                        deviations.append(0.0)

                if not used_sizes:
                    continue

                axis.errorbar(used_sizes, means, yerr=deviations, fmt="o",
                              color=COLORS[operator], capsize=4)

                if len(used_sizes) >= 2:
                    alpha, C = fit_power_law(used_sizes, means)
                    rates.setdefault(operator, {})[metric] = {"alpha": alpha, "C": C}
                    line_N = np.array([min(used_sizes), max(used_sizes)], dtype=float)
                    axis.plot(line_N, C * line_N ** (-alpha), "-", color=COLORS[operator],
                              label="%s, $\\alpha$ = %.2f" % (MODEL_NAMES[operator], alpha))
                else:
                    axis.plot([], [], "o", color=COLORS[operator], label=MODEL_NAMES[operator])

            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xticks(list(sizes))
            axis.set_xticklabels([str(N) for N in sizes])
            axis.set_xlabel("Training cases N")
            axis.set_ylabel(label)
            axis.legend()

        save_figure(fig, folder / "scaling")

    output = {"sizes": list(sizes), "seeds": list(seeds), "missing_runs": missing,
              "fit": "least squares of log(mean error) on log(N): error = C N^(-alpha)",
              "rates": rates}
    (folder / "scaling_rates.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(rates, indent=2), flush=True)
    return output
