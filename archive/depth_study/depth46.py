"""Isolated DeepONet depth study; original FNO/POD runs remain read-only inputs."""
import argparse
import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from src.config import Settings
from src.data_protocol import sha256_file
from src.model import DEEPONET_VARIANTS, make_model, parameter_count

SEEDS = (321, 777, 2026, 31415, 27182)
RATES = (1e-3, 3e-4, 1e-4)
BASE = Path("outputs/deeponet_depth46")


def register():
    for depth in (4, 6):
        DEEPONET_VARIANTS[f"depth{depth}_gelu"] = dict(
            branch_hidden=[256]*depth, trunk_hidden=[128]*depth,
            latent=128, activation="gelu")


def config(depth):
    root = BASE / f"depth{depth}"
    return Settings(operator="deeponet", variant=f"depth{depth}_gelu",
        results_root=str(root/"study_results"), logs_root=str(root/"logs"),
        checkpoints_root=str(root/"checkpoints"), figures_root=str(root/"figures"),
        device="cuda", precision="fp32", batch_size=8, point_chunk=2048,
        cpu_threads=4)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def link(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve() != source:
            raise ValueError("Unexpected existing destination: "+str(destination))
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def verify_fixed():
    saved = json.loads((BASE/"fixed_inputs.json").read_text())
    for path, digest in saved.items():
        if sha256_file(path) != digest:
            raise ValueError("Fixed input changed: "+path)


def preflight():
    from src.ml_dataset import load_bundle
    from scripts.check_final_figures import check_inputs
    torch.set_num_threads(4)
    old = Settings(device="cpu")
    check_inputs(old, list(SEEDS))
    data = load_bundle(old)
    original_scaling = Path("outputs/figure_benchmarks")/old.case_tag/"batch_scaling.json"
    if not original_scaling.is_file():
        raise FileNotFoundError("Restore existing final-figure timings: "+str(original_scaling))
    fixed = {old.dataset_path: sha256_file(old.dataset_path)}
    for op in ("fno", "pod_deeponet"):
        for seed in SEEDS:
            run = f"final_{op}_seed{seed}"
            for root in (old.checkpoints_root, old.results_root, old.logs_root):
                source = old._root(root)/old.case_tag/run
                for file in sorted(source.rglob("*")):
                    if file.is_file(): fixed[str(file)] = sha256_file(file)
                for depth in (4, 6):
                    cfg = config(depth)
                    newroot = {old.checkpoints_root: cfg.checkpoints_root,
                               old.results_root: cfg.results_root,
                               old.logs_root: cfg.logs_root}[root]
                    link(source, cfg._root(newroot)/cfg.case_tag/run)
    selection = old._root(old.results_root)/old.case_tag/"selection"
    for file in selection.glob("pod*"):
        fixed[str(file)] = sha256_file(file)
        for depth in (4, 6):
            cfg = config(depth)
            link(file, cfg._root(cfg.results_root)/cfg.case_tag/"selection"/file.name)
    selected = json.loads((selection/"pod_deeponet_selected.json").read_text())
    bases = [old._root(selected.get("basis_file", old.pod_basis_path))]
    bases += [old._root(row["basis_file"]) for row in selected["trials"] if row.get("basis_file")]
    for basis in set(bases):
        for file in (basis, basis.with_suffix(".json")):
            fixed[str(file)] = sha256_file(file)
    for depth in (4, 6):
        cfg = config(depth)
        # Default-basis references in the existing selection still resolve.
        source = old._root(old.results_root)/old.case_tag/"pod"
        if source.exists(): link(source, cfg._root(cfg.results_root)/cfg.case_tag/"pod")
        bench = BASE/f"depth{depth}"/"figure_benchmarks"/old.case_tag
        bench.mkdir(parents=True, exist_ok=True)
        for op in ("fno", "pod_deeponet"):
            source = original_scaling.parent/f"final_{op}_seed{SEEDS[0]}"
            for file in source.rglob("*"):
                if file.is_file(): fixed[str(file)] = sha256_file(file)
            link(source, bench/source.name)
        dest = bench/"batch_scaling.json"
        if not dest.exists(): shutil.copy2(original_scaling, dest)
    fixed[str(original_scaling)] = sha256_file(original_scaling)
    if (BASE/"fixed_inputs.json").exists(): verify_fixed()
    else: write_json(BASE/"fixed_inputs.json", fixed)
    # Shape/gradient checks are inexpensive; do not train on the login node.
    for depth in (4, 6):
        model = make_model(f"depth{depth}_gelu", int(data["sdf"].shape[1]))
        assert len(model.architecture["branch_hidden"]) == depth
        assert len(model.architecture["trunk_hidden"]) == depth
        y = model(torch.zeros(1, data["sdf"].shape[1]), torch.zeros(1, 3, 3))
        assert y.shape == (1, 3) and torch.isfinite(y).all()
        y.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print(f"Depth {depth}: {parameter_count(model):,} parameters; shape/gradient checks passed")
    print("Fixed inputs verified. Only DeepONet will be trained.")


def trial_name(depth, rate_index):
    return f"tune_deeponet_depth{depth}_lr{rate_index}_seed123"


def train(cfg):
    from scripts.train_operator import train_model
    summary = Path(cfg.log_outdir)/"training_summary.json"
    if summary.exists():
        info = json.loads(summary.read_text())
        if info["architecture"] != make_model(cfg.variant, 12288).architecture or \
           info["settings"]["lr"] != cfg.lr or info["seed"] != cfg.train_seed:
            raise ValueError("Completed run settings mismatch")
        print("Completed run retained:", cfg.run_name); return
    cfg.resume = (Path(cfg.checkpoint_outdir)/"last_checkpoint.pt").exists() or Path(cfg.model_path).exists()
    train_model(cfg)


def tune(task):
    verify_fixed()
    depth = (4, 6)[task//3]; ri = task%3
    train(replace(config(depth), lr=RATES[ri], epochs=2000, train_seed=123,
                  run_name=trial_name(depth, ri)))


def select():
    verify_fixed(); rows = []; winners = {}
    for depth in (4, 6):
        cfg = config(depth); trials = []
        for ri, rate in enumerate(RATES):
            run = trial_name(depth, ri)
            summary = json.loads((cfg._root(cfg.logs_root)/cfg.case_tag/run/"training_summary.json").read_text())
            if summary["dataset_sha256"] != sha256_file(cfg.dataset_path) or \
               summary["test_used_for_selection"] or summary["seed"] != 123 or \
               summary["settings"]["lr"] != rate:
                raise ValueError("Invalid tuning summary: "+run)
            row = dict(depth=depth, variant=cfg.variant, lr=rate, run_name=run,
                       validation_rmse_K=summary["best_val_rmse_K"],
                       best_epoch=summary["best_epoch"], parameters=summary["parameters"])
            trials.append(row); rows.append(row)
        winner = min(trials, key=lambda r:r["validation_rmse_K"])
        winners[str(depth)] = winner
        write_json(cfg._root(cfg.results_root)/cfg.case_tag/"selection/deeponet_depth_selected.json",
                   dict(winner=winner, trials=trials, test_used=False))
    selected = min(rows, key=lambda r:r["validation_rmse_K"])
    write_json(BASE/"selection.json", dict(winners=winners, reporting_depth=selected["depth"],
               trials=rows, test_used=False))
    export_csv(BASE/"depth_tuning.csv", rows)
    print(json.dumps(dict(winners=winners, reporting_depth=selected["depth"]), indent=2))


def selected_cfg(task):
    depth = (4, 6)[task//5]; seed = SEEDS[task%5]
    selection = json.loads((BASE/"selection.json").read_text())
    return replace(config(depth), lr=selection["winners"][str(depth)]["lr"],
                   train_seed=seed, epochs=5000, run_name=f"final_deeponet_seed{seed}")


def evaluate(task):
    from src.evaluation import evaluate as evaluate_model
    verify_fixed(); evaluate_model(selected_cfg(task)); verify_fixed()


def export_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def reports():
    from scripts.final_figures import run_suite, compute_plots
    from scripts.check_final_figures import check_inputs
    from scripts.depth46_benchmark import benchmark_scaling
    from src.paper_style import paper_style, save_paper_figure
    import matplotlib.pyplot as plt
    all_rows = []
    selection = json.loads((BASE/"selection.json").read_text())
    for depth in (4, 6):
        verify_fixed(); cfg = config(depth)
        records = check_inputs(cfg, list(SEEDS))
        benchroot = str(BASE/f"depth{depth}"/"figure_benchmarks")
        old = json.loads((Path(benchroot)/cfg.case_tag/"batch_scaling.json").read_text())
        benchmark_scaling(cfg, list(SEEDS), old["batch_sizes"], benchmark_root=benchroot)
        run_suite(cfg, list(SEEDS), skip_timing=True, force=True)
        output = cfg._root(cfg.figures_root)/cfg.case_tag/"paper_final"
        try:
            compute_plots(cfg, records, output, benchroot)
        except ValueError as exc:
            # Preserve provenance rather than weakening matched-hardware checks.
            if str(exc) != "Timing comparison protocols differ": raise
            (output/"TIMING_COMPARISON_NOTE.txt").write_text(
                "FNO/POD timings are retained from the previous session. New DeepONet hardware differs. "
                "Graphs are labelled as separate hardware sessions; see recorded_timings.csv.\n"+str(exc))
            recorded_timing_plots(cfg, records, output, benchroot)
        runtime_rows = []
        for op, runs in records.items():
            for run in runs:
                info = json.loads((cfg._root(cfg.logs_root)/cfg.case_tag/run["run"]/"training_summary.json").read_text())
                row = dict(depth=depth, operator=op, seed=run["seed"],
                    mean_case_rmse_K=run["summary"]["mean_case_rmse_K"],
                    point_weighted_rmse_K=run["summary"]["global_point_weighted_rmse_K"],
                    parameters=info["parameters"], best_epoch=info["best_epoch"],
                    epochs_completed=info["epochs_completed"])
                all_rows.append(row)
            b = json.loads((Path(benchroot)/cfg.case_tag/runs[0]["run"]/"benchmarks/benchmark_summary.json").read_text())
            runtime_rows.append(dict(operator=op, epoch_median_s=b["training_epoch"]["median_s"],
                inference_median_ms=1000*b["inference_end_to_end_median_s"],
                session="new" if op=="deeponet" else "retained", hardware=json.dumps(b["hardware"])))
        export_csv(output/"recorded_timings.csv", runtime_rows)
        timing_tex = [r"\begin{tabular}{lrrl}", r"\toprule", r"Model & Epoch (s) & Inference (ms/case) & Session\\", r"\midrule"]
        for row in runtime_rows:
            timing_tex.append(f"{row['operator'].replace('_', '-')} & {row['epoch_median_s']:.3f} & {row['inference_median_ms']:.3f} & {row['session']}"+r"\\")
        timing_tex += [r"\bottomrule", r"\end{tabular}"]
        (output/"recorded_timings.tex").write_text("\n".join(timing_tex))
        verify_fixed()
    export_csv(BASE/"all_seed_results.csv", all_rows)
    aggregate = []
    for depth in (4, 6):
        for op in ("deeponet", "fno", "pod_deeponet"):
            rows = [r for r in all_rows if r["depth"]==depth and r["operator"]==op]
            values = np.array([r["mean_case_rmse_K"] for r in rows])
            points = np.array([r["point_weighted_rmse_K"] for r in rows])
            aggregate.append(dict(depth=depth, operator=op, mean_case_rmse_K=float(values.mean()),
                mean_case_sd_K=float(values.std(ddof=1)), point_weighted_rmse_K=float(points.mean()),
                point_weighted_sd_K=float(points.std(ddof=1)), parameters=rows[0]["parameters"]))
    export_csv(BASE/"comparison.csv", aggregate)
    tex = [r"\begin{tabular}{llrrr}", r"\toprule", r"Depth & Model & Mean-case RMSE (K) & Point-weighted RMSE (K) & Parameters\\", r"\midrule"]
    names={"deeponet":"DeepONet", "fno":"FNO", "pod_deeponet":"POD-DeepONet"}
    for r in aggregate:
        tex.append(f"{r['depth']} & {names[r['operator']]} & ${r['mean_case_rmse_K']:.3f}\\pm{r['mean_case_sd_K']:.3f}$ & ${r['point_weighted_rmse_K']:.3f}\\pm{r['point_weighted_sd_K']:.3f}$ & {r['parameters']}"+r"\\")
    tex += [r"\bottomrule",r"\end{tabular}"]
    (BASE/"comparison.tex").write_text("\n".join(tex))
    tuning_tex = [r"\begin{tabular}{lrrr}",r"\toprule",r"Hidden layers & Initial LR & Validation RMSE (K) & Best epoch\\",r"\midrule"]
    for row in selection["trials"]:
        tuning_tex.append(f"{row['depth']} & {row['lr']:.0e} & {row['validation_rmse_K']:.4f} & {row['best_epoch']}"+r"\\")
    tuning_tex += [r"\bottomrule",r"\end{tabular}"]
    (BASE/"depth_tuning.tex").write_text("\n".join(tuning_tex))
    with paper_style():
        fig,ax=plt.subplots(figsize=(6,4),layout="constrained")
        trials=selection["trials"]
        for rate in RATES:
            rr=sorted([r for r in trials if r["lr"]==rate],key=lambda r:r["depth"])
            ax.plot([r["depth"] for r in rr],[r["validation_rmse_K"] for r in rr],marker="o",label=f"LR = {rate:.0e}")
        ax.set(xlabel="Hidden layers per network",ylabel="Validation RMSE (K)",xticks=[4,6]);ax.legend()
        save_paper_figure(fig,BASE/"deeponet_depth_tuning")
    (BASE/"REPORTING_DEPTH.txt").write_text(
        f"Use depth {selection['reporting_depth']} for the main comparison, selected BEFORE test evaluation by validation RMSE. Both depth studies remain in the appendix.\n")
    verify_fixed(); print("Complete:", BASE.resolve())


def recorded_timing_plots(cfg, records, output, benchroot):
    """Different hardware sessions remain explicitly labelled; never claim matching."""
    from src.paper_style import paper_style, save_paper_figure, MODEL_NAMES, COLORS
    import matplotlib.pyplot as plt
    root=Path(benchroot)/cfg.case_tag
    summaries=[json.loads((root/runs[0]["run"]/"benchmarks/benchmark_summary.json").read_text())
               for runs in records.values()]
    labels=[MODEL_NAMES[op]+(" (new)" if op=="deeponet" else " (retained)") for op in records]
    with paper_style():
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
        for i,b in enumerate(summaries):
            axes[0].bar(i,b["training_epoch"]["median_s"],fill=False,edgecolor=COLORS[i])
            axes[1].bar(i,1000*b["inference_end_to_end_median_s"],fill=False,edgecolor=COLORS[i])
        for ax,ylabel in zip(axes,("Time per epoch (s)","Inference time (ms/case)")):
            ax.set_xticks(range(len(labels)),labels,rotation=15);ax.set_ylabel(ylabel)
        fig.suptitle("Recorded timings: separate hardware sessions")
        save_paper_figure(fig,output/"10_timing")
    scaling=json.loads((root/"batch_scaling.json").read_text())
    with paper_style():
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
        for i,(op,label) in enumerate(zip(records,labels)):
            rows=[r for r in scaling["operators"][op]["rows"] if r["status"]=="completed"]
            batch=np.array([r["batch_size"] for r in rows]);lat=np.array([r["median_s"] for r in rows])
            for ax,values in zip(axes,(lat*1000,batch/lat)):
                ax.plot(batch,values,marker="o",color=COLORS[i],label=label)
        for ax,ylabel in zip(axes,("Batch latency (ms)","Throughput (cases/s)")):
            ax.set(xlabel="Batch size",ylabel=ylabel,xscale="log");ax.legend()
        fig.suptitle("Recorded batch scaling: separate hardware sessions")
        save_paper_figure(fig,output/"16_batch_scaling")
    with paper_style():
        fig,ax=plt.subplots(figsize=(6,4),layout="constrained")
        for i,((op,runs),b,label) in enumerate(zip(records.items(),summaries,labels)):
            ax.scatter(1000*b["inference_end_to_end_median_s"],runs[0]["summary"]["mean_case_rmse_K"],color=COLORS[i],label=label)
        ax.set(xlabel="Inference latency (ms/case)",ylabel="Test RMSE (K)",yscale="log");ax.legend()
        ax.set_title("Recorded timings: separate hardware sessions")
        save_paper_figure(fig,output/"17_accuracy_latency")


def main():
    register()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["preflight","tune","select","final","evaluate","reports"])
    parser.add_argument("--task",type=int,default=0); args=parser.parse_args()
    if args.stage=="preflight": preflight()
    elif args.stage=="tune":
        if not 0<=args.task<6: raise ValueError("Tuning task must be 0..5")
        tune(args.task)
    elif args.stage=="select": select()
    elif args.stage=="final":
        if not 0<=args.task<10: raise ValueError("Final task must be 0..9")
        verify_fixed(); train(selected_cfg(args.task))
    elif args.stage=="evaluate":
        if not 0<=args.task<10: raise ValueError("Evaluation task must be 0..9")
        evaluate(args.task)
    else: reports()


if __name__=="__main__": main()
