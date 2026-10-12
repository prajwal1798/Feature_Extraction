# =============================================================================
# RUN: THE ONE COMMAND-LINE ENTRY POINT OF THE PROJECT
#
#     python run.py <stage> [options]
#
# DATA
#   generate          FE dataset (data/generate.py)           --family --cases
#   generate-rounded  external rounded-rectangle test set     --rounded-cases
#   prepare-legacy    convert an old dataset to protocol 2
#   preview           solve and plot one fixed case
#   mesh-check        FE mesh convergence and heat balance
#   plot-sampling     sampling-space figure of the dataset
#
# ONE MODEL
#   train             train one run           --operator --variant --seed --run-name
#   evaluate          score a run on the test split (or --split val)
#   evaluate-rounded  score a run on the rounded-rectangle set
#   plot              loss history + one test case (--test-case)
#   benchmark         timings of a finished run
#
# TUNING (training/tuning.py)
#   tune              train the cells of a stage   --operator --stage-name [--cell k] --seeds
#                     (inside a SLURM array job the cell is the array task number)
#   tune-select       collect a stage, choose the winner, draw its figure
#   train-final       train the selected model     --operator --seed
#
# DATA-SIZE STUDY (data/scaling.py, training/scaling_runs.py)
#   generate --family scaling     Halton training pool + fixed val/test sets
#   train-scaling     train on the first N cases   --operator --sizes --seeds
#                     (or --train-size N --seed s; in an array job, task k =
#                      k-th combination of size and seed)
#   scaling-report    log-log learning curves and convergence rates
#   compare --family scaling --train-size N   full DeepONet-FNO report at size N
#
# STUDY
#   summarize         seed summary of the final runs   --operator --seeds
#   compare           DeepONet vs FNO report           --seeds
#   gpu-info          hardware record
#   smoke             tiny end-to-end check of all stages (minutes, CPU allowed)
#
# Data generation imports NGSolve but not PyTorch; learning stages import
# PyTorch. Imports are therefore made inside each stage.
# =============================================================================

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.config import Settings, PROJECT_ROOT

STAGES = ["generate", "generate-rounded", "prepare-legacy", "preview", "mesh-check",
          "plot-sampling", "train", "evaluate", "evaluate-rounded", "plot", "benchmark",
          "tune", "tune-select", "train-final", "summarize", "compare", "gpu-info", "smoke",
          "train-scaling", "scaling-report"]

SCALING_STAGES = ["train-scaling", "scaling-report"]
DEFAULT_SEEDS = "321,777,2026,31415,27182"
DEFAULT_SCALING_SEEDS = DEFAULT_SEEDS


def read_arguments():
    parser = argparse.ArgumentParser(description="DeepONet / FNO heat-conduction study")
    parser.add_argument("stage", choices=STAGES)

    # DATA
    parser.add_argument("--family", choices=["circle", "rect", "combined", "scaling"],
                        default="combined")
    parser.add_argument("--cases", type=int, default=None,
                        help="dataset size (default 500; 3200 for --family scaling)")
    parser.add_argument("--train-size", type=int, default=0,
                        help="use only the first N training cases (0 = all)")
    parser.add_argument("--sizes", default=None, help="training sizes, e.g. 200,400,800")
    parser.add_argument("--val-cases", type=int, default=200, help="scaling study: validation cases")
    parser.add_argument("--test-cases", type=int, default=400, help="scaling study: test cases")
    parser.add_argument("--rounded-cases", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="regenerate even if data exist")
    parser.add_argument("--small-grid", action="store_true",
                        help="20 cases on a 16 x 12 grid in verification/ (tests only)")

    # MODEL AND TRAINING
    parser.add_argument("--operator", choices=["deeponet", "fno"], default="deeponet")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", default=None, help="comma-separated, e.g. " + DEFAULT_SEEDS)
    parser.add_argument("--run-name")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--point-chunk", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--early-stop", type=int, default=800)
    parser.add_argument("--scheduler-patience", type=int, default=200)
    parser.add_argument("--resume", action="store_true")

    # EVALUATION AND TUNING
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--test-case", type=int, default=0)
    parser.add_argument("--stage-name", dest="tuning_stage", default=None,
                        help="tuning stage: deeponet grid|lr, fno modes_width|depth|lr")
    parser.add_argument("--cell", type=int, default=None, help="one cell of the stage (array task)")

    # BENCHMARK
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument("--benchmark-epochs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    return parser.parse_args()


def make_settings(arguments):
    cfg = Settings()
    family = arguments.family
    if arguments.stage in SCALING_STAGES:
        family = "scaling"
    cases = arguments.cases
    if cases is None:
        cases = 3200 if family == "scaling" else 500

    cfg.dataset_family = family
    cfg.n_cases = cases
    cfg.selected_dataset_size = cases
    cfg.train_size = arguments.train_size
    cfg.scaling_val_cases = arguments.val_cases
    cfg.scaling_test_cases = arguments.test_cases
    cfg.rounded_test_n_cases = arguments.rounded_cases
    cfg.force_generation = arguments.force

    cfg.operator = arguments.operator
    cfg.variant = arguments.variant
    cfg.epochs = arguments.epochs
    cfg.batch_size = arguments.batch_size
    cfg.lr = arguments.lr
    cfg.train_seed = arguments.seed
    cfg.device = arguments.device
    cfg.point_chunk = arguments.point_chunk
    cfg.cpu_threads = arguments.cpu_threads
    cfg.checkpoint_every = arguments.checkpoint_every
    cfg.early_stop_patience = arguments.early_stop
    cfg.scheduler_patience = arguments.scheduler_patience
    cfg.resume = arguments.resume
    cfg.test_case_number = arguments.test_case
    cfg.rounded_test_case_number = arguments.test_case

    if arguments.small_grid:
        if arguments.cases is None:
            cfg.n_cases = 20
            cfg.selected_dataset_size = 20
        cfg.Nx = 16
        cfg.Ny = 12
        cfg.maxh = 0.0012
        cfg.data_root = "verification/data"
        cfg.results_root = "verification/outputs/study_results"
        cfg.checkpoints_root = "verification/outputs/checkpoints"
        cfg.logs_root = "verification/outputs/logs"
        cfg.figures_root = "verification/outputs/figures"

    if arguments.run_name:
        cfg.run_name = arguments.run_name
    else:
        cfg.run_name = "%s_%s_seed%d" % (cfg.operator, cfg.variant, cfg.train_seed)

    # A RUN NAME BECOMES A FOLDER NAME
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    for character in cfg.run_name:
        if character not in allowed:
            raise ValueError("Run name must be a plain directory name")

    if min(cfg.epochs, cfg.batch_size, cfg.point_chunk, cfg.cpu_threads, cfg.checkpoint_every,
           cfg.early_stop_patience, cfg.n_cases) < 1:
        raise ValueError("Counts must be positive")
    if cfg.lr <= 0:
        raise ValueError("Learning rate must be positive")
    return cfg


def seed_list(arguments, default):
    text = arguments.seeds if arguments.seeds else default
    seeds = []
    for value in text.split(","):
        seeds.append(int(value))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct")
    return seeds


def size_list(arguments, cfg):
    if not arguments.sizes:
        return list(cfg.scaling_sizes)
    sizes = []
    for value in arguments.sizes.split(","):
        sizes.append(int(value))
    return sizes


def default_tuning_stage(arguments):
    if arguments.tuning_stage:
        return arguments.tuning_stage
    if arguments.operator == "deeponet":
        return "grid"
    return "modes_width"


# =============================================================================
# SMOKE TEST: EVERY STAGE ON 20 TINY CASES, 2 EPOCHS
# =============================================================================
def smoke(cfg):
    tiny = ["--small-grid", "--device", cfg.device, "--epochs", "2", "--batch-size", "2",
            "--point-chunk", "64", "--cpu-threads", "2"]
    commands = [
        ["generate", "--small-grid"],
        ["generate-rounded", "--small-grid", "--rounded-cases", "2"],
        ["plot-sampling", "--small-grid"],
    ]
    for operator in ("deeponet", "fno"):
        shared = tiny + ["--operator", operator]
        commands.append(["train"] + shared)
        commands.append(["evaluate"] + shared)
        commands.append(["evaluate-rounded"] + shared)
        commands.append(["plot"] + shared)
        commands.append(["benchmark"] + shared + ["--benchmark-repeats", "2",
                                                  "--benchmark-epochs", "1", "--warmups", "1"])
    for command in commands:
        print("\n>>> run.py " + " ".join(command), flush=True)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "run.py")] + command, check=True)
    print("\nSmoke test passed.")


def main():
    arguments = read_arguments()
    cfg = make_settings(arguments)
    stage = arguments.stage

    # ---------------- DATA ----------------
    if stage == "generate":
        if Path(cfg.dataset_path).exists() and not cfg.force_generation:
            print("Reusing dataset:", cfg.dataset_path)
        elif cfg.dataset_family == "scaling":
            from src.data.scaling import generate_scaling_dataset
            generate_scaling_dataset(cfg)
        else:
            from src.data.generate import generate_dataset
            generate_dataset(cfg)

    elif stage == "generate-rounded":
        from src.data.external import generate_rounded_rect_test_dataset
        generate_rounded_rect_test_dataset(cfg)

    elif stage == "prepare-legacy":
        from src.data.protocol import prepare_legacy
        prepare_legacy(cfg)

    elif stage == "preview":
        from src.data.generate import run_preview
        cfg.run_name = "preview"
        run_preview(cfg)

    elif stage == "mesh-check":
        from src.analysis.mesh_check import run_mesh_check
        run_mesh_check(cfg)

    elif stage == "plot-sampling":
        from src.data.dataset import load_bundle
        from src.analysis.plots import plot_sampling_space
        plot_sampling_space(cfg, load_bundle(cfg))

    # ---------------- ONE MODEL ----------------
    elif stage == "train":
        from src.training.trainer import train_model
        from src.analysis.benchmark import hardware_report
        train_model(cfg)
        Path(cfg.log_outdir, "hardware.json").write_text(json.dumps(hardware_report(), indent=2))

    elif stage == "evaluate":
        from src.analysis.metrics import evaluate
        evaluate(cfg, split=arguments.split)

    elif stage == "evaluate-rounded":
        from src.analysis.metrics import evaluate
        evaluate(cfg, external=True)

    elif stage == "plot":
        from src.analysis.plots import plot_loss
        from src.analysis.metrics import evaluate
        plot_loss(cfg)
        evaluate(cfg, split=arguments.split, selected_only=True, make_plots=True)

    elif stage == "benchmark":
        from src.analysis.benchmark import benchmark_model
        benchmark_model(cfg, repeats=arguments.benchmark_repeats, warmups=arguments.warmups,
                        epoch_repeats=arguments.benchmark_epochs)

    # ---------------- TUNING ----------------
    elif stage == "tune":
        from src.training.tuning import run_stage
        cell = arguments.cell
        if cell is None and "SLURM_ARRAY_TASK_ID" in os.environ:
            cell = int(os.environ["SLURM_ARRAY_TASK_ID"])
        run_stage(cfg, cfg.operator, default_tuning_stage(arguments),
                  seed_list(arguments, "123"), index=cell)

    elif stage == "tune-select":
        from src.training.tuning import select_stage
        from src.analysis.report import plot_tuning
        tuning_stage = default_tuning_stage(arguments)
        select_stage(cfg, cfg.operator, tuning_stage, seed_list(arguments, "123"))
        plot_tuning(cfg, cfg.operator, tuning_stage)

    elif stage == "train-final":
        from src.training.tuning import final_settings
        from src.training.trainer import train_model
        from src.analysis.benchmark import hardware_report
        final_cfg = final_settings(cfg, cfg.operator, cfg.train_seed)
        if arguments.run_name:
            final_cfg.run_name = arguments.run_name
        train_model(final_cfg)
        Path(final_cfg.log_outdir, "hardware.json").write_text(json.dumps(hardware_report(), indent=2))

    # ---------------- STUDY ----------------
    elif stage == "summarize":
        from src.analysis.report import summarize_seeds
        if cfg.dataset_family == "scaling":
            if cfg.train_size == 0:
                cfg.train_size = max(cfg.scaling_sizes)
            summarize_seeds(cfg, cfg.operator, seed_list(arguments, DEFAULT_SCALING_SEEDS))
        else:
            summarize_seeds(cfg, cfg.operator, seed_list(arguments, DEFAULT_SEEDS))

    elif stage == "compare":
        from src.analysis.report import compare
        if cfg.dataset_family == "scaling":
            if cfg.train_size == 0:
                cfg.train_size = max(cfg.scaling_sizes)
            compare(cfg, seed_list(arguments, DEFAULT_SCALING_SEEDS))
        else:
            compare(cfg, seed_list(arguments, DEFAULT_SEEDS))

    elif stage == "gpu-info":
        from src.training.runtime import configure_device
        from src.analysis.benchmark import hardware_report
        configure_device(cfg)
        report = hardware_report()
        cfg.study_dir.mkdir(parents=True, exist_ok=True)
        (cfg.study_dir / "hardware_preflight.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))

    elif stage == "smoke":
        smoke(cfg)

    # ---------------- DATA-SIZE STUDY ----------------
    elif stage == "train-scaling":
        from src.training.scaling_runs import train_scaling_run
        combinations = []
        if cfg.train_size > 0:
            combinations.append((cfg.train_size, cfg.train_seed))
        else:
            for N in size_list(arguments, cfg):
                for seed in seed_list(arguments, DEFAULT_SCALING_SEEDS):
                    combinations.append((N, seed))
            if "SLURM_ARRAY_TASK_ID" in os.environ:
                task = int(os.environ["SLURM_ARRAY_TASK_ID"])
                combinations = [combinations[task]]
        for N, seed in combinations:
            train_scaling_run(cfg, cfg.operator, N, seed)

    elif stage == "scaling-report":
        from src.analysis.scaling_report import scaling_report
        scaling_report(cfg, size_list(arguments, cfg), seed_list(arguments, DEFAULT_SCALING_SEEDS))


if __name__ == "__main__":
    main()
