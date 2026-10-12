# =============================================================================
# DATA-SIZE STUDY: TRAIN ONE MODEL ON THE FIRST N TRAINING CASES
#
# Every run of the study uses
#   - the scaling dataset (data/scaling.py) and its fixed validation/test sets
#   - the architecture and learning rate selected by tuning on this dataset
#     (the winner of the operator's last stage, "lr"); without that selection
#     the run stops with an error, so no run is ever trained on a default
#     architecture by mistake
#   - the same epochs, early stopping and scheduler for every N
# and is called  scale_<operator>_N<NNNN>_seed<seed>, e.g. scale_fno_N0800_seed321.
# After training the run is scored on the test set (not used for any choice).
# =============================================================================

from pathlib import Path

from ..analysis.metrics import evaluate
from .tuning import STAGES, tuning_dir, final_settings
from .trainer import train_model


def scaling_run_name(operator, train_size, seed):
    return "scale_%s_N%04d_seed%d" % (operator, train_size, seed)


def scaling_settings(cfg, operator, train_size, seed):
    last_stage = STAGES[operator][-1]
    selection = tuning_dir(cfg) / ("%s_%s_selected.json" % (operator, last_stage))

    if not selection.exists():
        raise FileNotFoundError(
            "No tuning selection for %s: %s\n"
            "Run the tuning stages first, ending with\n"
            "  tune / tune-select --stage-name %s  (same --family scaling --train-size 800)"
            % (operator, selection, last_stage))

    run_cfg = final_settings(cfg, operator, seed)
    print("Architecture: tuning selection %s" % selection, flush=True)
    if operator == "deeponet":
        names = ("branch_hidden", "trunk_hidden", "latent_dim", "deeponet_activation", "lr")
    else:
        names = ("fno_modes_y", "fno_modes_x", "fno_width", "fno_blocks", "fno_activation", "lr")
    for name in names:
        print("  %-20s %s" % (name, getattr(run_cfg, name)), flush=True)

    run_cfg.train_size = train_size
    run_cfg.run_name = scaling_run_name(operator, train_size, seed)
    return run_cfg


def train_scaling_run(cfg, operator, train_size, seed):
    run_cfg = scaling_settings(cfg, operator, train_size, seed)

    # A FINISHED RUN IS KEPT; AN INTERRUPTED ONE CONTINUES
    finished = Path(run_cfg.outdir) / "all_test_cases_grid_summary.json"
    if finished.exists():
        print("Finished run kept: %s" % run_cfg.run_name, flush=True)
        return
    summary = Path(run_cfg.log_outdir) / "training_summary.json"
    if not summary.exists():
        recovery = Path(run_cfg.checkpoint_outdir) / "last_checkpoint.pt"
        if recovery.exists() or Path(run_cfg.model_path).exists():
            run_cfg.resume = True
        train_model(run_cfg)

    evaluate(run_cfg, split="test")
