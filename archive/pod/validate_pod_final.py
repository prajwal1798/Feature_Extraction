"""Refuse to reuse a finished POD run that belongs to another selection."""
import argparse
import json
from pathlib import Path
from src.config import Settings
from src.data_protocol import sha256_file


def validate_final(cfg):
    folder = cfg._root(cfg.results_root)/cfg.case_tag/"selection"
    selection = json.loads((folder/"pod_deeponet_selected.json").read_text())
    result = json.loads((Path(cfg.log_outdir)/"training_summary.json").read_text())
    winner = selection["winner"]
    if (result["operator"] != "pod_deeponet" or result["variant"] != winner["variant"]
            or result["settings"]["pod_modes"] != winner["pod_modes"]
            or result["settings"]["lr"] != selection["initial_learning_rate"]
            or result["settings"]["train_seed"] != cfg.train_seed
            or result["pod_metadata"]["basis_sha256"] != selection["basis_sha256"]
            or result["dataset_sha256"] != selection["dataset_sha256"]
            or result["dataset_sha256"] != sha256_file(cfg.dataset_path)
            or result["test_used_for_selection"] is not False
            or result["settings"]["epochs"] != 5000
            or not Path(cfg.model_path).is_file()):
        raise ValueError("Existing final run differs from selected protocol. Preserve it under a separate name before rerunning: " + cfg.run_name)
    print("Validated finished final run:", cfg.run_name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    validate_final(Settings(operator="pod_deeponet", run_name=args.run_name, train_seed=args.seed))
