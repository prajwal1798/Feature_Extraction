"""Report the isolated rank check, and verify nested basis consistency."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def check_rank(original, extended, requested):
    old = torch.load(original, map_location="cpu", weights_only=True)
    new = torch.load(extended, map_location="cpu", weights_only=True)
    if old["dataset_sha256"] != new["dataset_sha256"] or not torch.equal(old["train_idx"], new["train_idx"]):
        raise ValueError("Rank caches differ in dataset or training split")
    if new["retained_modes"] < requested:
        raise ValueError(f"Only {new['retained_modes']} nonzero modes are available; requested {requested}")
    np.testing.assert_allclose(old["mean"].numpy(), new["mean"].numpy(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(old["basis"].numpy(), new["basis"][:, :old["retained_modes"]].numpy(), rtol=2e-5, atol=2e-5)
    diagnostics = json.loads(Path(extended).with_suffix(".json").read_text())
    print("Modes  Validation oracle RMSE (K)", flush=True)
    for row in diagnostics["projection"]:
        if row["split"] == "val":
            print(f"{row['modes']:5d}  {row['point_weighted_oracle_rmse_K']:.4f}", flush=True)
    print("Test set unused. Smaller-rank basis prefixes agree with the original cache.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--original", default="outputs/study_results/combined_0500/pod/pod_basis.pt")
    args = parser.parse_args()
    check_rank(args.original, args.basis, args.rank)
