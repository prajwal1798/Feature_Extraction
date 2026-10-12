#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/slurm
record="outputs/slurm/deeponet_depth4_single_results_$(date +%Y%m%d_%H%M%S).jobs"
eval_id=$(sbatch --parsable --array=0 jobs/28_dn4_eval.sbatch)
eval_id=${eval_id%%;*}
echo "evaluate $eval_id" > "$record"
plots_id=$(sbatch --parsable --dependency="afterok:$eval_id" jobs/31_dn4_single_plots.sbatch)
plots_id=${plots_id%%;*}
echo "plots $plots_id" >> "$record"
cat "$record"
echo "Live evaluation: tail -n 30 -F outputs/slurm/dn4_eval_${eval_id}_0.out"
echo "Live plots: tail -n 30 -F outputs/slurm/dn4_single_plots_${plots_id}.out"
echo "Figures: outputs/deeponet_depth46/depth4/single_seed_figures/combined_0500/paper_final/"
echo "Tables: outputs/deeponet_depth46/single_seed_report/"
