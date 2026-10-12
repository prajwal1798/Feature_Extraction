#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/slurm
existing_tune="${1:-}"
if [ -n "$existing_tune" ] && [[ ! "$existing_tune" =~ ^[0-9]+$ ]]; then
    echo "Expected original tuning array ID as first argument" >&2; exit 1
fi
PYTHON="/scratch/${USER}/envs/operator_learning/bin/python"
"${PYTHON}" -u -m scripts.depth4 preflight
record=outputs/slurm/deeponet_depth4.jobs
if ! (set -o noclobber; : > "$record") 2>/dev/null; then
    echo "Submission record already exists: $record. Inspect before resubmission." >&2; exit 1
fi
submit() {
    local id
    id=$(sbatch --parsable "$@"); id=${id%%;*}; echo "$id"
}
if [ -n "$existing_tune" ]; then
    tune="$existing_tune"
    dependency="afterok:${tune}_0:${tune}_1:${tune}_2"
    echo "Reusing running/completed 4-layer tuning tasks ${tune}_0, ${tune}_1, ${tune}_2"
else
    tune=$(submit jobs/30_dn4_tune.sbatch)
    dependency="afterok:$tune"
fi
echo "tune $tune" >> "$record"
select=$(submit --dependency="$dependency" jobs/26_dn4_select.sbatch)
echo "select $select" >> "$record"
final=$(submit --dependency="afterok:$select" jobs/27_dn4_final.sbatch)
echo "final $final" >> "$record"
evaluate=$(submit --dependency="afterok:$final" jobs/28_dn4_eval.sbatch)
echo "evaluate $evaluate" >> "$record"
reports=$(submit --dependency="afterok:$evaluate" jobs/29_dn4_reports.sbatch)
echo "reports $reports" >> "$record"
cat "$record"
echo "Start estimates: squeue --start -j $select,$final,$evaluate,$reports"
echo "Live final: tail -n 30 -F outputs/slurm/dn4_final_${final}_0.out"
echo "Live figures: tail -n 30 -F outputs/slurm/dn4_reports_${reports}.out"
echo "Figures: outputs/deeponet_depth46/depth4/figures/combined_0500/paper_final/"
