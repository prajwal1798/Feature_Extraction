#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/slurm
PYTHON="/scratch/${USER}/envs/operator_learning/bin/python"
"${PYTHON}" -u -m scripts.depth46 preflight
record=outputs/slurm/deeponet_depth46.jobs
if ! (set -o noclobber; : > "$record") 2>/dev/null; then
    echo "Submission record exists: $record. Inspect before any resubmission." >&2
    exit 1
fi
submit() {
    local id
    id=$(sbatch --parsable "$@"); id=${id%%;*}
    echo "$id"
}
tune=$(submit jobs/21_dn46_tune.sbatch)
echo "tune $tune" >> "$record"
select=$(submit --dependency="afterok:$tune" jobs/22_dn46_select.sbatch)
echo "select $select" >> "$record"
final=$(submit --dependency="afterok:$select" jobs/23_dn46_final.sbatch)
echo "final $final" >> "$record"
evaluate=$(submit --dependency="afterok:$final" jobs/24_dn46_eval.sbatch)
echo "evaluate $evaluate" >> "$record"
reports=$(submit --dependency="afterok:$evaluate" jobs/25_dn46_reports.sbatch)
echo "reports $reports" >> "$record"
cat "$record"
echo "Start estimates: squeue --start -j $tune,$select,$final,$evaluate,$reports"
echo "Live tuning: tail -n 30 -F outputs/slurm/dn46_tune_${tune}_0.out"
echo "Live final: tail -n 30 -F outputs/slurm/dn46_final_${final}_0.out"
echo "Live figures: tail -n 30 -F outputs/slurm/dn46_reports_${reports}.out"
echo "Results: outputs/deeponet_depth46/"
