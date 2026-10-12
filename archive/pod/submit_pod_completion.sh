#!/bin/bash
# Run from /scratch/$USER/operator_learning after unzipping the update.
# Usage: bash scripts/submit_pod_completion.sh [check|finish|all] [rank]
set -euo pipefail
stage="${1:-all}"
rank="${2:-256}"
if [[ ! "$rank" =~ ^[0-9]+$ ]] || (( rank <= 128 )); then
    echo "Rank must be an integer above 128" >&2; exit 1
fi
if [[ "$stage" != check && "$stage" != finish && "$stage" != all ]]; then
    echo "Use check, finish, or all" >&2; exit 1
fi
mkdir -p outputs/slurm
test -f data/processed/combined_0500/protocol_v2/dataset.npz
test -f outputs/study_results/combined_0500/pod/pod_basis.pt
for modes in 32 64 128; do
    for variant in baseline activation_gelu; do
        for lr in 3e-4 1e-4; do
            test -f "outputs/logs/combined_0500/tune_pod_m${modes}_${variant}_lr${lr}_seed123/training_summary.json"
        done
    done
done
# Fail before submission if the existing baseline results/checkpoints are missing.
if [[ "$stage" != check ]]; then
    for operator in fno deeponet; do
        for seed in 321 777 2026 31415 27182; do
            run="final_${operator}_seed${seed}"
            for file in "outputs/study_results/combined_0500/${run}/all_test_cases_grid_summary.json" \
                        "outputs/study_results/combined_0500/${run}/all_test_cases_grid_metrics.csv" \
                        "outputs/logs/combined_0500/${run}/loss_history.csv" \
                        "outputs/checkpoints/combined_0500/${run}/best_model.pt"; do
                if [[ ! -f "$file" ]]; then
                    echo "Missing baseline file: $file. Use 'check' to run the rank study first." >&2
                    exit 1
                fi
            done
        done
    done
fi
record="outputs/slurm/pod_completion_${stage}_rank${rank}.jobs"
if [[ -f "$record" ]]; then
    echo "A submission record already exists: $record. Inspect it and squeue before resubmitting." >&2
    exit 1
fi
touch "$record"
submit() {
    local label="$1"; shift
    local id
    id=$(sbatch --parsable --export="ALL,POD_RANK=${rank}" "$@")
    id="${id%%;*}"
    echo "${label} ${id}" >> "$record"
    echo "${label}: ${id}" >&2
    echo "$id"
}
selection_id=""
if [[ "$stage" != finish ]]; then
    prepare_id=$(submit rank_prepare jobs/11_pod_rank_prepare.sbatch)
    tune_id=$(submit rank_tune --dependency="afterok:${prepare_id}" jobs/12_pod_rank_tune.sbatch)
    selection_id=$(submit selection --dependency="afterok:${tune_id}" jobs/13_pod_rank_select.sbatch)
else
    test -f outputs/study_results/combined_0500/selection/pod_deeponet_selected.json
fi
if [[ "$stage" != check ]]; then
    dependency=()
    if [[ -n "$selection_id" ]]; then dependency=(--dependency="afterok:${selection_id}"); fi
    final_id=$(submit final "${dependency[@]}" jobs/09_pod_final.sbatch)
    evaluate_id=$(submit evaluate --dependency="afterok:${final_id}" jobs/10_pod_evaluate.sbatch)
    benchmark_id=$(submit benchmark --dependency="afterok:${evaluate_id}" jobs/15_study_benchmark.sbatch)
    submit paper_plots --dependency="afterok:${benchmark_id}" jobs/14_paper_plots.sbatch > /dev/null
fi
echo "Submission record: $record"
echo 'Monitor: squeue -u "$USER" -r'
echo 'Live logs: tail -n 30 -F outputs/slurm/pod_rank_prepare_<JOBID>.out'
