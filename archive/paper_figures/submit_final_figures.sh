#!/bin/bash
# Run from the SUNBIRD project root, with its conda environment active.
set -euo pipefail
mkdir -p outputs/slurm
python -m scripts.check_final_figures
record=outputs/slurm/final_figures.jobs
# Atomic creation prevents a second submission from the same update.
if ! (set -o noclobber; : > "$record") 2>/dev/null; then
    echo "Submission record exists: $record. Inspect its job IDs before rerunning." >&2
    exit 1
fi
benchmark_id=$(sbatch --parsable jobs/16_figure_benchmark.sbatch)
benchmark_id=${benchmark_id%%;*}
echo "benchmark $benchmark_id" >> "$record"
figures_id=$(sbatch --parsable --dependency="afterok:${benchmark_id}" jobs/17_final_figures.sbatch)
figures_id=${figures_id%%;*}
echo "figures $figures_id" >> "$record"
echo "Benchmark: $benchmark_id; figures: $figures_id"
echo "Submission record: $record"
echo "Live benchmark: tail -n 30 -F outputs/slurm/figure_benchmark_${benchmark_id}.out"
echo "Live figures: tail -n 30 -F outputs/slurm/final_figures_${figures_id}.out"
echo "Start estimate: squeue --start -j ${benchmark_id},${figures_id}"
