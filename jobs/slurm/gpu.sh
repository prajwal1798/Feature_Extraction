#!/bin/bash
# EDIT THE THREE CLUSTER LINES: --account (your project code), --partition
# (the GPU partition of your cluster) and --gres (one GPU, type name as on
# your cluster). Everything else stays.
#SBATCH --job-name=gpu
#SBATCH --account=ACCOUNT
#SBATCH --partition=accel_ai
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%j.out

# =============================================================================
# GPU JOB: RUNS ONE run.py COMMAND ON ONE GPU
#
#   sbatch jobs/slurm/gpu.sh  <exactly what you would type after "python run.py">
#
#   example:  sbatch jobs/slurm/gpu.sh train --operator fno --seed 321
#
# Submit from the project folder. The Python used is the environment made
# in the project folder:  <project>/env/bin/python  (see README.md).
# Only the lines marked EDIT depend on the cluster.
# =============================================================================

# =============================================================================
# HEADER OF THE OUTPUT FILE: WHICH JOB, WHERE, WHEN, WHAT
# =============================================================================
echo "JOB      : ${SLURM_JOB_ID}   (array ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none})"
echo "NODE     : $(hostname)"
echo "STARTED  : $(date)"
echo "COMMAND  : python run.py $*"
echo

# THE PYTHON OF THE PROJECT ENVIRONMENT, CALLED BY ITS FULL PATH
PY="${SLURM_SUBMIT_DIR}/env/bin/python"
echo "PYTHON   : ${PY}"
"${PY}" -c "import torch; print('TORCH    :', torch.__version__)"
"${PY}" -c "import ngsolve; print('NGSOLVE  :', ngsolve.__version__)"
echo

export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4

"${PY}" -u run.py "$@" --device cuda

STATUS=$?

echo
echo "FINISHED : $(date)"
echo "EXIT CODE: ${STATUS}   (0 = success; anything else = look for the error above)"
exit ${STATUS}
