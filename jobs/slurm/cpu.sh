#!/bin/bash
# EDIT THE TWO CLUSTER LINES: --account (your project code), --partition
# (the CPU partition of your cluster). Everything else stays.
#SBATCH --job-name=cpu
#SBATCH --account=ACCOUNT
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%j.out

# =============================================================================
# CPU JOB: RUNS ONE run.py COMMAND ON A CPU NODE
# (data generation, selection of a tuning stage, reports)
#
#   sbatch jobs/slurm/cpu.sh  <exactly what you would type after "python run.py">
#
#   example:  sbatch jobs/slurm/cpu.sh generate --family scaling
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
export OMP_NUM_THREADS=4

"${PY}" -u run.py "$@" --device cpu

STATUS=$?

echo
echo "FINISHED : $(date)"
echo "EXIT CODE: ${STATUS}   (0 = success; anything else = look for the error above)"
exit ${STATUS}
