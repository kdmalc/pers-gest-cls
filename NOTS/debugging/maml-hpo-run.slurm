#!/bin/bash
#SBATCH --job-name=maml-hpo
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
# Optional, only if your site really has those node features:
# COMMENT: SBATCH --constraint="ampere|lovelace"
# Use a small whitelist, not ALL:
# COMMENT: SBATCH --export=HOME,USER,LOGNAME,SHELL,LANG,LC_ALL,PATH,PROJECTS,WORK,SHARED_SCRATCH,LOCAL_SCRATCH
#SBATCH --export=ALL

# Safer bash: exit on error/undefined vars; fail pipelines if any step fails
echo "JOB_START host=$(hostname) date=$(date)"
#set -euo pipefail
#echo "after set -euo pipefail"

######## 1) Re-enable Environment Modules (whitelist doesn't give you 'module') ########
source /etc/profile.d/modules.sh

######## 2) Load exactly what you need ########
module purge
module load Mamba/23.11.0-0          # use the exact name from `module spider Mamba`

# Load conda shell functions first
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
# Then the mamba convenience hook (optional, but fine)
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh

######## 3) Activate your env ########
mamba activate /projects/my13/kai/meta-pers-gest/envs/fl-torch

######## 4) Paths & run ########
export CODE_DIR=/projects/my13/kai/meta-pers-gest/fl-gestures
export DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
export RUN_DIR=/scratch/my13/kai/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
export DEBUG_RUN_DIR=/scratch/my13/kai/runs
#mkdir -p "$RUN_DIR" /scratch/my13/kai/logs
# For relative file imports (in other .py files)
export MOE_DIR="$CODE_DIR/April_25/MOE"
export APRIL_DIR="$CODE_DIR/April_25"

# before running python
# Put CODE_DIR (for April_25.* imports) and MOE_DIR (for MOE_model_classes, etc.) on PYTHONPATH
export PYTHONPATH="$CODE_DIR:$MOE_DIR:$APRIL_DIR:$PYTHONPATH"

#echo "===== Sanity ====="
#echo "Host: $(hostname)"
#echo "which python: $(which python)"
#python -V
#python -c "import sys; print('sys.prefix:', sys.prefix)"
#python -c "import numpy as np; print('numpy:', np.__version__)"
#nvidia-smi || true
#echo "=================="

echo "CWD:"; pwd
python -u "$CODE_DIR/April_25/NOTS/debugging/maml_hpo.py" \
    --data_dir "$DATA_DIR" \
    --out_dir  "$DEBUG_RUN_DIR"