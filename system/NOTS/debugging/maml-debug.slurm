#!/bin/bash
#SBATCH --job-name=maml-test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=00:29:00
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

MOE_DIR="${CODE_DIR}/April_25/MOE"
APRIL_DIR="${CODE_DIR}/April_25"  # This should cover both config and utils folders

echo "CODE_DIR: $CODE_DIR"
echo "MOE_DIR:  $MOE_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "RUN_DIR:  $RUN_DIR"
echo "APRIL_DIR: $APRIL_DIR"

# Make sure Python can see both the project root and the MOE modules
export PYTHONPATH="${CODE_DIR}:${MOE_DIR}:${APRIL_DIR}:${PYTHONPATH}"

echo "PYTHONPATH: $PYTHONPATH"

# (optional, but usually nice so outputs land in RUN_DIR)
# TODO: Would need to change how/where things are saved. Everything is still being saved to the general RUNS folder and cluttering it up, so I'll remove this for now
#mkdir -p "$RUN_DIR"
#cd "$RUN_DIR"

# Use the env's python explicitly if you want:
# /projects/my13/kai/meta-pers-gest/envs/fl-torch/bin/python \
#     "$CODE_DIR/April_25/NOTS/first_run/maml_run1.py"

python "$CODE_DIR/April_25/NOTS/first_run/maml_run1.py"

echo "===== Sanity ====="
echo "Host: $(hostname)"
echo "which python: $(which python)"
python -V
python -c "import sys; print('sys.prefix:', sys.prefix)"
python -c "import numpy as np; print('numpy:', np.__version__)"
#nvidia-smi || true
echo "=================="

echo "CWD:"; pwd
python "$CODE_DIR/April_25/NOTS/first_run/maml_run1.py" \
    --data_dir "$DATA_DIR" \
    --out_dir  "$RUN_DIR"