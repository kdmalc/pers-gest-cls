#!/bin/bash
#SBATCH --job-name=maml-hpo
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --time=11:45:00
#SBATCH --gres=gpu:1
# Optional, only if your site really has those node features:
# # SBATCH --constraint="ampere|lovelace"
# Use a small whitelist, not ALL:
# SBATCH --export=HOME,USER,LOGNAME,SHELL,LANG,LC_ALL,PATH,PROJECTS,WORK,SHARED_SCRATCH,LOCAL_SCRATCH
#SBATCH --export=ALL

# Launch 5 workers (Trial IDs will be managed by Optuna automatically)
#SBATCH --array=1-5

# Safer bash: exit on error/undefined vars; fail pipelines if any step fails
echo "JOB_START host=$(hostname) date=$(date)"
# ArrayID=${SLURM_ARRAY_TASK_ID}" idk if this is defined? Unless it is automatically created...
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
export CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
export DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
export RUN_DIR=/scratch/my13/kai/runs/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
export DEBUG_RUN_DIR=/scratch/my13/kai/runs
#mkdir -p "$RUN_DIR" /scratch/my13/kai/logs
# For relative file imports (in other .py files)
export MOE_DIR="$CODE_DIR/system/MAML_MOE"
# before running python:
# Put CODE_DIR (for ...) and MOE_DIR (for ...) on PYTHONPATH
export PYTHONPATH="$CODE_DIR:$MOE_DIR:$PYTHONPATH"

echo "CWD:"; pwd

# Run the script
# Each array task will run the python script, which runs 1 trial and exits.
python -u "$CODE_DIR/system/NOTS/debugging/maml_hpo.py" \
    --data_dir "$DATA_DIR" #\
#    --out_dir  "$DEBUG_RUN_DIR"
# ^ Would maybe rather just let it write to its log dir? Is this why the created dirs are always empty presumably?