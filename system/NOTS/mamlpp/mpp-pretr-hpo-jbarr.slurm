#!/bin/bash
#SBATCH --job-name=mamlpp-hpo-jb
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

######## JOB ARRAY SETTINGS ########
# For 10 trials now:
#SBATCH --array=0-9
# For 100 trials, 5 at a time (Uncomment below and comment above to switch):
# #SBATCH --array=0-99%3

# SBATCH --export=ALL

echo "JOB_START host=$(hostname) date=$(date)"

source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0 

source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh

mamba activate /projects/my13/kai/meta-pers-gest/envs/fl-torch

######## Paths & Unique Naming ########
export CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
export DATA_DIR=/scratch/my13/kai/meta-pers-gest/data

# This creates a unique directory for EVERY trial so they never interfere
export RUN_DIR=/scratch/my13/kai/runs/logs/${SLURM_JOB_NAME}/trial_${SLURM_ARRAY_TASK_ID}
mkdir -p "$RUN_DIR"

export MOE_DIR="$CODE_DIR/system/MAML_MOE"
export PYTHONPATH="$CODE_DIR:$MOE_DIR:$PYTHONPATH"

MODEL_NAME=${MODEL_NAME:-TST}

echo "CWD: $(pwd)"
echo "Output Directory: $RUN_DIR"
python -u "$CODE_DIR/system/NOTS/mamlpp/mamlpp_pretrained_hpo.py" \
    --model_type "$MODEL_NAME" \
    --data_dir "$DATA_DIR" \
    --out_dir  "$DEBUG_RUN_DIR"
