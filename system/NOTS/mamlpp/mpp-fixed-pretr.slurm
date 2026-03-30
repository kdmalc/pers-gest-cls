#!/bin/bash
#SBATCH --job-name=mpp-pretr-ablation
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
# SBATCH --output=logs/ablation_%A_%a.out

echo "JOB_START host=$(hostname) date=$(date) ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"

source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba activate /projects/my13/kai/meta-pers-gest/envs/fl-torch

export CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
export SAVE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls/maml_res
export DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
export MAML_DIR="$CODE_DIR/system/MAML"
export MOE_DIR="$CODE_DIR/system/MOE"
export PYTHONPATH="$CODE_DIR:$MAML_DIR:$MOE_DIR:$PYTHONPATH"

# Map array index → pretrain_approach (MoE disabled for all — debug separately)
# Index | pretrain_approach
#   0   | None        (no pretraining baseline)
#   1   | full_best
#   2   | full_last
#   3   | enc_best
#   4   | enc_last

case $SLURM_ARRAY_TASK_ID in
    0) PRETRAIN="None"      ;;
    1) PRETRAIN="full_best" ;;
    2) PRETRAIN="full_last" ;;
    3) PRETRAIN="enc_best"  ;;
    4) PRETRAIN="enc_last"  ;;
esac

export RUN_DIR=/scratch/my13/kai/runs/ablation/pretrain_${PRETRAIN}_task_${SLURM_ARRAY_TASK_ID}
mkdir -p "$RUN_DIR"

echo "Condition: pretrain=$PRETRAIN | fold=0 | MoE=off"

python -u "$MAML_DIR/mpp_run.py" \
    --model_type DeepCNNLSTM \
    --fold 0 \
    --pretrain_approach $PRETRAIN