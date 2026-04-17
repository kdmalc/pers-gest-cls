#!/bin/bash
#SBATCH --job-name=mpp-moe-ablation
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
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

# Map array index → (pretrain_approach, MOE, placement) — all fold 0
# Index | pretrain    | MOE | placement
#   0   | None        | off | —
#   1   | full_best   | off | —
#   2   | None        | on  | middle
#   3   | full_best   | on  | middle
#   4   | None        | on  | encoder
#   5   | full_best   | on  | encoder

case $SLURM_ARRAY_TASK_ID in
    0) PRETRAIN="None";      MOE_FLAG="" ;;#    1) PRETRAIN="full_best"; MOE_FLAG="" ;;
    2) PRETRAIN="None";      MOE_FLAG="--use_MOE --MOE_placement middle" ;;
    3) PRETRAIN="full_best"; MOE_FLAG="--use_MOE --MOE_placement middle" ;;
    4) PRETRAIN="None";      MOE_FLAG="--use_MOE --MOE_placement encoder" ;;
    5) PRETRAIN="full_best"; MOE_FLAG="--use_MOE --MOE_placement encoder" ;;
esac

export RUN_DIR=/scratch/my13/kai/runs/ablation/task_${SLURM_ARRAY_TASK_ID}
mkdir -p "$RUN_DIR"

echo "Condition: pretrain=$PRETRAIN $MOE_FLAG | fold=0"

python -u "$MAML_DIR/mpp_run.py" \
    --model_type DeepCNNLSTM \
    --fold 0 \
    --pretrain_approach $PRETRAIN \
    $MOE_FLAG