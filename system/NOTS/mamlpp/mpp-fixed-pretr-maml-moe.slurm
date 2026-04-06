#!/bin/bash
#SBATCH --job-name=mpp-pretr-hpo
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mpp_pretr_hpo_%j.out
#
# Single job — no array needed.
# Phase 1 (pretraining) runs first, then Phase 2 (15-trial MAML HPO)
# runs in the same allocation using the checkpoint written by Phase 1.
#
# Walltime estimate:
#   Phase 1 pretraining : ~1-2 h  (100 epochs, early stopping)
#   Phase 2 MAML HPO    : ~6-8 h  (15 trials × ~25-35 min each)
#   Total with margin   : 12 h
#
# If you want to run Phase 2 separately against an existing checkpoint:
#   python mamlpp_pretrained_hpo.py \
#       --skip_pretraining \
#       --pretrained_ckpt /path/to/DeepCNNLSTM_pretrained_YYYYMMDD_HHMM_best.pt

echo "JOB_START host=$(hostname) date=$(date) jobid=${SLURM_JOB_ID}"

source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba activate /projects/my13/kai/meta-pers-gest/envs/fl-torch

export CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
export DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
export MAML_DIR="$CODE_DIR/system/MAML"
export MOE_DIR="$CODE_DIR/system/MOE"
export PYTHONPATH="$CODE_DIR:$MAML_DIR:$MOE_DIR:$PYTHONPATH"

# All outputs (checkpoints, Optuna journal, summary JSON) go to RUN_DIR on scratch
export RUN_DIR=/scratch/my13/kai/runs/mamlpp/run_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
mkdir -p logs

echo "──────────────────────────────────────────────"
echo "  Phase 1: supervised pretraining"
echo "  Phase 2: MAML++ HPO (15 trials)"
echo "  model_type : DeepCNNLSTM  (DeepCNNLSTM + MoE encoder)"
echo "  RUN_DIR    : $RUN_DIR"
echo "──────────────────────────────────────────────"

python -u "$CODE_DIR/system/NOTS/mamlpp/mamlpp_pretr_moe_hpo.py" \
    --model_type DeepCNNLSTM

echo "JOB_END date=$(date)"
