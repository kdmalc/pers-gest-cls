#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# pretrain_hpo_debug.slurm  [DEBUG VERSION]
# ══════════════════════════════════════════════════════════════════════════════
# Single-job debug run (no array). Runs 2 trials on the debug partition.
#
# USAGE
# ──────
#   sbatch --export=ALL,MODEL=DeepCNNLSTM pretr_hpo_debug.slurm
#   sbatch --export=ALL,MODEL=ContrastiveNet pretr_hpo_debug.slurm
#
# With MoE:
#   sbatch --export=ALL,MODEL=DeepCNNLSTM,USE_MOE=1 pretr_hpo_debug.slurm
# ══════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=d-pretr
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
# SBATCH --output=logs/pretrain_hpo_debug_%j.out
# SBATCH --error=logs/pretrain_hpo_debug_%j.err

#mkdir -p logs

echo "══════════════════════════════════════════════════════════════"
echo "PRETRAIN HPO WORKER [DEBUG]"
echo "  host        : $(hostname)"
echo "  date        : $(date)"
echo "  job id      : ${SLURM_JOB_ID}"
echo "══════════════════════════════════════════════════════════════"

# ── Environment ───────────────────────────────────────────────────────────────
source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba activate /projects/my13/kai/meta-pers-gest/envs/fl-torch

# ── Project paths ──────────────────────────────────────────────────────────────
export CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
export DATA_DIR=/scratch/my13/kai/meta-pers-gest/data

export RUN_DIR=/scratch/my13/kai/runs/pretrain_hpo_debug
mkdir -p "$RUN_DIR"

export OPTUNA_DB_DIR=/scratch/my13/kai/meta-pers-gest/optuna_dbs_debug
mkdir -p "$OPTUNA_DB_DIR"

# ── Python path ────────────────────────────────────────────────────────────────
export PRETRAIN_DIR="$CODE_DIR/system/pretraining"
export MOE_DIR="$CODE_DIR/system/MOE"
export MAML_DIR="$CODE_DIR/system/MAML"
export PYTHONPATH="$CODE_DIR:$PRETRAIN_DIR:$MOE_DIR:$MAML_DIR:$PYTHONPATH"

# ── Study config ───────────────────────────────────────────────────────────────
MODEL="${MODEL:-DeepCNNLSTM}"
USE_MOE="${USE_MOE:1}"

# Debug: only 2 trials
N_TRIALS_PER_WORKER=2

# ── Build study name + journal path ───────────────────────────────────────────
if [ "$USE_MOE" = "1" ]; then
    MOE_SUFFIX="_moe"
    MOE_FLAG="--use_moe"
else
    MOE_SUFFIX=""
    MOE_FLAG=""
fi

STUDY_NAME="pretrain_hpo_debug_${MODEL}${MOE_SUFFIX}"
JOURNAL_PATH="${OPTUNA_DB_DIR}/${STUDY_NAME}.log"

TENSOR_DICT="${DATA_DIR}/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl"

echo ""
echo "Study       : ${STUDY_NAME}"
echo "Model       : ${MODEL}  |  MoE=${USE_MOE}"
echo "Journal     : ${JOURNAL_PATH}"
echo "Trials/wrkr : ${N_TRIALS_PER_WORKER}"
echo "Tensor dict : ${TENSOR_DICT}"
echo "RUN_DIR     : ${RUN_DIR}"
echo ""

# ── Launch ─────────────────────────────────────────────────────────────────────
python -u "$PRETRAIN_DIR/pretrain_hpo.py" \
    --model        "$MODEL"                \
    --tensor_dict  "$TENSOR_DICT"          \
    --n_trials     "$N_TRIALS_PER_WORKER"  \
    --study_name   "$STUDY_NAME"           \
    --journal_path "$JOURNAL_PATH"         \
    $MOE_FLAG

echo ""
echo "Debug job finished at $(date)"
