#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# pretrain_hpo.slurm
# ══════════════════════════════════════════════════════════════════════════════
# Parallel pretrain HPO on the cluster via a SLURM job array.
# All workers share one Optuna JournalStorage log file. 

#SBATCH --job-name=pretrain-hpo
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-19
#SBATCH --output=logs/pretrain_hpo_%A_%a.out
#SBATCH --error=logs/pretrain_hpo_%A_%a.err

mkdir -p logs

echo "══════════════════════════════════════════════════════════════"
echo "PRETRAIN HPO WORKER"
echo "  host        : $(hostname)"
echo "  date        : $(date)"
echo "  job id      : ${SLURM_JOB_ID}"
echo "  array task  : ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_MAX}"
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
export DATA_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds

export RUN_DIR=/scratch/my13/kai/runs/pretrain_hpo
mkdir -p "$RUN_DIR"

export OPTUNA_DB_DIR=/scratch/my13/kai/meta-pers-gest/optuna_dbs
mkdir -p "$OPTUNA_DB_DIR"

# ── Python path ────────────────────────────────────────────────────────────────
export PRETRAIN_DIR="$CODE_DIR/system/pretraining"
export MOE_DIR="$CODE_DIR/system/MOE"
export MAML_DIR="$CODE_DIR/system/MAML"
export PYTHONPATH="$CODE_DIR:$PRETRAIN_DIR:$MOE_DIR:$MAML_DIR:$PYTHONPATH" 

# ── Study config ───────────────────────────────────────────────────────────────
# MODEL: Defaults to DeepCNNLSTM if not provided. 
MODEL="${MODEL:-DeepCNNLSTM}"

# USE_MOE: Defaulted to 1 (True). To turn off, use --export=ALL,USE_MOE=0. 
USE_MOE="${USE_MOE:-1}"

# Trials per worker. 
N_TRIALS_PER_WORKER="${N_TRIALS_PER_WORKER:-5}"

# ── Build study name + journal path ───────────────────────────────────────────
if [ "$USE_MOE" = "1" ]; then
    MOE_SUFFIX="_moe"
    MOE_FLAG="--use_moe"
else
    MOE_SUFFIX=""
    MOE_FLAG=""
fi

STUDY_NAME="pretrain_hpo_${MODEL}${MOE_SUFFIX}"
JOURNAL_PATH="${OPTUNA_DB_DIR}/${STUDY_NAME}.log"
TENSOR_DICT="${DATA_DIR}/segfilt_rts_tensor_dict.pkl" 

echo ""
echo "Study       : ${STUDY_NAME}"
echo "Model       : ${MODEL}  |  MoE=${USE_MOE}"
echo "Journal     : ${JOURNAL_PATH}"
echo "Trials/wrkr : ${N_TRIALS_PER_WORKER}"
echo "Tensor dict : ${TENSOR_DICT}"
echo ""

# ── Launch ─────────────────────────────────────────────────────────────────────
# Python script will now receive --use_moe by default. 
python -u "$CODE_DIR/system/NOTS/pretraining/pretrain_hpo.py" \
    --model        "$MODEL"               \
    --tensor_dict  "$TENSOR_DICT"          \
    --n_trials     "$N_TRIALS_PER_WORKER"  \
    --study_name   "$STUDY_NAME"           \
    --journal_path "$JOURNAL_PATH"         \
    $MOE_FLAG 

echo ""
echo "Worker ${SLURM_ARRAY_TASK_ID} finished at $(date)"