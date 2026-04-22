#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=commons
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --array=0-99%10          # <-- 99 jobs, max 10 concurrent
#SBATCH --output=/scratch/my13/kai/runs/paper/logs/%x_%A_%a.out

echo "JOB_START host=$(hostname) date=$(date) jobid=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID:-none}"

if [[ -z "${HPO_SCRIPT:-}" ]]; then
    echo "ERROR: HPO_SCRIPT env var is not set." >&2
    exit 1
fi

SCRIPT_ARGS="${SCRIPT_ARGS:-}"

source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba activate /projects/my13/kai/meta-pers-gest/envs/fl-torch

export HPO_USE_JOURNAL=1
export CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
export DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
export MAML_DIR="$CODE_DIR/system/MAML"
export MOE_DIR="$CODE_DIR/system/MOE"
export PYTHONPATH="$CODE_DIR:$MAML_DIR:$MOE_DIR:${PYTHONPATH:-}"

# RUN_DIR is per-array-task (for model checkpoints)
export RUN_DIR=/scratch/my13/kai/runs/paper/logs/${SLURM_JOB_NAME}/trial_${SLURM_ARRAY_TASK_ID}
mkdir -p "$RUN_DIR"
mkdir -p /scratch/my13/kai/runs/paper/logs

# N_TRIALS=1: each job runs exactly one Optuna trial, then exits
export N_TRIALS=1

echo "===== Environment Diagnostics ====="
echo "Host        : $(hostname)"
echo "HPO_SCRIPT  : ${HPO_SCRIPT}"
echo "RUN_DIR     : ${RUN_DIR}"
echo "ARRAY_TASK  : ${SLURM_ARRAY_TASK_ID}"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"
nvidia-smi || true
echo "==================================="

OVERRIDES_ARG=()
[[ -n "${OVERRIDES:-}" ]] && OVERRIDES_ARG=(--overrides "$OVERRIDES")

python -u "$CODE_DIR/system/NOTS/paper/${HPO_SCRIPT}" \
    --data_dir "$DATA_DIR" \
    --out_dir  "$RUN_DIR" \
    "${OVERRIDES_ARG[@]}" \
    $SCRIPT_ARGS

echo "JOB_END date=$(date)"