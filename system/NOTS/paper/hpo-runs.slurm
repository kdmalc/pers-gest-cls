#!/bin/bash
# Unified HPO launcher — works for any *_hpo.py script.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
# This --output control where the slurm.outs go
#SBATCH --output=/scratch/my13/kai/paper/logs/%x_%A_%a.out

echo "JOB_START host=$(hostname) date=$(date) jobid=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID:-none}"

# ---------- Validate required inputs ------------------------------------------
if [[ -z "${HPO_SCRIPT:-}" ]]; then
    echo "ERROR: HPO_SCRIPT env var is not set. Pass it via --export, e.g.:" >&2
    echo "  --export=ALL,HPO_SCRIPT=mamlpp/mamlpp_pretrained_hpo.py" >&2
    exit 1
fi

SCRIPT_ARGS="${SCRIPT_ARGS:-}"

# ---------- 1) Environment modules --------------------------------------------
source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0

source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh

# ---------- 2) Activate env ---------------------------------------------------
mamba activate /projects/my13/kai/meta-pers-gest/envs/fl-torch

# ---------- 3) Paths ----------------------------------------------------------
export CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
export DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
export MAML_DIR="$CODE_DIR/system/MAML"
export MOE_DIR="$CODE_DIR/system/MOE"
export PYTHONPATH="$CODE_DIR:$MAML_DIR:$MOE_DIR:${PYTHONPATH:-}"

# RUN_DIR: unique per trial if this is an array job, otherwise per job.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    export RUN_DIR=/scratch/my13/kai/runs/paper/logs/${SLURM_JOB_NAME}/trial_${SLURM_ARRAY_TASK_ID}
else
    export RUN_DIR=/scratch/my13/kai/runs/paper/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
fi
mkdir -p "$RUN_DIR"
mkdir -p /scratch/my13/kai/runs/paper/logs

# ---------- 4) Diagnostics ----------------------------------------------------
echo "===== Environment Diagnostics ====="
echo "Host      : $(hostname)"
echo "HPO_SCRIPT: ${HPO_SCRIPT}"
echo "SCRIPT_ARGS: ${SCRIPT_ARGS}"
echo "RUN_DIR   : ${RUN_DIR}"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"
python -c "import sys; print('sys.prefix:', sys.prefix)"
python -c "import numpy as np; print('numpy:', np.__version__)"
nvidia-smi || true
echo "==================================="

# ---------- 5) Launch ---------------------------------------------------------
echo "CWD: $(pwd)"
# shellcheck disable=SC2086  # SCRIPT_ARGS is intentionally word-split here
# This --out-dir controls where the checkpoints and such from Python go
python -u "$CODE_DIR/system/NOTS/paper/${HPO_SCRIPT}" \
    --data_dir "$DATA_DIR" \
    --out_dir  "$RUN_DIR" \
    $SCRIPT_ARGS

echo "JOB_END date=$(date)"
