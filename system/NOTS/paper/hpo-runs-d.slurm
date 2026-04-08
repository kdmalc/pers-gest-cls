#!/bin/bash
# ==============================================================================
# Debug HPO launcher — single job, debug partition, 1 Optuna trial.
#
# USAGE:
#   export HPO_SCRIPT="mamlpp/mamlpp_pretr_moe_hpo.py"
#   export SCRIPT_ARGS="--model_type DeepCNNLSTM"   # optional
#   sbatch mpp-hpo-debug.slurm
#
# To override time or mem at the call site:
#   sbatch --time=00:30:00 --mem=20G mpp-hpo-debug.slurm
# ==============================================================================

#SBATCH --job-name=hpo-debug
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

echo "JOB_START host=$(hostname) date=$(date) jobid=${SLURM_JOB_ID}"

# ---------- Validate required inputs ------------------------------------------
if [[ -z "${HPO_SCRIPT:-}" ]]; then
    echo "ERROR: HPO_SCRIPT env var is not set. Example:" >&2
    echo "  export HPO_SCRIPT=mamlpp/mamlpp_pretr_moe_hpo.py" >&2
    exit 1
fi

SCRIPT_ARGS="${SCRIPT_ARGS:-}"

# Force a single Optuna trial — overrides the N_TRIALS constant in the script
export N_TRIALS=1

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

export RUN_DIR=/scratch/my13/kai/runs/debug/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

# ---------- 4) Diagnostics ----------------------------------------------------
echo "===== Environment Diagnostics ====="
echo "Host       : $(hostname)"
echo "HPO_SCRIPT : ${HPO_SCRIPT}"
echo "SCRIPT_ARGS: ${SCRIPT_ARGS}"
echo "N_TRIALS   : ${N_TRIALS}"
echo "RUN_DIR    : ${RUN_DIR}"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"
python -c "import sys; print('sys.prefix:', sys.prefix)"
python -c "import numpy as np; print('numpy:', np.__version__)"
nvidia-smi || true
echo "==================================="

# ---------- 5) Launch ---------------------------------------------------------
echo "CWD: $(pwd)"
# shellcheck disable=SC2086
python -u "$CODE_DIR/system/NOTS/paper/${HPO_SCRIPT}" \
    --data_dir "$DATA_DIR" \
    --out_dir  "$RUN_DIR" \
    $SCRIPT_ARGS

echo "JOB_END date=$(date)"
