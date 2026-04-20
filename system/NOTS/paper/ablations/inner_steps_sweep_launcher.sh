#!/bin/bash
# inner_steps_sweep_launcher.sh
# ==============================
# Submit the M0 maml_inner_steps_eval sweep as a single SLURM job.
#
# Usage:
#   bash inner_steps_sweep_launcher.sh              # submit
#   bash inner_steps_sweep_launcher.sh --dry-run    # print sbatch cmd without submitting
#
# Output:
#   Plots + JSON : $OUT_DIR/
#   SLURM log    : $OUT_DIR/logs/inner_steps_sweep_<jobid>.out

set -euo pipefail

# =============================================================================
# Paths
# =============================================================================
CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
CHECKPOINT_PATH=/scratch/my13/kai/runs/paper/ablations/hpo/M0/trial_25/trial_64_fold0_best.pt
OUT_DIR=/scratch/my13/kai/runs/paper/ablations/inner_steps_sweep

SCRIPT_PATH="$CODE_DIR/system/NOTS/paper/ablations/M0_inner_steps_eval_sweep.py"
ENV_PATH=/projects/my13/kai/meta-pers-gest/envs/fl-torch
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# =============================================================================
# Cluster resources
# =============================================================================
PARTITION=commons
CPUS=10
MEM=32G
TIME="02:00:00"   # 14 step counts × ~200 episodes each; well within 2h on GPU

# =============================================================================
# Args
# =============================================================================
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "WARNING: Unknown argument '$arg' — ignoring." ;;
    esac
done

# =============================================================================
# Wrap script
# =============================================================================
read -r -d '' WRAP_BODY << 'WRAPEOF' || true
source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba activate $ENV_PATH

echo "JOB_START host=$(hostname) date=$(date) jobid=${SLURM_JOB_ID}"
echo "CHECKPOINT : $CHECKPOINT_PATH"
echo "OUT_DIR    : $OUT_DIR"

which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"
nvidia-smi || true

python -u $SCRIPT_PATH

echo "JOB_END date=$(date)"
WRAPEOF

# =============================================================================
# Build sbatch command
# =============================================================================
SBATCH_CMD=(
    sbatch
    --job-name="inner_steps_sweep_M0"
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --time="$TIME"
    --gres=gpu:1
    --output="$LOG_DIR/inner_steps_sweep_%j.out"
    --export="ALL,\
CODE_DIR=$CODE_DIR,\
DATA_DIR=$DATA_DIR,\
CHECKPOINT_PATH=$CHECKPOINT_PATH,\
OUT_DIR=$OUT_DIR,\
MAML_DIR=$CODE_DIR/system/MAML,\
MOE_DIR=$CODE_DIR/system/MOE,\
PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-},\
ENV_PATH=$ENV_PATH,\
SCRIPT_PATH=$SCRIPT_PATH"
    --wrap="$WRAP_BODY"
)

# =============================================================================
# Print summary
# =============================================================================
echo ""
echo "════════════════════════════════════════════════════"
echo "  Job        : inner_steps_sweep_M0"
echo "  Partition  : $PARTITION"
echo "  Time       : $TIME"
echo "  Memory     : $MEM"
echo "  Checkpoint : $CHECKPOINT_PATH"
echo "  Output dir : $OUT_DIR"
echo "  Log        : $LOG_DIR/inner_steps_sweep_<jobid>.out"
echo "════════════════════════════════════════════════════"

if [[ "$DRY_RUN" == true ]]; then
    echo "  [DRY RUN] Would submit:"
    echo "  ${SBATCH_CMD[*]}"
else
    JOB_ID=$("${SBATCH_CMD[@]}")
    echo "  Submitted: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f $LOG_DIR/inner_steps_sweep_<jobid>.out"
fi
echo ""
