#!/bin/bash
# eval_hp_launchers.sh
# ====================
# SLURM launchers for two post-training eval HP jobs:
#
#   1) A11 extended eval HPO (a11_eval_hpo_extended.py)
#      Fixes the boundary-hitting problem in the original A11 HPO by widening
#      the search ranges for ft_lr and ft_steps.
#      Runs as a SLURM array job (one trial per task), same pattern as
#      hpo_ablation_launcher.sh. Uses a NEW Optuna study name so there is
#      zero collision with the existing A11 v1 study.
#
#   2) M0 (and optionally A12) eval HP sweep (maml_eval_hp_sweep.py)
#      Sweeps maml_inner_steps_eval x maml_alpha_init_eval jointly on the
#      val set given a trained checkpoint. Single job, no Optuna needed.
#
# Usage:
#   # A11 extended HPO (50 trials):
#   bash eval_hp_launchers.sh a11 --n-trials 50
#
#   # M0 eval sweep (steps only — fast):
#   bash eval_hp_launchers.sh m0_sweep --checkpoint /path/to/checkpoint.pt
#
#   # M0 eval sweep (steps + alpha — recommended):
#   bash eval_hp_launchers.sh m0_sweep --checkpoint /path/to/checkpoint.pt --sweep-alpha
#
#   # A12 eval sweep (reuses the same script, pass --ablation-id A12):
#   bash eval_hp_launchers.sh m0_sweep \
#       --checkpoint /path/to/A12_checkpoint.pt \
#       --ablation-id A12 --sweep-alpha
#
#   # Dry run:
#   bash eval_hp_launchers.sh a11 --n-trials 50 --dry-run
#   bash eval_hp_launchers.sh m0_sweep --checkpoint /path/to/checkpoint.pt --dry-run
#
#   # Debug (single trial, debug partition, no journal write):
#   bash eval_hp_launchers.sh a11 --debug

set -euo pipefail

# =============================================================================
# Paths — edit these to match your cluster layout
# =============================================================================
CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
HPO_DB_DIR=/scratch/my13/kai/meta-pers-gest/optuna_dbs
LOG_DIR=/scratch/my13/kai/runs/paper/ablations/eval_hp/logs
ENV_PATH=/projects/my13/kai/meta-pers-gest/envs/fl-torch

# Script paths
A11_HPO_SCRIPT="$CODE_DIR/system/NOTS/paper/ablations/a11_eval_hpo_extended.py"
M0_SWEEP_SCRIPT="$CODE_DIR/system/NOTS/paper/ablations/maml_eval_hp_sweep.py"

# Output dirs
A11_OUT_BASE=/scratch/my13/kai/runs/paper/ablations/eval_hp/A11_v2
M0_SWEEP_OUT=/scratch/my13/kai/runs/paper/ablations/eval_hp/M0_sweep

mkdir -p "$HPO_DB_DIR" "$LOG_DIR" "$A11_OUT_BASE" "$M0_SWEEP_OUT"

# =============================================================================
# Parse arguments
# =============================================================================
MODE=""             # "a11" or "m0_sweep"
DRY_RUN=false
DEBUG=false
N_TRIALS=50         # Default for A11 HPO
CHECKPOINT=""       # Required for m0_sweep
SWEEP_ALPHA=false
ABLATION_ID="M0"    # For m0_sweep; can be overridden to A12

i=0
args_array=("$@")
while [[ $i -lt ${#args_array[@]} ]]; do
    arg="${args_array[$i]}"
    case "$arg" in
        a11)          MODE="a11" ;;
        m0_sweep)     MODE="m0_sweep" ;;
        --dry-run)    DRY_RUN=true ;;
        --debug)      DEBUG=true ;;
        --sweep-alpha) SWEEP_ALPHA=true ;;
        --n-trials)    i=$((i+1)); N_TRIALS="${args_array[$i]}" ;;
        --checkpoint)  i=$((i+1)); CHECKPOINT="${args_array[$i]}" ;;
        --ablation-id) i=$((i+1)); ABLATION_ID="${args_array[$i]}" ;;
        *) echo "WARNING: Unknown argument '$arg' — ignoring." ;;
    esac
    i=$((i+1))
done

if [[ -z "$MODE" ]]; then
    echo "ERROR: Must specify mode: 'a11' or 'm0_sweep'."
    echo "Usage:"
    echo "  bash eval_hp_launchers.sh a11 [--n-trials N] [--dry-run] [--debug]"
    echo "  bash eval_hp_launchers.sh m0_sweep --checkpoint /path/to/ckpt.pt [--sweep-alpha] [--ablation-id A12] [--dry-run]"
    exit 1
fi

# =============================================================================
# Shared environment setup (written into --wrap for both modes)
# =============================================================================
_env_setup() {
    cat <<'ENVEOF'
source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
ENVEOF
}

# =============================================================================
# Mode: A11 extended HPO
# =============================================================================
if [[ "$MODE" == "a11" ]]; then

    PARTITION=commons
    TIME="00:35:00"      # head-only FT is fast; 35 min is generous per trial
    MEM=24G
    CPUS=10
    JOB_NAME="eval_hpo_A11_v2"

    if [[ "$DEBUG" == true ]]; then
        PARTITION=debug
        TIME="00:15:00"
        N_TRIALS=1
        echo "DEBUG MODE: partition=debug, 1 trial, no journal write"
    fi

    ARRAY_END=$((N_TRIALS - 1))

    _make_a11_wrap() {
        local use_task_id="$1"   # "true" = array, "false" = debug single
        if [[ "$use_task_id" == "true" ]]; then
            local run_dir_line="export RUN_DIR=$A11_OUT_BASE/trial_\${SLURM_ARRAY_TASK_ID}"
            local echo_line='echo "JOB_START host=$(hostname) date=$(date) jobid=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"'
        else
            local run_dir_line="export RUN_DIR=$A11_OUT_BASE/debug_trial_\${SLURM_JOB_ID}"
            local echo_line='echo "JOB_START host=$(hostname) date=$(date) jobid=${SLURM_JOB_ID} [debug]"'
        fi
        cat <<WRAPEOF
$(_env_setup)
mamba activate $ENV_PATH

$run_dir_line
mkdir -p "\$RUN_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$echo_line
echo "SCRIPT : $A11_HPO_SCRIPT"

which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"
nvidia-smi || true

python -u $A11_HPO_SCRIPT

echo "JOB_END date=\$(date)"
WRAPEOF
    }

    if [[ "$DEBUG" == true ]]; then
        SBATCH_CMD=(
            sbatch
            --job-name="$JOB_NAME"
            --partition="$PARTITION"
            --nodes=1 --ntasks=1
            --cpus-per-task="$CPUS"
            --mem="$MEM"
            --time="$TIME"
            --gres=gpu:1
            --output="$LOG_DIR/%x_%j.out"
            --export="ALL,\
CODE_DIR=$CODE_DIR,DATA_DIR=$DATA_DIR,\
HPO_DB_DIR=$HPO_DB_DIR,HPO_USE_JOURNAL=0,N_TRIALS=1,\
MAML_DIR=$CODE_DIR/system/MAML,MOE_DIR=$CODE_DIR/system/MOE,\
PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-}"
            --wrap="$(_make_a11_wrap false)"
        )
    else
        SBATCH_CMD=(
            sbatch
            --job-name="$JOB_NAME"
            --partition="$PARTITION"
            --nodes=1 --ntasks=1
            --cpus-per-task="$CPUS"
            --mem="$MEM"
            --time="$TIME"
            --gres=gpu:1
            --array="0-${ARRAY_END}%10"
            --output="$LOG_DIR/%x_%A_%a.out"
            --export="ALL,\
CODE_DIR=$CODE_DIR,DATA_DIR=$DATA_DIR,\
HPO_DB_DIR=$HPO_DB_DIR,HPO_USE_JOURNAL=1,N_TRIALS=1,\
MAML_DIR=$CODE_DIR/system/MAML,MOE_DIR=$CODE_DIR/system/MOE,\
PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-}"
            --wrap="$(_make_a11_wrap true)"
        )
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Job        : A11 Extended Eval HPO (v2)"
    echo "  Trials     : $N_TRIALS"
    echo "  Partition  : $PARTITION"
    echo "  Time/trial : $TIME"
    echo "  Memory     : $MEM"
    echo "  Study name : ablation_A11_eval_hpo_v2"
    echo "  Journal    : $HPO_DB_DIR/ablation_A11_eval_hpo_v2.log"
    echo "  Output dir : $A11_OUT_BASE"
    echo "  Log dir    : $LOG_DIR"
    echo "════════════════════════════════════════════════════"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would submit: ${SBATCH_CMD[*]}"
    else
        JOB_ID=$("${SBATCH_CMD[@]}")
        echo "  Submitted: $JOB_ID"
    fi

# =============================================================================
# Mode: M0 (or A12) eval HP sweep
# =============================================================================
elif [[ "$MODE" == "m0_sweep" ]]; then

    if [[ -z "$CHECKPOINT" ]]; then
        echo "ERROR: --checkpoint is required for m0_sweep mode."
        exit 1
    fi

    PARTITION=commons
    # 2D sweep: 16 step values × 9 alpha values = 144 configs × ~1 min each ≈ 2.5h
    # Steps-only sweep: 16 configs × ~1 min each ≈ 20 min
    if [[ "$SWEEP_ALPHA" == true ]]; then
        TIME="03:00:00"
    else
        TIME="00:45:00"
    fi
    MEM=32G
    CPUS=10
    JOB_NAME="eval_sweep_${ABLATION_ID}"

    ALPHA_FLAG=""
    if [[ "$SWEEP_ALPHA" == true ]]; then
        ALPHA_FLAG="--sweep_alpha"
    fi

    if [[ "$DEBUG" == true ]]; then
        PARTITION=debug
        TIME="00:15:00"
        echo "DEBUG MODE: partition=debug"
    fi

    SWEEP_WRAP=$(cat <<WRAPEOF
$(_env_setup)
mamba activate $ENV_PATH

export RUN_DIR=$M0_SWEEP_OUT/${ABLATION_ID}
mkdir -p "\$RUN_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "JOB_START host=\$(hostname) date=\$(date) jobid=\${SLURM_JOB_ID}"
echo "ABLATION  : $ABLATION_ID"
echo "CHECKPOINT: $CHECKPOINT"
echo "SWEEP_ALPHA: $SWEEP_ALPHA"

which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"
nvidia-smi || true

python -u $M0_SWEEP_SCRIPT \\
    --checkpoint $CHECKPOINT \\
    --ablation_id $ABLATION_ID \\
    --out_dir \$RUN_DIR \\
    $ALPHA_FLAG

echo "JOB_END date=\$(date)"
WRAPEOF
)

    SBATCH_CMD=(
        sbatch
        --job-name="$JOB_NAME"
        --partition="$PARTITION"
        --nodes=1 --ntasks=1
        --cpus-per-task="$CPUS"
        --mem="$MEM"
        --time="$TIME"
        --gres=gpu:1
        --output="$LOG_DIR/%x_%j.out"
        --export="ALL,\
CODE_DIR=$CODE_DIR,DATA_DIR=$DATA_DIR,\
MAML_DIR=$CODE_DIR/system/MAML,MOE_DIR=$CODE_DIR/system/MOE,\
PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-}"
        --wrap="$SWEEP_WRAP"
    )

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Job         : ${ABLATION_ID} Eval HP Sweep"
    echo "  Checkpoint  : $CHECKPOINT"
    echo "  Sweep alpha : $SWEEP_ALPHA"
    echo "  Partition   : $PARTITION"
    echo "  Time        : $TIME"
    echo "  Memory      : $MEM"
    echo "  Output dir  : $M0_SWEEP_OUT/$ABLATION_ID"
    echo "  Log dir     : $LOG_DIR"
    echo "════════════════════════════════════════════════════"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would submit: ${SBATCH_CMD[*]}"
    else
        JOB_ID=$("${SBATCH_CMD[@]}")
        echo "  Submitted: $JOB_ID"
    fi

fi

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs:          $LOG_DIR"
echo ""
echo "Inspect A11 v2 results:"
echo "  python -c \""
echo "    import optuna"
echo "    from optuna.storages.journal import JournalStorage, JournalFileBackend"
echo "    storage = JournalStorage(JournalFileBackend('$HPO_DB_DIR/ablation_A11_eval_hpo_v2.log'))"
echo "    study = optuna.load_study(study_name='ablation_A11_eval_hpo_v2', storage=storage)"
echo "    print(study.best_trial)"
echo "  \""
echo ""
echo "Inspect M0 sweep results:"
echo "  cat $M0_SWEEP_OUT/M0/eval_hp_sweep_M0_*.json | python -m json.tool | grep -A3 best"