#!/bin/bash
# eval_hp_launchers.sh
# ====================
# SLURM launchers for all four eval HP jobs.
#
# Four modes (run in this order):
#
#   A11          Extended A11 eval HPO — finds best ft_lr and ft_steps.
#                Array job, one trial per task.
#
#   A11_CURVE    A11 paper figure curve — sweeps ft_steps at fixed best ft_lr.
#                Single job. Run after A11 HPO completes.
#
#   M0_SWEEP     M0 2D eval sweep — finds best (maml_inner_steps_eval, alpha).
#                Single job. Run in parallel with A11 if cluster allows.
#
#   M0_CURVE     M0 paper figure curve — sweeps steps at fixed best alpha.
#                Single job. Run after M0_SWEEP completes.
#
# Full workflow:
#   bash eval_hp_launchers.sh A11 --n-trials 50
#   bash eval_hp_launchers.sh M0_SWEEP
#   # ... wait for both to finish, inspect JSONs for best ft_lr and best alpha ...
#   bash eval_hp_launchers.sh A11_CURVE --ft-lr <best_ft_lr>
#   bash eval_hp_launchers.sh M0_CURVE  [--alpha <best_alpha>]
#
# A12 uses the M0 paper curve result directly — no separate sweep needed.
# If you want to verify, pass --ablation-id A12 to M0_SWEEP / M0_CURVE.
#
# Usage examples:
#   bash eval_hp_launchers.sh A11 --n-trials 50
#   bash eval_hp_launchers.sh A11 --n-trials 50 --dry-run
#   bash eval_hp_launchers.sh A11 --debug
#   bash eval_hp_launchers.sh A11_CURVE --ft-lr 0.05
#   bash eval_hp_launchers.sh M0_SWEEP
#   bash eval_hp_launchers.sh M0_SWEEP  --checkpoint /path/to/ckpt.pt
#   bash eval_hp_launchers.sh M0_CURVE
#   bash eval_hp_launchers.sh M0_CURVE  --alpha 0.005
#   bash eval_hp_launchers.sh M0_SWEEP  --checkpoint /path/to/A12_ckpt.pt --ablation-id A12

set -euo pipefail

# =============================================================================
# Paths — edit these to match your cluster layout
# =============================================================================
CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
HPO_DB_DIR=/scratch/my13/kai/meta-pers-gest/optuna_dbs
LOG_DIR=/scratch/my13/kai/runs/paper/ablations/eval_hp/logs
ENV_PATH=/projects/my13/kai/meta-pers-gest/envs/fl-torch

A11_HPO_SCRIPT="$CODE_DIR/system/NOTS/paper/ablations/sweeps/A11_eval_hpo_extended.py"
M0_SWEEP_SCRIPT="$CODE_DIR/system/NOTS/paper/ablations/sweeps/maml_eval_hp_sweep.py"

A11_OUT_BASE=/scratch/my13/kai/runs/paper/ablations/eval_hp/A11_v2
M0_SWEEP_OUT=/scratch/my13/kai/runs/paper/ablations/eval_hp/M0_sweep

mkdir -p "$HPO_DB_DIR" "$LOG_DIR" "$A11_OUT_BASE" "$M0_SWEEP_OUT"

# =============================================================================
# Defaults — edit these when you have new best values from sweeps
# =============================================================================
DEFAULT_CHECKPOINT=/projects/my13/kai/meta-pers-gest/checkpoints/M0_best.pt
DEFAULT_ALPHA=0.005   # update after M0_SWEEP completes

# =============================================================================
# Parse arguments
# =============================================================================
MODE=""
DRY_RUN=false
DEBUG=false
N_TRIALS=100
CHECKPOINT=""
ABLATION_ID="M0"
BEST_FT_LR=""
BEST_ALPHA=""

i=0
args_array=("$@")
while [[ $i -lt ${#args_array[@]} ]]; do
    arg="${args_array[$i]}"
    case "$arg" in
        A11)          MODE="A11" ;;
        A11_CURVE)    MODE="A11_CURVE" ;;
        M0_SWEEP)     MODE="M0_SWEEP" ;;
        M0_CURVE)     MODE="M0_CURVE" ;;
        --dry-run)    DRY_RUN=true ;;
        --debug)      DEBUG=true ;;
        --n-trials)    i=$((i+1)); N_TRIALS="${args_array[$i]}" ;;
        --checkpoint)  i=$((i+1)); CHECKPOINT="${args_array[$i]}" ;;
        --ablation-id) i=$((i+1)); ABLATION_ID="${args_array[$i]}" ;;
        --ft-lr)       i=$((i+1)); BEST_FT_LR="${args_array[$i]}" ;;
        --alpha)       i=$((i+1)); BEST_ALPHA="${args_array[$i]}" ;;
        *) echo "WARNING: Unknown argument '$arg' — ignoring." ;;
    esac
    i=$((i+1))
done

if [[ -z "$MODE" ]]; then
    echo "ERROR: Must specify mode: A11, A11_CURVE, M0_SWEEP, or M0_CURVE."
    echo ""
    echo "Usage:"
    echo "  bash eval_hp_launchers.sh A11       [--n-trials N] [--dry-run] [--debug]"
    echo "  bash eval_hp_launchers.sh A11_CURVE --ft-lr <value> [--dry-run]"
    echo "  bash eval_hp_launchers.sh M0_SWEEP  [--checkpoint /path/to/ckpt.pt] [--ablation-id A12] [--dry-run]"
    echo "  bash eval_hp_launchers.sh M0_CURVE  [--checkpoint /path/to/ckpt.pt] [--alpha <value>] [--ablation-id A12] [--dry-run]"
    exit 1
fi

# Apply defaults for checkpoint and alpha after parsing so explicit CLI args take precedence
if [[ -z "$CHECKPOINT" ]]; then
    CHECKPOINT="$DEFAULT_CHECKPOINT"
    echo "INFO: --checkpoint not specified, using default: $CHECKPOINT"
fi
if [[ -z "$BEST_ALPHA" ]]; then
    BEST_ALPHA="$DEFAULT_ALPHA"
    # Only notify when alpha is actually relevant
    if [[ "$MODE" == "M0_CURVE" ]]; then
        echo "INFO: --alpha not specified, using default: $BEST_ALPHA"
    fi
fi

# =============================================================================
# Shared environment setup snippet
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

_common_exports() {
    echo "CODE_DIR=$CODE_DIR,DATA_DIR=$DATA_DIR,\
HPO_DB_DIR=$HPO_DB_DIR,\
MAML_DIR=$CODE_DIR/system/MAML,MOE_DIR=$CODE_DIR/system/MOE,\
PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-}"
}

# =============================================================================
# MODE: A11 — extended HPO array job
# =============================================================================
if [[ "$MODE" == "A11" ]]; then

    PARTITION=commons
    TIME="00:35:00"
    MEM=24G
    CPUS=10
    JOB_NAME="eval_hpo_A11_v2"

    if [[ "$DEBUG" == true ]]; then
        PARTITION=debug; TIME="00:15:00"; N_TRIALS=1
        echo "DEBUG MODE: partition=debug, 1 trial, no journal write"
    fi

    ARRAY_END=$((N_TRIALS - 1))

    _make_wrap() {
        local use_task_id="$1"
        if [[ "$use_task_id" == "true" ]]; then
            local run_dir_line="export RUN_DIR=$A11_OUT_BASE/trial_\${SLURM_ARRAY_TASK_ID}"
            local echo_line='echo "JOB_START jobid=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"'
        else
            local run_dir_line="export RUN_DIR=$A11_OUT_BASE/debug_\${SLURM_JOB_ID}"
            local echo_line='echo "JOB_START jobid=${SLURM_JOB_ID} [debug] host=$(hostname)"'
        fi
        cat <<WRAPEOF
$(_env_setup)
mamba activate $ENV_PATH
$run_dir_line
mkdir -p "\$RUN_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$echo_line
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  GPU: {torch.cuda.is_available()}')"
nvidia-smi || true
python -u $A11_HPO_SCRIPT
echo "JOB_END date=\$(date)"
WRAPEOF
    }

    if [[ "$DEBUG" == true ]]; then
        SBATCH_CMD=(sbatch --job-name="$JOB_NAME" --partition="$PARTITION"
            --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM"
            --time="$TIME" --gres=gpu:1
            --output="$LOG_DIR/%x_%j.out"
            --export="ALL,$(_common_exports),HPO_USE_JOURNAL=0,N_TRIALS=1"
            --wrap="$(_make_wrap false)")
    else
        SBATCH_CMD=(sbatch --job-name="$JOB_NAME" --partition="$PARTITION"
            --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM"
            --time="$TIME" --gres=gpu:1
            --array="0-${ARRAY_END}%10"
            --output="$LOG_DIR/%x_%A_%a.out"
            --export="ALL,$(_common_exports),HPO_USE_JOURNAL=1,N_TRIALS=1"
            --wrap="$(_make_wrap true)")
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Mode       : A11 Extended Eval HPO"
    echo "  Trials     : $N_TRIALS"
    echo "  Partition  : $PARTITION"
    echo "  Time/trial : $TIME"
    echo "  Memory     : $MEM"
    echo "  Study name : ablation_A11_eval_hpo_v2"
    echo "  Journal    : $HPO_DB_DIR/ablation_A11_eval_hpo_v2.log"
    echo "  Output dir : $A11_OUT_BASE"
    echo "════════════════════════════════════════════════════"

# =============================================================================
# MODE: A11_CURVE — paper figure curve at fixed ft_lr
# =============================================================================
elif [[ "$MODE" == "A11_CURVE" ]]; then

    if [[ -z "$BEST_FT_LR" ]]; then
        echo "ERROR: --ft-lr is required for A11_CURVE mode."
        exit 1
    fi

    PARTITION=commons
    TIME="00:45:00"
    MEM=24G
    CPUS=10
    JOB_NAME="A11_paper_curve"
    OUT_DIR="$A11_OUT_BASE/paper_curve"
    mkdir -p "$OUT_DIR"

    WRAP=$(cat <<WRAPEOF
$(_env_setup)
mamba activate $ENV_PATH
export RUN_DIR=$OUT_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "JOB_START jobid=\${SLURM_JOB_ID} host=\$(hostname)"
echo "A11 paper curve at ft_lr=$BEST_FT_LR"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  GPU: {torch.cuda.is_available()}')"
nvidia-smi || true
python -u $A11_HPO_SCRIPT \\
    --paper-curve \\
    --ft-lr $BEST_FT_LR \\
    --out-dir $OUT_DIR
echo "JOB_END date=\$(date)"
WRAPEOF
)

    SBATCH_CMD=(sbatch --job-name="$JOB_NAME" --partition="$PARTITION"
        --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM"
        --time="$TIME" --gres=gpu:1
        --output="$LOG_DIR/%x_%j.out"
        --export="ALL,$(_common_exports)"
        --wrap="$WRAP")

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Mode       : A11 Paper Curve"
    echo "  ft_lr      : $BEST_FT_LR"
    echo "  Steps grid : 1 3 5 10 15 25 50 100 150 200"
    echo "  Partition  : $PARTITION"
    echo "  Time       : $TIME"
    echo "  Output dir : $OUT_DIR"
    echo "════════════════════════════════════════════════════"

# =============================================================================
# MODE: M0_SWEEP — 2D eval sweep (find best steps + alpha)
# --sweep-alpha is always passed to the python script for this mode;
# it is not a CLI toggle because M0_SWEEP always does a sweep by definition.
# =============================================================================
elif [[ "$MODE" == "M0_SWEEP" ]]; then

    PARTITION=commons
    TIME="03:30:00"   # 8 steps x 9 alphas x ~2 min + buffer
    MEM=32G
    CPUS=10
    JOB_NAME="eval_sweep_${ABLATION_ID}"
    OUT_DIR="$M0_SWEEP_OUT/${ABLATION_ID}"
    mkdir -p "$OUT_DIR"

    WRAP=$(cat <<WRAPEOF
$(_env_setup)
mamba activate $ENV_PATH
export RUN_DIR=$OUT_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "JOB_START jobid=\${SLURM_JOB_ID} host=\$(hostname)"
echo "M0 2D sweep: checkpoint=$CHECKPOINT  ablation=$ABLATION_ID"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  GPU: {torch.cuda.is_available()}')"
nvidia-smi || true
python -u $M0_SWEEP_SCRIPT \\
    --checkpoint $CHECKPOINT \\
    --ablation-id $ABLATION_ID \\
    --sweep-alpha \\
    --out-dir $OUT_DIR
echo "JOB_END date=\$(date)"
WRAPEOF
)

    SBATCH_CMD=(sbatch --job-name="$JOB_NAME" --partition="$PARTITION"
        --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM"
        --time="$TIME" --gres=gpu:1
        --output="$LOG_DIR/%x_%j.out"
        --export="ALL,$(_common_exports)"
        --wrap="$WRAP")

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Mode        : ${ABLATION_ID} 2D Eval Sweep"
    echo "  Checkpoint  : $CHECKPOINT"
    echo "  Steps grid  : 50 75 100 125 150 175 200 250"
    echo "  Alpha grid  : 0.001 0.002 0.003 0.005 0.007 0.010 0.015 0.020 0.030"
    echo "  Configs     : 72 total"
    echo "  Partition   : $PARTITION"
    echo "  Time        : $TIME"
    echo "  Output dir  : $OUT_DIR"
    echo "════════════════════════════════════════════════════"

# =============================================================================
# MODE: M0_CURVE — paper figure curve at fixed alpha
# =============================================================================
elif [[ "$MODE" == "M0_CURVE" ]]; then

    PARTITION=commons
    TIME="00:45:00"   # 10 steps x ~2-3 min each
    MEM=32G
    CPUS=10
    JOB_NAME="${ABLATION_ID}_paper_curve"
    OUT_DIR="$M0_SWEEP_OUT/${ABLATION_ID}_paper_curve"
    mkdir -p "$OUT_DIR"

    WRAP=$(cat <<WRAPEOF
$(_env_setup)
mamba activate $ENV_PATH
export RUN_DIR=$OUT_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "JOB_START jobid=\${SLURM_JOB_ID} host=\$(hostname)"
echo "${ABLATION_ID} paper curve: alpha=$BEST_ALPHA  checkpoint=$CHECKPOINT"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  GPU: {torch.cuda.is_available()}')"
nvidia-smi || true
python -u $M0_SWEEP_SCRIPT \\
    --checkpoint $CHECKPOINT \\
    --ablation-id $ABLATION_ID \\
    --paper-curve \\
    --alpha $BEST_ALPHA \\
    --out-dir $OUT_DIR
echo "JOB_END date=\$(date)"
WRAPEOF
)

    SBATCH_CMD=(sbatch --job-name="$JOB_NAME" --partition="$PARTITION"
        --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM"
        --time="$TIME" --gres=gpu:1
        --output="$LOG_DIR/%x_%j.out"
        --export="ALL,$(_common_exports)"
        --wrap="$WRAP")

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Mode        : ${ABLATION_ID} Paper Curve"
    echo "  Checkpoint  : $CHECKPOINT"
    echo "  Alpha       : $BEST_ALPHA"
    echo "  Steps grid  : 1 3 5 10 15 25 50 100 150 200"
    echo "  Partition   : $PARTITION"
    echo "  Time        : $TIME"
    echo "  Output dir  : $OUT_DIR"
    echo "════════════════════════════════════════════════════"

fi

# =============================================================================
# Submit or dry-run
# =============================================================================
if [[ "$DRY_RUN" == true ]]; then
    echo "  [DRY RUN] Would submit: ${SBATCH_CMD[*]}"
else
    JOB_ID=$("${SBATCH_CMD[@]}")
    echo "  Submitted: $JOB_ID"
fi

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     $LOG_DIR"
echo ""
#echo "After A11 HPO completes — inspect best trial:"
#echo "  python -c \""
#echo "    import optuna"
#echo "    from optuna.storages.journal import JournalStorage, JournalFileBackend"
#echo "    s = JournalStorage(JournalFileBackend('$HPO_DB_DIR/ablation_A11_eval_hpo_v2.log'))"
#echo "    study = optuna.load_study(study_name='ablation_A11_eval_hpo_v2', storage=s)"
#echo "    t = study.best_trial"
#echo "    print(f'best ft_lr={t.params[\"ft_lr\"]:.4e}  ft_steps={t.params[\"ft_steps\"]}  acc={t.value*100:.2f}%')"
#echo "  \""
#echo ""
#echo "After M0_SWEEP completes — inspect best (steps, alpha):"
#echo "  python -c \""
#echo "    import json, glob"
#echo "    f = sorted(glob.glob('$M0_SWEEP_OUT/M0/*.json'))[-1]"
#echo "    d = json.load(open(f))"
#echo "    print(f'best_steps={d[\"best_steps_so_far\"]}  best_alpha={d[\"best_alpha_so_far\"]}  acc={d[\"best_mean_acc_so_far\"]*100:.2f}%')"
#echo "  \""