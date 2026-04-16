#!/bin/bash
# ablation_launcher.sh
# ====================
# Submit one or more ablation jobs to the cluster.
#
# Usage:
#   bash ablation_launcher.sh M0                    # submit M0
#   bash ablation_launcher.sh A1 A2 A3              # submit multiple
#   bash ablation_launcher.sh all                   # submit all implemented ablations
#   bash ablation_launcher.sh A5 --dry-run          # print sbatch command without submitting
#   bash ablation_launcher.sh A5 --debug            # use debug partition (15 min time limit)
#   bash ablation_launcher.sh A5 --debug --dry-run  # combine both
#
# Each ablation is its own SLURM job (separate log, separate GPU, separate RUN_DIR).
# A5 (expert count sweep, 8×5=40 training runs) gets extra time and memory.
# A7/A8 (per-subject, all subjects × 5 seeds) get the most time.
#
# All jobs write logs to: /scratch/my13/kai/runs/paper/ablations/logs/
# All jobs write outputs to: /scratch/my13/kai/runs/paper/ablations/<ABLATION_ID>/

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
ABLATION_DIR="$CODE_DIR/system/NOTS/paper/ablations"
LOG_DIR=/scratch/my13/kai/runs/paper/ablations/logs
OUTPUT_BASE=/scratch/my13/kai/runs/paper/ablations

mkdir -p "$LOG_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────
DRY_RUN=false
DEBUG=false
ABLATIONS=()
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    elif [[ "$arg" == "--debug" ]]; then
        DEBUG=true
    elif [[ "$arg" == "all" ]]; then
        ABLATIONS=(M0 A1 A2 A3 A4 A5 A7 A8 A11 A12)
        # A9, A10 are not included in "all"
    else
        ABLATIONS+=("$arg")
    fi
done

if [[ ${#ABLATIONS[@]} -eq 0 ]]; then
    echo "ERROR: No ablations specified."
    echo "Usage: bash ablation_launcher.sh [M0|A1|A2|A3|A4|A5|A7|A8|A9|A10|A11|A12|all] [--dry-run]"
    exit 1
fi

# ── Shared sbatch defaults ─────────────────────────────────────────────────────
PARTITION=commons
CPUS=10
MEM=32G
ENV_PATH=/projects/my13/kai/meta-pers-gest/envs/fl-torch

# ── Per-ablation resource overrides ───────────────────────────────────────────
# Format: TIME_<ID>=HH:MM:SS  MEM_<ID>=XG
# Defaults apply if not overridden.
TIME_DEFAULT="20:00:00"

# ── Debug mode overrides ───────────────────────────────────────────────────────
if [[ "$DEBUG" == true ]]; then
    echo "DEBUG MODE: partition=debug, time=00:15:00"
    PARTITION=debug
    TIME_DEFAULT="00:15:00"
fi

# Time limit on commons is 24 hours I think. So I'm gonan comment all these for now...
#TIME_M0="24:00:00";  MEM_M0=32G    # 5 seeds × full MAML
#TIME_A1="16:00:00";  MEM_A1=24G    # 5 seeds × supervised (faster)
#TIME_A2="12:00:00";  MEM_A2=24G    # 5 seeds × supervised, no MoE
#TIME_A3="24:00:00";  MEM_A3=32G    # 5 seeds × MAML, no MoE
#TIME_A4="24:00:00";  MEM_A4=32G    # 5 seeds × MAML, wider encoder
#TIME_A5="72:00:00";  MEM_A5=48G    # 8 expert counts × 5 seeds × full MAML — HEAVY
#TIME_A7="48:00:00";  MEM_A7=32G    # N_subjects × 5 seeds × supervised per-subject
#TIME_A8="72:00:00";  MEM_A8=48G    # N_subjects × 5 seeds × full MAML per-subject — HEAVY
#TIME_A9="24:00:00";  MEM_A9=32G    # 5 seeds × MAML + separate modality experts
#TIME_A12="24:00:00"; MEM_A12=32G   # 5 seeds × full MAML, different data

# ── Helper to get resource value ──────────────────────────────────────────────
get_resource() {
    local varname="${1}_${2}"   # e.g. TIME_M0
    echo "${!varname:-$3}"      # use default $3 if not set
}

# ── Per-ablation script and extra args ────────────────────────────────────────
# Format: script_name [extra_args]
declare -A ABLATION_SCRIPT
ABLATION_SCRIPT[M0]="M0_full_model.py"
ABLATION_SCRIPT[A1]="A1_no_maml_moe.py"
ABLATION_SCRIPT[A2]="A2_no_maml_no_moe.py"
ABLATION_SCRIPT[A3]="A3_A4_maml_no_moe.py --ablation A3"
ABLATION_SCRIPT[A4]="A3_A4_maml_no_moe.py --ablation A4"
ABLATION_SCRIPT[A5]="A5_expert_count_sweep.py"
# A6: stub — not launchable yet
ABLATION_SCRIPT[A7]="A7_A8_subject_specific.py --ablation A7"
ABLATION_SCRIPT[A8]="A7_A8_subject_specific.py --ablation A8"
ABLATION_SCRIPT[A9]="A9_modality_encoding.py --variant separate"  # NOTE: I dont care about this one
ABLATION_SCRIPT[A10]="A10_A11_A12_meta_pretrained.py --ablation A10"  # NOTE: I dont think we care about the fake zero-shot Meta model either
ABLATION_SCRIPT[A11]="A10_A11_A12_meta_pretrained.py --ablation A11"
ABLATION_SCRIPT[A12]="A10_A11_A12_meta_pretrained.py --ablation A12"

# ── Submit each ablation ──────────────────────────────────────────────────────
for ABLATION in "${ABLATIONS[@]}"; do
    if [[ -z "${ABLATION_SCRIPT[$ABLATION]+_}" ]]; then
        echo "WARNING: '$ABLATION' is not a launchable ablation (stub or unknown). Skipping."
        continue
    fi

    SCRIPT_AND_ARGS="${ABLATION_SCRIPT[$ABLATION]}"
    SCRIPT=$(echo "$SCRIPT_AND_ARGS" | awk '{print $1}')
    EXTRA_ARGS=$(echo "$SCRIPT_AND_ARGS" | cut -d' ' -f2-)
    [[ "$EXTRA_ARGS" == "$SCRIPT" ]] && EXTRA_ARGS=""   # no extra args

    TIME=$(get_resource TIME "$ABLATION" "$TIME_DEFAULT")
    MEM=$(get_resource MEM  "$ABLATION" "$MEM")

    OUT_DIR="$OUTPUT_BASE/$ABLATION"
    mkdir -p "$OUT_DIR"

    JOB_NAME="abl_${ABLATION}"

    SBATCH_CMD=(
        sbatch
        --job-name="$JOB_NAME"
        --partition="$PARTITION"
        --nodes=1
        --ntasks=1
        --cpus-per-task="$CPUS"
        --mem="$MEM"
        --time="$TIME"
        --gres=gpu:1
        --output="$LOG_DIR/%x_%j.out"
        --export="ALL,\
CODE_DIR=$CODE_DIR,\
DATA_DIR=/scratch/my13/kai/meta-pers-gest/data,\
RUN_DIR=$OUT_DIR,\
MAML_DIR=$CODE_DIR/system/MAML,\
MOE_DIR=$CODE_DIR/system/MOE,\
PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-}"
        --wrap="
source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba activate $ENV_PATH

echo 'JOB_START host=\$(hostname) date=\$(date) jobid=\${SLURM_JOB_ID}'
echo 'CODE_DIR : $CODE_DIR'
echo 'RUN_DIR  : $OUT_DIR'
echo 'ABLATION : $ABLATION'
echo 'SCRIPT   : $SCRIPT'
echo 'EXTRA    : $EXTRA_ARGS'

which python
python -c \"import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.is_available()}')\"
nvidia-smi || true

python -u $ABLATION_DIR/$SCRIPT $EXTRA_ARGS

echo 'JOB_END date=\$(date)'
"
    )

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Ablation : $ABLATION"
    echo "  Script   : $SCRIPT $EXTRA_ARGS"
    echo "  Time     : $TIME"
    echo "  Mem      : $MEM"
    echo "  Output   : $OUT_DIR"
    echo "  Log      : $LOG_DIR/${JOB_NAME}_<jobid>.out"
    echo "════════════════════════════════════════════════════"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would submit:"
        echo "  ${SBATCH_CMD[*]}"
    else
        JOB_ID=$("${SBATCH_CMD[@]}")
        echo "  Submitted: $JOB_ID"
    fi
done

echo ""
echo "Done. Check job status with: squeue -u \$USER"
echo "Logs: $LOG_DIR/"