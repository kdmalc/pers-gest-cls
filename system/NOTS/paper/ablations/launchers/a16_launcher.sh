#!/bin/bash
# a16_launcher.sh
# ===============
# Submit the A16 upper-bound job (reviewer ARaC).
#
# This is a STANDALONE launcher rather than a patch to eval_launcher.sh. A16
# has a two-step workflow (build a checkpoint manifest, then consume it) that
# does not fit eval_launcher's one-token-one-script model, and eval_launcher is
# a working ~900-line production file that other rebuttal jobs still depend on.
# Keeping A16 separate means a mistake here cannot break M0/A13/portA/portB.
# Conventions (paths, module block, exit-code propagation) are copied from it.
#
# Usage:
#   bash a16_launcher.sh manifest              # step 1: find checkpoints (CPU, ~2 min)
#   bash a16_launcher.sh smoke                 # step 2: tiny end-to-end check
#   bash a16_launcher.sh run                   # step 3: the real job, both bases
#   bash a16_launcher.sh run --base A2         # single base
#   bash a16_launcher.sh run --dry-run         # print sbatch without submitting
#   bash a16_launcher.sh run --partition commons
#
# Step 1 writes $EVAL_OUT_BASE/a16_manifest.json. Steps 2 and 3 read it.
# Run step 1 on the login node (it is CPU-only and quick) or as a debug job.

set -euo pipefail

CLUSTER="NOTS"
for _a in "$@"; do [[ "$_a" == "RANGE" ]] && CLUSTER="RANGE"; done

if [[ "$CLUSTER" == "NOTS" ]]; then
    CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
    DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
    EVAL_OUT_BASE=/scratch/my13/kai/runs/paper/ablations/eval
    LOG_DIR=/scratch/my13/kai/runs/paper/ablations/eval/logs
    ENV_PATH=/projects/my13/kai/meta-pers-gest/envs/fl-torch
    MODULE_LOAD_BLOCK='source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh'
else
    CODE_DIR=/home/km82/pers-gest-cls
    DATA_DIR=/home/km82/data
    EVAL_OUT_BASE=/home/km82/runs/paper/ablations/eval
    LOG_DIR=/home/km82/runs/paper/ablations/eval/logs
    ENV_PATH=/home/km82/envs/fl-torch
    MODULE_LOAD_BLOCK='# RANGE: fill in module block'
fi

ABLATIONS_DIR="$CODE_DIR/system/NOTS/paper/ablations/test_eval_files"
MANIFEST="$EVAL_OUT_BASE/a16_manifest.json"

mkdir -p "$EVAL_OUT_BASE" "$LOG_DIR"

# ── Defaults, mirroring eval_launcher.sh ─────────────────────────────────────
PARTITION=commons
CPUS=10
MEM="32G"
# A16 does NO training. Cost is ~1.6k fine-tune-and-eval calls per base
# (4 test subjects x 10 leave-one-rep-out folds x 40 class combos), plus the
# HP grid on val subjects. M0 is MoE with 22 experts and runs slower than A2,
# so "both" gets the larger budget.
TIME_RUN="10:00:00"
TIME_SMOKE="00:15:00"   # debug partition caps wall time at 15:00

MODE="${1:-}"
shift || true

DRY_RUN=false
BASE="both"
EXTRA_ARGS=""
OVERRIDE_PARTITION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true ;;
        --base)      shift; BASE="$1" ;;
        --partition) shift; OVERRIDE_PARTITION="$1" ;;
        --cluster)   shift ;;  # already parsed
        *)           EXTRA_ARGS="$EXTRA_ARGS $1" ;;
    esac
    shift
done
[[ -n "$OVERRIDE_PARTITION" ]] && PARTITION="$OVERRIDE_PARTITION"

if [[ -z "$MODE" ]]; then
    echo "Usage: bash a16_launcher.sh [manifest|smoke|run] [--base A2|M0|both] [--dry-run] [--partition P]"
    exit 1
fi

# =============================================================================
# submit: one non-array job. Exit-code propagation is deliberate -- a trailing
# echo would mask a Python traceback and make sacct report COMPLETED 0:0 for a
# job that died, which has already burned several runs in this project.
# =============================================================================
submit() {
    local job_name="$1" out_dir="$2" time="$3" mem="$4" gres="$5" pycmd="$6"
    mkdir -p "$out_dir"

    local wrap_body
    wrap_body=$(cat <<WRAPEOF
$MODULE_LOAD_BLOCK
mamba activate $ENV_PATH

# mamba activate can silently no-op on some nodes (seen on the debug partition:
# module/hook sourcing appears to succeed with no error, but PATH still points
# at the base Mamba install rather than the target env). That failure is silent
# by default -- the job then runs base python and dies confusingly on
# "ModuleNotFoundError: torch/numpy" deep inside a script that has nothing
# wrong with it. Check explicitly and abort with a clear message instead.
ACTIVATED_PY=\$(which python)
echo "Activated python: \$ACTIVATED_PY"
if [[ "\$ACTIVATED_PY" != "$ENV_PATH"* ]]; then
    echo "ERROR: mamba activate did not switch to the target environment."
    echo "  Expected python under : $ENV_PATH"
    echo "  Got                   : \$ACTIVATED_PY"
    echo "  PATH=\$PATH"
    echo "  --- mamba/conda env list ---"
    mamba env list 2>&1 || conda env list 2>&1 || true
    echo "JOB_FAILED: environment activation did not take effect on this node."
    exit 1
fi

export RUN_DIR=$out_dir
mkdir -p "\$RUN_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "JOB_START host=\$(hostname) date=\$(date) jobid=\${SLURM_JOB_ID}"
echo "CLUSTER  : $CLUSTER"
echo "JOB      : $job_name"
echo "RUN_DIR  : $out_dir"
echo "CMD      : $pycmd"

which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"
nvidia-smi || true

set +e
$pycmd
PY_RC=\$?
set -e

echo "PYTHON_EXIT_CODE=\${PY_RC}"
echo "JOB_END date=\$(date) rc=\${PY_RC}"
if [[ \${PY_RC} -ne 0 ]]; then
    echo "JOB_FAILED: python exited \${PY_RC} -- this job did NOT produce valid results."
fi
exit \${PY_RC}
WRAPEOF
)

    local sbatch_cmd=(
        sbatch
        --job-name="$job_name"
        --partition="$PARTITION"
        --nodes=1 --ntasks=1
        --cpus-per-task="$CPUS"
        --mem="$mem"
        --time="$time"
        --output="$LOG_DIR/%x_%j.out"
        --export="ALL,CODE_DIR=$CODE_DIR,DATA_DIR=$DATA_DIR,MAML_DIR=$CODE_DIR/system/MAML,MOE_DIR=$CODE_DIR/system/MOE,PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-}"
    )
    [[ -n "$gres" ]] && sbatch_cmd+=(--gres="$gres")
    sbatch_cmd+=(--wrap="$wrap_body")

    echo ""
    echo "=================================================="
    echo "  Job       : $job_name"
    echo "  Partition : $PARTITION"
    echo "  Time      : $time"
    echo "  Memory    : $mem"
    echo "  GPU       : ${gres:-none}"
    echo "  Out dir   : $out_dir"
    echo "  Log       : $LOG_DIR/${job_name}_<jobid>.out"
    echo "=================================================="

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would submit:"
        echo "  ${sbatch_cmd[*]}"
    else
        echo "  Submitted: $("${sbatch_cmd[@]}")"
    fi
}

case "$MODE" in

    manifest)
        # CPU-only checkpoint scan. Fast enough to just run inline on the login
        # node; submitted as a job here only so the output is logged with the rest.
        echo "Scanning for pretrained A2/M0 checkpoints..."
        echo "This reads every .pt under the eval tree and screens out the April"
        echo "A2 checkpoints, which predate parameter matching (~0.6M params vs"
        echo "the matched ~6.1M) and would understate the CNN-LSTM ceiling."
        submit "a16_manifest" "$EVAL_OUT_BASE" "00:30:00" "16G" "" \
            "python -u $ABLATIONS_DIR/find_a16_checkpoints.py --out $MANIFEST"
        echo ""
        echo "To run it inline instead (faster, no queue):"
        echo "  mamba activate $ENV_PATH"
        echo "  python $ABLATIONS_DIR/find_a16_checkpoints.py --out $MANIFEST"
        ;;

    smoke)
        if [[ ! -f "$MANIFEST" ]]; then
            echo "ERROR: manifest not found at $MANIFEST"
            echo "Run:  bash a16_launcher.sh manifest"
            exit 1
        fi
        # Smoke runs default to the debug partition -- short, low-priority jobs
        # queue almost instantly there, and 15:00 comfortably fits the smoke
        # workload. Override with --partition if debug is unavailable/full.
        if [[ -z "$OVERRIDE_PARTITION" ]]; then
            PARTITION=debug
            echo "  (smoke mode defaults to the debug partition; pass --partition"
            echo "   to override, e.g. --partition commons)"
        fi
        submit "a16_smoke_${BASE}" "$EVAL_OUT_BASE/A16_smoke" \
            "$TIME_SMOKE" "32G" "gpu:1" \
            "python -u $ABLATIONS_DIR/A16_upper_bound.py --base $BASE --manifest $MANIFEST --smoke$EXTRA_ARGS"
        ;;

    run)
        if [[ ! -f "$MANIFEST" ]]; then
            echo "ERROR: manifest not found at $MANIFEST"
            echo "Run:  bash a16_launcher.sh manifest"
            exit 1
        fi
        submit "a16_run_${BASE}" "$EVAL_OUT_BASE/A16" \
            "$TIME_RUN" "$MEM" "gpu:1" \
            "python -u $ABLATIONS_DIR/A16_upper_bound.py --base $BASE --manifest $MANIFEST --stage both$EXTRA_ARGS"
        ;;

    *)
        echo "Unknown mode '$MODE'. Use: manifest | smoke | run"
        exit 1
        ;;
esac
