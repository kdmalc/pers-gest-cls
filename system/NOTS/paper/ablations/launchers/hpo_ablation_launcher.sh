#!/bin/bash
# hpo_ablation_launcher.sh
# ========================
# Submit Optuna HPO array jobs for one or more ablation IDs.
#
# Each ablation gets its own SLURM array job (100 tasks by default, 1 trial
# per task). All tasks for the same ablation share one Optuna journal file,
# so TPE can use results from already-finished tasks to guide later ones.
#
# Usage:
#   bash hpo_ablation_launcher.sh M0                      # HPO for M0
#   bash hpo_ablation_launcher.sh A1 A2 A3                # multiple ablations
#   bash hpo_ablation_launcher.sh all                     # all HPO-able ablations
#   bash hpo_ablation_launcher.sh M0 --dry-run            # print without submitting
#   bash hpo_ablation_launcher.sh A1 --debug              # single trial, debug partition,
#                                                          # no journal write, warm-start enqueued
#   bash hpo_ablation_launcher.sh A3 --n-trials 50        # custom trial count
#   bash hpo_ablation_launcher.sh X2 F1 T0 --emg-repr fs64 --n-trials 24 --study-suffix screen \
#        --fixed-params /scratch/.../shared_maml.json          # (hpo_tools.py freeze makes the JSON)
#                                                          # architecture screen (ARCH_VARIANTS tokens
#                                                          # in ablation_hpo.py: X2-X6 F1-F3 T0 T2 T4
#                                                          # P1 P1_moe P1_match P1_rel P1_ridge P1_feat
#                                                          # P3 A4_ANIL M0_ANIL Q1 Q1_nola Q1_mlp Q1_uni Q1_ce)
#
# Per-trial cost: every trial prints one "HPO_TRIAL_TIMING {json}" line and stores the same fields in
# the journal; summarise with  python system/NOTS/paper/ablations/hpo_tools.py report
#
# Output layout:
#   Optuna journals : $HPO_DB_DIR/ablation_<ID>_10w1s_kfold7_<EMG_REPR>_v1.log
#   Trial checkpts  : $HPO_OUT_BASE/<ID>/trial_<array_task_id>/
#   SLURM logs      : $LOG_DIR/<jobname>_<jobid>_<taskid>.out
#
# A10 is intentionally excluded: zero-shot protocol, no HPs to tune.
# A6  is intentionally excluded: not yet implemented.
# A9  is intentionally excluded: not needed for paper (per ablation_launcher.sh).

set -euo pipefail

# =============================================================================
# Paths — edit these for your cluster layout
# =============================================================================
CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
HPO_SCRIPT_PATH="$CODE_DIR/system/NOTS/paper/ablations/ablation_hpo.py"
MOE_HPO_SCRIPT_PATH="$CODE_DIR/system/NOTS/paper/ablations/M0_MOE_hpo.py"
# sub2 = the resubmission runs (10-way 1-shot, kfold7 protocol, dev-user HPO)
HPO_OUT_BASE=/scratch/my13/kai/runs/paper/sub2/hpo
HPO_DB_DIR=$HPO_OUT_BASE/optuna_dbs
LOG_DIR=$HPO_OUT_BASE/logs

ENV_PATH=/projects/my13/kai/meta-pers-gest/envs/fl-torch

# This launcher is copied to scratch and run from there: refuse a stale copy.
REPO_LAUNCHER="$CODE_DIR/system/NOTS/paper/ablations/launchers/$(basename "${BASH_SOURCE[0]}")"
if ! cmp -s "${BASH_SOURCE[0]}" "$REPO_LAUNCHER"; then
    echo "ERROR: this launcher differs from the repo copy (stale?). Refresh it with:"
    echo "  cp $REPO_LAUNCHER $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"
    exit 1
fi

mkdir -p "$HPO_DB_DIR" "$HPO_OUT_BASE" "$LOG_DIR"

# =============================================================================
# Parse args
# =============================================================================
# Initialize defaults
ABLATIONS=()
DRY_RUN=false
DEBUG=false
N_TRIALS=60     # equal budget per method family (docs/specs/SPEC_hpo.md)
EXPLICIT_N_TRIALS=""   
OVERRIDE_PARTITION=""
EMG_REPR_ARG=""        # required: env20 | fs64 (no silent default)
STUDY_SUFFIX=""        # optional: separate study of the same ablation (screen, Exk grid)
FIXED_PARAMS=""        # optional: absolute path of a {param: value} JSON frozen in every trial (screen)
i=0
args_array=("$@")

# Single, clean parsing loop
while [[ $i -lt ${#args_array[@]} ]]; do
    arg="${args_array[$i]}"
    case "$arg" in
        --dry-run)    DRY_RUN=true ;;
        --debug)      DEBUG=true ;;
        --n-trials)   i=$((i+1)); N_TRIALS="${args_array[$i]}"; EXPLICIT_N_TRIALS="${args_array[$i]}" ;;
        --partition)  i=$((i+1)); OVERRIDE_PARTITION="${args_array[$i]}" ;;
        --emg-repr)   i=$((i+1)); EMG_REPR_ARG="${args_array[$i]}" ;;
        --study-suffix) i=$((i+1)); STUDY_SUFFIX="${args_array[$i]}" ;;
        --fixed-params) i=$((i+1)); FIXED_PARAMS="${args_array[$i]}" ;;
        all)         ABLATIONS=(M0 A1 A2 A3 A4 A5 A7 A8 A11 A12) ;;
        -*)           echo "ERROR: unknown flag '$arg'"; exit 1 ;;
        *)            ABLATIONS+=("$arg") ;;
    esac
    i=$((i+1))
done

if [[ ${#ABLATIONS[@]} -eq 0 ]]; then
    echo "ERROR: No ablations specified."
    echo "Usage: bash hpo_ablation_launcher.sh [M0|A1|A2|A3|A4|A5|A7|A8|A11|A12|M0_MOE_hpo|all] [--dry-run] [--debug] [--n-trials N] [--partition PARTITION]"
    exit 1
fi

if [[ ${#ABLATIONS[@]} -eq 0 ]]; then
    echo "ERROR: No ablations specified."
    echo "Usage: bash hpo_ablation_launcher.sh [M0|A1|A2|A3|A4|A5|A7|A8|A11|A12|M0_MOE_hpo|all] --emg-repr {env20|fs64} [--dry-run] [--debug] [--n-trials N] [--partition PARTITION]"
    exit 1
fi
if [[ -n "$FIXED_PARAMS" ]]; then
    [[ "$FIXED_PARAMS" = /* && -f "$FIXED_PARAMS" ]] || { echo "ERROR: --fixed-params needs an existing absolute path (got '$FIXED_PARAMS')"; exit 1; }
    [[ -n "$STUDY_SUFFIX" ]] || { echo "ERROR: --fixed-params needs --study-suffix (a frozen-HP study must not share the main study's journal)"; exit 1; }
fi
case "$EMG_REPR_ARG" in
    env20|fs64|raw3s) ;;
    *) echo "ERROR: --emg-repr env20|fs64|raw3s is required (got '${EMG_REPR_ARG}')"; exit 1 ;;
esac

# =============================================================================
# Cluster defaults
# =============================================================================
PARTITION=commons
CPUS=10
MEM_DEFAULT=32G
TIME_DEFAULT="12:00:00"    # 10-way MAML trials cost ~3x the old 3-way ones (old: 3-4h)

# =============================================================================
# Per-ablation resource overrides
# (comment these back in and adjust if you have per-ablation time data)
# =============================================================================
# TIME_M0="06:00:00";  MEM_M0=32G    # full MAML+MoE: ~3-4h
TIME_A1="01:30:00";  MEM_A1=24G    # supervised MoE; 10-way dev fine-tune evaluation (100 episodes)
TIME_A2="01:30:00";  MEM_A2=16G    # supervised, no MoE
# TIME_A3="04:00:00";  MEM_A3=24G    # MAML, no MoE
# TIME_A4="04:00:00";  MEM_A4=24G    # MAML, no MoE, wider
# TIME_A5="06:00:00";  MEM_A5=32G    # same as M0
TIME_A7="00:35:00";  MEM_A7=16G    # subject-specific supervised: fast (~15 min observed)
# TIME_A8="06:00:00";  MEM_A8=32G    # subject-specific MAML+MoE
TIME_A11="00:35:00"; MEM_A11=24G   # Meta pretrained, ft_lr only: fast (~15 min observed)
# TIME_A12="06:00:00"; MEM_A12=32G   # Our model on 2kHz data

# MOE_hpo: commons partition by default (scavenge max walltime is only 1h — too short).
# Each trial is a full MAML+MoE training run (~8+h observed for M0).
# The JournalFileBackend survives preemption so completed trials are never lost.
# Failed/preempted trials count against the 500-task array budget — hence
# submitting 500 rather than 200 to absorb expected failures.
# Adjust TIME_M0_MOE_hpo based on your observed per-trial wall time.
TIME_M0_MOE_hpo="20:00:00"; MEM_M0_MOE_hpo=32G
PARTITION_M0_MOE_hpo=commons
CONCURRENCY_M0_MOE_hpo=20

# Debug overrides:
#   - Single non-array job (N_TRIALS forced to 1, --array flag suppressed below)
#   - HPO_USE_JOURNAL=0 so nothing is written to the shared Optuna journal
#   - Warm-start params are still enqueued (InMemoryStorage supports enqueue_trial)
#   - Log file pattern uses %x_%j.out (no %a task suffix — no array)
if [[ "$DEBUG" == true ]]; then
    echo "DEBUG MODE: partition=debug, time=00:10:00, single trial, no journal write"
    PARTITION=debug
    TIME_DEFAULT="00:10:00"
    N_TRIALS=1
fi
# ^ I hardcoded this in later so I think this does nothing now? Might as well leave it

get_resource() {
    # get_resource TIME M0 "06:00:00"  ->  value of $TIME_M0, or default
    local varname="${1}_${2}"
    echo "${!varname:-$3}"
}

# Default trial count for M0_MOE_hpo when --n-trials is not explicitly passed.
# For all other ablations the default remains N_TRIALS (100).
_resolve_n_trials() {
    local ablation="$1"
    local explicit_n_trials="$2"   # "" if not set by user
    if [[ -n "$explicit_n_trials" ]]; then
        echo "$explicit_n_trials"
    elif [[ "$ablation" == "M0_MOE_hpo" ]]; then
        echo "500"
    else
        echo "$N_TRIALS"
    fi
}

# =============================================================================
# Submit each ablation
# =============================================================================

# NOTE: ARRAY_END is now computed per-ablation inside the loop, since
# MOE_hpo defaults to 500 trials while all other ablations default to N_TRIALS.

# Shared wrap script body — identical for both debug and production.
# Uses shell variables that are set per-ablation in the loop below.
# $4 = hpo_script_path: either $HPO_SCRIPT_PATH (ablation_hpo.py)
#                     or $MOE_HPO_SCRIPT_PATH (M0_MOE_hpo.py)
_make_wrap_body() {
    local out_dir="$1"
    local ablation="$2"
    local use_task_id="$3"   # "true" = production array (use SLURM_ARRAY_TASK_ID),
                              # "false" = debug single job (use SLURM_JOB_ID)
    local script_path="$4"   # full path to the HPO python script to invoke
    if [[ "$use_task_id" == "true" ]]; then
        local run_dir_line="export RUN_DIR=$out_dir/trial_\${SLURM_ARRAY_TASK_ID}"
        local job_echo='echo "JOB_START host=$(hostname) date=$(date) jobid=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"'
    else
        local run_dir_line="export RUN_DIR=$out_dir/debug_trial_\${SLURM_JOB_ID}"
        local job_echo='echo "JOB_START host=$(hostname) date=$(date) jobid=${SLURM_JOB_ID} [debug-single]"'
    fi

    cat <<WRAPEOF
source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba activate $ENV_PATH

$run_dir_line
mkdir -p "\$RUN_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$job_echo
echo 'ABLATION  : $ablation'
echo 'RUN_DIR   : '"\$RUN_DIR"
echo 'HPO_DB_DIR: $HPO_DB_DIR'

which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"
nvidia-smi || true

python -u $script_path --ablation $ablation

echo "JOB_END date=\$(date)"
WRAPEOF
}

for ABLATION in "${ABLATIONS[@]}"; do
    TIME=$(get_resource TIME "$ABLATION" "$TIME_DEFAULT")
    MEM=$(get_resource  MEM  "$ABLATION" "$MEM_DEFAULT")

    # ── Per-ablation overrides for M0_MOE_hpo ────────────────────────────────
    if [[ "$ABLATION" == "M0_MOE_hpo" ]]; then
        EFFECTIVE_PARTITION=$(get_resource PARTITION "$ABLATION" "commons")
        EFFECTIVE_CONCURRENCY=$(get_resource CONCURRENCY "$ABLATION" "10")
        EFFECTIVE_SCRIPT="$MOE_HPO_SCRIPT_PATH"
        # Journal name matches STUDY_NAME constant in M0_MOE_hpo.py
        EFFECTIVE_JOURNAL="$HPO_DB_DIR/moe_hpo_1s3w_hpo_v1.log"
    else
        EFFECTIVE_PARTITION="$PARTITION"
        EFFECTIVE_CONCURRENCY="10"
        EFFECTIVE_SCRIPT="$HPO_SCRIPT_PATH"
        EFFECTIVE_JOURNAL="$HPO_DB_DIR/ablation_${ABLATION}_10w1s_kfold7_${EMG_REPR_ARG}_v1${STUDY_SUFFIX:+_$STUDY_SUFFIX}.log"
    fi

    # --partition flag overrides whatever was resolved above for all ablations
    if [[ -n "$OVERRIDE_PARTITION" ]]; then
        EFFECTIVE_PARTITION="$OVERRIDE_PARTITION"
    fi

    # ── Resolve trial count ───────────────────────────────────────────────────
    EFFECTIVE_N_TRIALS=$(_resolve_n_trials "$ABLATION" "$EXPLICIT_N_TRIALS")
    ARRAY_END=$((EFFECTIVE_N_TRIALS - 1))

    JOB_NAME="hpo_${ABLATION}"
    OUT_DIR="$HPO_OUT_BASE/$EMG_REPR_ARG/$ABLATION${STUDY_SUFFIX:+_$STUDY_SUFFIX}"
    mkdir -p "$OUT_DIR"

    if [[ "$DEBUG" == true ]]; then
        # ── Debug: single non-array job, no journal write ─────────────────────
        # HPO_USE_JOURNAL=0  → InMemoryStorage (results discarded after job ends)
        # N_TRIALS=1         → run exactly one trial
        # No --array flag    → single job, log uses %x_%j.out (no %a suffix)
        SBATCH_CMD=(
            sbatch
            --job-name="$JOB_NAME"
            --partition="$PARTITION"
            --nodes=1
            --ntasks=1
            --cpus-per-task="$CPUS"
            --mem="$MEM"
            --time="00:15:00"
            --gres=gpu:1
            --output="$LOG_DIR/%x_%j.out"
            --export="ALL,\
CODE_DIR=$CODE_DIR,\
DATA_DIR=$DATA_DIR,\
HPO_DB_DIR=$HPO_DB_DIR,\
HPO_USE_JOURNAL=0,\
EMG_REPR=$EMG_REPR_ARG,\
HPO_STUDY_SUFFIX=$STUDY_SUFFIX,\
HPO_FIXED_PARAMS_JSON=$FIXED_PARAMS,\
N_TRIALS=1,\
MAML_DIR=$CODE_DIR/system/MAML,\
MOE_DIR=$CODE_DIR/system/MOE,\
PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-}"
            --wrap="$(_make_wrap_body "$OUT_DIR" "$ABLATION" "false" "$EFFECTIVE_SCRIPT")"
        )

        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Ablation   : $ABLATION  [DEBUG]"
        echo "  Script     : $EFFECTIVE_SCRIPT"
        echo "  Mode       : single job, 1 trial, no journal write"
        echo "  Partition  : $PARTITION  (debug always uses commons/debug)"
        echo "  Time       : 00:15:00"
        echo "  Memory     : $MEM"
        echo "  Output dir : $OUT_DIR"
        echo "  Log        : $LOG_DIR/${JOB_NAME}_<jobid>.out"
        echo "════════════════════════════════════════════════════"

    else
        # ── Production: array job, journal enabled ────────────────────────────
        SBATCH_CMD=(
            sbatch
            --job-name="$JOB_NAME"
            --partition="$EFFECTIVE_PARTITION"
            --nodes=1
            --ntasks=1
            --cpus-per-task="$CPUS"
            --mem="$MEM"
            --time="$TIME"
            --gres=gpu:1
            --array="0-${ARRAY_END}%${EFFECTIVE_CONCURRENCY}"
            --output="$LOG_DIR/%x_%A_%a.out"
            --export="ALL,\
CODE_DIR=$CODE_DIR,\
DATA_DIR=$DATA_DIR,\
HPO_DB_DIR=$HPO_DB_DIR,\
HPO_USE_JOURNAL=1,\
EMG_REPR=$EMG_REPR_ARG,\
HPO_STUDY_SUFFIX=$STUDY_SUFFIX,\
HPO_FIXED_PARAMS_JSON=$FIXED_PARAMS,\
N_TRIALS=1,\
MAML_DIR=$CODE_DIR/system/MAML,\
MOE_DIR=$CODE_DIR/system/MOE,\
PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-}"
            --wrap="$(_make_wrap_body "$OUT_DIR" "$ABLATION" "true" "$EFFECTIVE_SCRIPT")"
        )

        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Ablation   : $ABLATION"
        echo "  Script     : $EFFECTIVE_SCRIPT"
        echo "  Trials     : $EFFECTIVE_N_TRIALS  (array 0-${ARRAY_END}, max ${EFFECTIVE_CONCURRENCY} concurrent)"
        echo "  Partition  : $EFFECTIVE_PARTITION"
        echo "  Time/trial : $TIME"
        echo "  Memory     : $MEM"
        echo "  Output dir : $OUT_DIR"
        echo "  Optuna DB  : $EFFECTIVE_JOURNAL"
        echo "  Frozen HPs : ${FIXED_PARAMS:-none}"
        echo "  Log dir    : $LOG_DIR"
        echo "════════════════════════════════════════════════════"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would submit:"
        echo "  ${SBATCH_CMD[*]}"
    else
        JOB_ID=$("${SBATCH_CMD[@]}")
        echo "  Submitted: $JOB_ID"
    fi
done

echo ""
echo "Done. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Log locations:"
echo "  Production (array) : $LOG_DIR/hpo_<ABLATION>_<arrayjobid>_<taskid>.out"
echo "  Debug (single job) : $LOG_DIR/hpo_<ABLATION>_<jobid>.out"
echo "  tail example       : tail -f $LOG_DIR/hpo_M0_<jobid>.out"
echo ""
echo "Inspect ablation HPO results:"
echo "  python -c \""
echo "    import optuna"
echo "    from optuna.storages.journal import JournalStorage, JournalFileBackend"
echo "    storage = JournalStorage(JournalFileBackend('$HPO_DB_DIR/ablation_<ID>_10w1s_kfold7_<EMG_REPR>_v1.log'))"
echo "    study = optuna.load_study(study_name='ablation_<ID>_1s3w_hpo_v1', storage=storage)"
echo "    print(study.best_trial)"
echo "  \""
echo ""
echo "Inspect M0_MOE_hpo results:"
echo "  python -c \""
echo "    import optuna"
echo "    from optuna.storages.journal import JournalStorage, JournalFileBackend"
echo "    storage = JournalStorage(JournalFileBackend('$HPO_DB_DIR/moe_hpo_1s3w_hpo_v1.log'))"
echo "    study = optuna.load_study(study_name='moe_hpo_1s3w_hpo_v1', storage=storage)"
echo "    print(study.best_trial)"
echo "    print(study.best_params)"
echo "  \""
