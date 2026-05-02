#!/bin/bash
# eval_launcher.sh
# =========================
# Submit final evaluation jobs for one or more ablation IDs.
#
# Most ablations are a single (non-array) SLURM job calling their dedicated
# Python script. Exceptions that spawn multiple parallel jobs:
#
#   M0      : L2SO mode spawns one job per subject fold (--fold-idx 0..N-1).
#             hpo_test_split mode spawns a single job (pass --test-procedure hpo_test_split).
#   A5      : sweeps num_experts over 8 values -> 8 jobs (see A5_EXPERT_COUNTS)
#   grid    : k-shot x n-way grid (M0/MAML+MoE) -> 9 jobs
#   grid_A2 : k-shot x n-way grid (A2/supervised CNN-LSTM) -> 9 jobs
#
# Script mapping:
#   M0       → M0_full_model.py
#   A1       → A1_no_maml_moe.py
#   A2       → A2_no_maml_no_moe.py
#   A3       → A3_A4_maml_no_moe.py  --ablation A3
#   A4       → A3_A4_maml_no_moe.py  --ablation A4
#   A5       → A5_expert_count_sweep.py      [one job per expert count]
#   A7       → A7_A8_subject_specific.py
#   A8       → A7_A8_subject_specific.py
#   A11      → A10_A11_A12_meta_pretrained.py
#   grid     → fewshot_grid.py               [one job per (k_shot, n_way) cell]
#   grid_A2  → fewshot_grid_A2.py            [one job per (k_shot, n_way) cell]
#   steps_M0  → num_eval_steps_sweep.py    --model-type maml        [steps sweep, no training]
#   steps_A2  → num_eval_steps_sweep.py    --model-type a2          [steps sweep, trains A2 inline]
#   steps_A7  → num_eval_steps_sweep.py    --model-type supervised  [steps sweep, no training]
#   steps_A11 → num_eval_steps_sweep.py    --model-type a11         [steps sweep, no training]
#
# Usage:
#   bash eval_launcher.sh M0                              # L2SO (default): 32 parallel fold jobs
#   bash eval_launcher.sh M0 --test-procedure hpo_test_split  # single fixed-split job
#   bash eval_launcher.sh A3                              # eval A3 only
#   bash eval_launcher.sh A4                              # eval A4 only
#   bash eval_launcher.sh A1 A2 A4                       # multiple ablations
#   bash eval_launcher.sh all                             # all ablations (no grid)
#   bash eval_launcher.sh grid                            # k-shot/n-way grid (M0/MAML+MoE)
#   bash eval_launcher.sh grid_A2                         # k-shot/n-way grid (A2/supervised)
#   bash eval_launcher.sh all grid grid_A2                # everything
#   bash eval_launcher.sh M0 --dry-run                   # print without submitting
#   bash eval_launcher.sh A1 --debug                     # debug partition, 15 min limit
#   bash eval_launcher.sh A5 --partition commons
#
#   For A2:
#   bash eval_launcher.sh A2 → no flag passed → Python argparse default (set to L2SO) runs L2SO
#   bash eval_launcher.sh A2 --test-procedure hpo_test_split → overrides to hpo_test_split
#
#   bash eval_launcher.sh steps_M0 steps_A2 steps_A7 steps_A11   # adaptation steps sweeps
#   bash eval_launcher.sh steps_A2 --ft-mode full                 # A2 steps sweep, full finetuning
#
# Output layout:
#   $EVAL_OUT_BASE/<ID>/                          (M0 hpo_test_split, A1, A2, A3, A4, A7, A8, A11)
#   $EVAL_OUT_BASE/M0/fold<NN>/                   (M0 L2SO, one subdir per fold)
#   $EVAL_OUT_BASE/A5/E<N>/                       (A5, one subdir per expert count)
#   $EVAL_OUT_BASE/grid/k<K>_n<N>/               (grid M0, one subdir per cell)
#   $EVAL_OUT_BASE/grid_A2/k<K>_n<N>/            (grid A2, one subdir per cell)
#   $EVAL_OUT_BASE/steps_sweep/<ID>/              (steps_M0, steps_A2, steps_A7, steps_A11)
#   $LOG_DIR/eval_<ID>_<jobid>.out

set -euo pipefail

# =============================================================================
# Paths — keep in sync with hpo_ablation_launcher.sh
# =============================================================================
CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
ABLATIONS_DIR="$CODE_DIR/system/NOTS/paper/ablations/test_eval_files"
EVAL_OUT_BASE=/scratch/my13/kai/runs/paper/ablations/eval
LOG_DIR=/scratch/my13/kai/runs/paper/ablations/eval/logs

ENV_PATH=/projects/my13/kai/meta-pers-gest/envs/fl-torch

mkdir -p "$EVAL_OUT_BASE" "$LOG_DIR"

# =============================================================================
# M0 L2SO: subject IDs — hardcoded to avoid login-node env dependency.
# MUST match the order of all_PIDs in ablation_config.py (TRAIN + VAL + TEST
# from fold 0 of hpo_strat_kapanji_split.json, concatenated in that order).
# The fold index i passed to --fold-idx corresponds to all_PIDs[i], so the
# ordering here must be IDENTICAL to config["all_PIDs"] in Python.
#
# To verify: activate your env, then run:
#   python -c "import sys; sys.path.insert(0, '$ABLATIONS_DIR'); from ablation_config import make_base_config; c=make_base_config('M0'); print(c['all_PIDs'])"
# =============================================================================
M0_L2SO_ALL_PIDS=(P011 P010 P008 P006 P111 P119 P124 P110 P112 P123 P132 P126 P102 P114 P107 P103 P125 P127 P128 P118 P108 P122 P106 P115 P005 P131 P116 P109 P004 P104 P105 P121)

M0_NUM_FOLDS=${#M0_L2SO_ALL_PIDS[@]}


# =============================================================================
# Adaptation steps sweep — checkpoint paths and eval HPs.
# UPDATE THESE before running steps_M0 or steps_A7.
# steps_A2 and steps_A11 do not need a checkpoint (A2 trains inline;
# A11 MetaEMGWrapper loads its own weights).
# =============================================================================
# Best M0 checkpoint from Trial 89 HPO (maml_alpha_init_eval = 5.066e-3).
STEPS_M0_CHECKPOINT="/projects/my13/kai/meta-pers-gest/pers-gest-cls/models/final_eval_models/best_M0_model.pt"
STEPS_M0_ALPHA="0.005066"   # maml_alpha_init_eval from Trial 89 — fixed, no alpha sweep needed

# Best A7 checkpoint. ft_lr defaults to 0.001 in the sweep script.
# TODO: Make A7 just train its own model... or just dont test A7....
#STEPS_A7_CHECKPOINT="/projects/my13/kai/meta-pers-gest/pers-gest-cls/hpo_best_models/A7_best.pt"

# A2: no checkpoint path needed — model is trained inline (~2 min).
# ft_lr defaults to 0.001 in the sweep script.

# A11: no checkpoint path needed — MetaEMGWrapper loads from the hardcoded Meta path.
# ft_lr defaults to 0.01 in the sweep script.

# ft_mode for supervised models. head_only is standard for 1-shot.
# Override per-run with --ft-mode full if you want the full-finetuning curve.
STEPS_FT_MODE="full"

# =============================================================================
# A5 expert count sweep definition.
# MUST be kept in sync with EXPERT_COUNTS in A5_expert_count_sweep.py.
# =============================================================================
A5_EXPERT_COUNTS=(4 8 12 16 20 24 32 40)

# =============================================================================
# Few-shot grid definition.
# MUST be kept in sync with:
#   GRID_K_SHOTS / GRID_N_WAYS in fewshot_grid.py    (M0)
#   GRID_K_SHOTS / GRID_N_WAYS in fewshot_grid_A2.py (A2)
# Both grids use the same (k, n) combinations.
# =============================================================================
GRID_K_SHOTS=(1 3 5)
GRID_N_WAYS=(3 5 10)

# =============================================================================
# Ablation ID -> Python script mapping.
# A3 and A4 share a script — the --ablation flag selects the variant.
# A7/A8 and A10/A11/A12 also share scripts similarly.
# A5, grid, and grid_A2 are handled separately in the submission loop.
# steps_M0 / steps_A2 / steps_A7 / steps_A11 all call num_eval_steps_sweep.py.
# =============================================================================
declare -A ABLATION_SCRIPT
ABLATION_SCRIPT[M0]="M0_full_model.py"
ABLATION_SCRIPT[A1]="A1_no_maml_moe.py"
ABLATION_SCRIPT[A2]="A2_no_maml_no_moe.py"
ABLATION_SCRIPT[A3]="A3_A4_maml_no_moe.py"
ABLATION_SCRIPT[A4]="A3_A4_maml_no_moe.py"
ABLATION_SCRIPT[A5]="A5_expert_count_sweep.py"
ABLATION_SCRIPT[A7]="A7_A8_subject_specific.py"
ABLATION_SCRIPT[A8]="A7_A8_subject_specific.py"
ABLATION_SCRIPT[A11]="A10_A11_A12_meta_pretrained.py"
ABLATION_SCRIPT[grid]="fewshot_grid.py"
ABLATION_SCRIPT[grid_A2]="fewshot_grid_A2.py"
ABLATION_SCRIPT[steps_M0]="num_eval_steps_sweep.py"
ABLATION_SCRIPT[steps_A2]="num_eval_steps_sweep.py"
ABLATION_SCRIPT[steps_A7]="num_eval_steps_sweep.py"
ABLATION_SCRIPT[steps_A11]="num_eval_steps_sweep.py"

# "all" expands to the standard ablation set only — grid and steps sweeps are opt-in.
VALID_ABLATIONS=(M0 A1 A2 A3 A4 A5 A7 A8 A11)
ALL_TOKENS=(M0 A1 A2 A4 A5 A7 A8 A11 grid grid_A2 steps_M0 steps_A2 steps_A7 steps_A11)  # for usage string

# =============================================================================
# Parse args
# =============================================================================
ABLATIONS=()
DRY_RUN=false
DEBUG=false
OVERRIDE_PARTITION=""
TEST_PROCEDURE_ARG=""
FT_MODE_ARG=""

i=0
args_array=("$@")

while [[ $i -lt ${#args_array[@]} ]]; do
    arg="${args_array[$i]}"
    case "$arg" in
        --dry-run)        DRY_RUN=true ;;
        --debug)          DEBUG=true ;;
        --test-procedure) i=$((i+1)); TEST_PROCEDURE_ARG="${args_array[$i]}" ;;
        --partition)      i=$((i+1)); OVERRIDE_PARTITION="${args_array[$i]}" ;;
        --ft-mode)        i=$((i+1)); FT_MODE_ARG="${args_array[$i]}" ;;
        all)              ABLATIONS+=("${VALID_ABLATIONS[@]}") ;;
        -*)               echo "WARNING: Unknown flag '$arg' -- ignoring." ;;
        *)                ABLATIONS+=("$arg") ;;
    esac
    i=$((i+1))
done

if [[ ${#ABLATIONS[@]} -eq 0 ]]; then
    echo "ERROR: No ablations specified."
    echo "Usage: bash eval_launcher.sh [$(IFS='|'; echo "${ALL_TOKENS[*]}")|all] [--dry-run] [--debug] [--partition PARTITION] [--ft-mode head_only|full] [--test-procedure hpo_test_split|L2SO]"
    echo "       'all' expands to: ${VALID_ABLATIONS[*]}  (grid, grid_A2, and steps sweeps are opt-in, not included in 'all')"
    exit 1
fi

# Validate all requested tokens upfront so we fail fast before submitting anything.
for ABL in "${ABLATIONS[@]}"; do
    if [[ -z "${ABLATION_SCRIPT[$ABL]+x}" ]]; then
        echo "ERROR: Unknown ablation '$ABL'. Valid tokens: ${ALL_TOKENS[*]}"
        exit 1
    fi
done

# Validate steps sweep checkpoint paths exist (fail fast before submitting).
# steps_A2 and steps_A11 are excluded — they have no checkpoint to validate.
for ABL in "${ABLATIONS[@]}"; do
    if [[ "$ABL" == "steps_M0" ]]; then
        if [[ ! -f "$STEPS_M0_CHECKPOINT" ]]; then
            echo "ERROR: steps_M0 checkpoint not found: $STEPS_M0_CHECKPOINT"
            echo "       Update STEPS_M0_CHECKPOINT at the top of this script."
            exit 1
        fi
    fi
    if [[ "$ABL" == "steps_A7" ]]; then
        if [[ ! -f "$STEPS_A7_CHECKPOINT" ]]; then
            echo "ERROR: steps_A7 checkpoint not found: $STEPS_A7_CHECKPOINT"
            echo "       Update STEPS_A7_CHECKPOINT at the top of this script."
            exit 1
        fi
    fi
done

# =============================================================================
# Cluster defaults
# =============================================================================
PARTITION=commons
CPUS=10
MEM_DEFAULT=32G
TIME_DEFAULT="12:00:00"

# =============================================================================
# Per-ablation resource overrides.
# Supervised ablations (A1, A2, A7, A11) are much faster than MAML ablations.
# MAML+MoE ablations (M0, A5, A8, grid) are the slowest.
# Adjust TIME_* based on observed wall times from HPO runs.
# steps_A2: training (~2 min) + 10 step values (~5 min each) = ~1h budget.
# steps_A7/A11: no training; 10 step values only.
# grid_A2 is supervised (like A2) so each cell should be comparable to A2 time.
# =============================================================================
TIME_A1="02:00:00";        MEM_A1=24G
TIME_A2="02:00:00";        MEM_A2=16G
TIME_A7="02:00:00";        MEM_A7=16G
TIME_A8="03:00:00";        MEM_A8=16G
TIME_A11="04:00:00";       MEM_A11=24G
TIME_steps_M0="04:00:00";  MEM_steps_M0=32G
TIME_steps_A2="02:00:00";  MEM_steps_A2=16G
TIME_steps_A7="02:00:00";  MEM_steps_A7=16G
TIME_steps_A11="03:00:00"; MEM_steps_A11=24G
TIME_grid_A2="02:00:00";   MEM_grid_A2=16G   # supervised; same budget as A2
# Uncomment and tune once you have wall-time data from HPO runs:
# TIME_M0="08:00:00";   MEM_M0=32G
# TIME_A3="05:00:00";   MEM_A3=24G
# TIME_A4="05:00:00";   MEM_A4=24G
# TIME_A5="08:00:00";   MEM_A5=32G    # applied per-expert-count job
# TIME_grid="08:00:00"; MEM_grid=32G  # applied per grid cell; higher k/n may need more time

get_resource() {
    # get_resource TIME M0 "08:00:00"  ->  value of $TIME_M0, or default
    local varname="${1}_${2}"
    echo "${!varname:-$3}"
}

# =============================================================================
# submit_single_job: submit one non-array SLURM job.
# Args: ablation label, script path, output dir, time, mem, partition, extra_args
# extra_args is optional; pass "" to omit.
# =============================================================================
submit_single_job() {
    local ablation="$1"
    local script_path="$2"
    local out_dir="$3"
    local time="$4"
    local mem="$5"
    local effective_partition="$6"
    local extra_args="${7:-}"

    local job_name="eval_${ablation}"
    mkdir -p "$out_dir"

    local wrap_body
    wrap_body=$(cat <<WRAPEOF
source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba activate $ENV_PATH

export RUN_DIR=$out_dir
mkdir -p "\$RUN_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "JOB_START host=\$(hostname) date=\$(date) jobid=\${SLURM_JOB_ID}"
echo "ABLATION  : $ablation"
echo "SCRIPT    : $script_path"
echo "RUN_DIR   : $out_dir"

which python
python -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"
nvidia-smi || true

python -u $script_path $extra_args

echo "JOB_END date=\$(date)"
WRAPEOF
)

    local sbatch_cmd=(
        sbatch
        --job-name="$job_name"
        --partition="$effective_partition"
        --nodes=1
        --ntasks=1
        --cpus-per-task="$CPUS"
        --mem="$mem"
        --time="$time"
        --gres=gpu:1
        --output="$LOG_DIR/%x_%j.out"
        --export="ALL,CODE_DIR=$CODE_DIR,DATA_DIR=$DATA_DIR,MAML_DIR=$CODE_DIR/system/MAML,MOE_DIR=$CODE_DIR/system/MOE,PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-}"
        --wrap="$wrap_body"
    )

    echo ""
    echo "=================================================="
    echo "  Job        : $job_name"
    echo "  Script     : $script_path"
    [[ -n "$extra_args" ]] && echo "  Args       : $extra_args"
    echo "  Partition  : $effective_partition"
    echo "  Time       : $time"
    echo "  Memory     : $mem"
    echo "  Output dir : $out_dir"
    echo "  Log        : $LOG_DIR/${job_name}_<jobid>.out"
    echo "=================================================="

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would submit:"
        echo "  ${sbatch_cmd[*]}"
    else
        local job_id
        job_id=$("${sbatch_cmd[@]}")
        echo "  Submitted: $job_id"
    fi
}

# =============================================================================
# Submit each ablation
# =============================================================================
for ABLATION in "${ABLATIONS[@]}"; do
    SCRIPT_NAME="${ABLATION_SCRIPT[$ABLATION]}"
    SCRIPT_PATH="$ABLATIONS_DIR/$SCRIPT_NAME"
    TIME=$(get_resource TIME "$ABLATION" "$TIME_DEFAULT")
    MEM=$(get_resource  MEM  "$ABLATION" "$MEM_DEFAULT")

    # Resolve test procedure: CLI --test-procedure overrides the default L2SO
    RESOLVED_TEST_PROCEDURE="${TEST_PROCEDURE_ARG:-L2SO}"

    # If doing L2SO, bump the time up to 23 hours for all runs EXCEPT M0
    # (since M0 is currently the only one set up to submit parallel fold jobs)
    if [[ "$RESOLVED_TEST_PROCEDURE" == "L2SO" && "$ABLATION" != "M0" ]]; then
        TIME="23:00:00"
    fi

    EFFECTIVE_PARTITION="$PARTITION"
    if [[ "$DEBUG" == true ]]; then
        EFFECTIVE_PARTITION=debug
        TIME="00:15:00"
    fi
    if [[ -n "$OVERRIDE_PARTITION" ]]; then
        EFFECTIVE_PARTITION="$OVERRIDE_PARTITION"
    fi

    # Resolve ft_mode for steps sweeps: CLI --ft-mode overrides STEPS_FT_MODE default.
    RESOLVED_FT_MODE="${FT_MODE_ARG:-$STEPS_FT_MODE}"

    if [[ "$ABLATION" == "M0" ]]; then
        # ── M0: behaviour depends on --test-procedure ──────────────────────────
        #
        # hpo_test_split → single job, mirrors the HPO fixed split (debugging only).
        # L2SO (default) → one job per subject fold, submitted in parallel.
        #                   Each job receives --fold-idx <i> and writes its results
        #                   to $EVAL_OUT_BASE/M0/fold<NN>/.
        #                   Aggregate stats are computed offline (see README / notebook).
        #
        # The resolved procedure:
        #   - If --test-procedure was passed on the CLI, use that.
        #   - Otherwise default to L2SO (matches the Python config default).

        if [[ "$RESOLVED_TEST_PROCEDURE" == "hpo_test_split" ]]; then
            echo ""
            echo "##################################################"
            echo "  M0 hpo_test_split: single job"
            echo "##################################################"
            submit_single_job \
                "M0_hpo" \
                "$SCRIPT_PATH" \
                "$EVAL_OUT_BASE/M0" \
                "$TIME" \
                "$MEM" \
                "$EFFECTIVE_PARTITION" \
                "--test-procedure hpo_test_split"

        else
            # L2SO: one job per fold
            echo ""
            echo "##################################################"
            echo "  M0 L2SO: ${M0_NUM_FOLDS} parallel fold jobs"
            echo "  Subject IDs: ${M0_L2SO_ALL_PIDS[*]}"
            echo "  Each job: --test-procedure L2SO --fold-idx <i>"
            echo "  Output  : $EVAL_OUT_BASE/M0/fold<NN>/"
            echo "##################################################"

            for FOLD_IDX in $(seq 0 $((M0_NUM_FOLDS - 1))); do
                PID="${M0_L2SO_ALL_PIDS[$FOLD_IDX]}"
                FOLD_LABEL=$(printf "fold%02d" "$FOLD_IDX")
                submit_single_job \
                    "M0_${FOLD_LABEL}_pid${PID}" \
                    "$SCRIPT_PATH" \
                    "$EVAL_OUT_BASE/M0/${FOLD_LABEL}" \
                    "$TIME" \
                    "$MEM" \
                    "$EFFECTIVE_PARTITION" \
                    "--test-procedure L2SO --fold-idx ${FOLD_IDX}"
            done
        fi

    elif [[ "$ABLATION" == "A5" ]]; then
        # ── A5: one job per expert count ──────────────────────────────────────
        echo ""
        echo "##################################################"
        echo "  A5 Expert Count Sweep: ${#A5_EXPERT_COUNTS[@]} jobs"
        echo "  Expert counts: ${A5_EXPERT_COUNTS[*]}"
        echo "##################################################"
        for NUM_EXPERTS in "${A5_EXPERT_COUNTS[@]}"; do
            submit_single_job \
                "A5_E${NUM_EXPERTS}" \
                "$SCRIPT_PATH" \
                "$EVAL_OUT_BASE/A5/E${NUM_EXPERTS}" \
                "$TIME" \
                "$MEM" \
                "$EFFECTIVE_PARTITION" \
                "--num-experts ${NUM_EXPERTS}"
        done

    elif [[ "$ABLATION" == "grid" ]]; then
        # ── grid (M0): one job per (k_shot, n_way) cell ───────────────────────
        local_n_cells=$(( ${#GRID_K_SHOTS[@]} * ${#GRID_N_WAYS[@]} ))
        echo ""
        echo "##################################################"
        echo "  Few-Shot Grid (M0/MAML+MoE): ${local_n_cells} jobs"
        echo "  k_shot: ${GRID_K_SHOTS[*]}"
        echo "  n_way : ${GRID_N_WAYS[*]}"
        echo "  Note  : (k=1, n=3) cell is identical to M0 by construction."
        echo "##################################################"
        for K in "${GRID_K_SHOTS[@]}"; do
            for N in "${GRID_N_WAYS[@]}"; do
                submit_single_job \
                    "grid_k${K}_n${N}" \
                    "$SCRIPT_PATH" \
                    "$EVAL_OUT_BASE/grid/k${K}_n${N}" \
                    "$TIME" \
                    "$MEM" \
                    "$EFFECTIVE_PARTITION" \
                    "--k-shot ${K} --n-way ${N}"
            done
        done

    elif [[ "$ABLATION" == "grid_A2" ]]; then
        # ── grid_A2 (A2/supervised): one job per (k_shot, n_way) cell ─────────
        local_n_cells=$(( ${#GRID_K_SHOTS[@]} * ${#GRID_N_WAYS[@]} ))
        echo ""
        echo "##################################################"
        echo "  Few-Shot Grid (A2/No-MAML No-MoE): ${local_n_cells} jobs"
        echo "  k_shot: ${GRID_K_SHOTS[*]}"
        echo "  n_way : ${GRID_N_WAYS[*]}"
        echo "  Note  : (k=1, n=3) cell is identical to A2 by construction."
        echo "  Eval  : both head_only and full fine-tuning per cell."
        echo "##################################################"
        for K in "${GRID_K_SHOTS[@]}"; do
            for N in "${GRID_N_WAYS[@]}"; do
                submit_single_job \
                    "grid_A2_k${K}_n${N}" \
                    "$SCRIPT_PATH" \
                    "$EVAL_OUT_BASE/grid_A2/k${K}_n${N}" \
                    "$TIME" \
                    "$MEM" \
                    "$EFFECTIVE_PARTITION" \
                    "--k-shot ${K} --n-way ${N}"
            done
        done

    elif [[ "$ABLATION" == "A2" ]]; then
        submit_single_job \
            "$ABLATION" \
            "$SCRIPT_PATH" \
            "$EVAL_OUT_BASE/$ABLATION" \
            "$TIME" \
            "$MEM" \
            "$EFFECTIVE_PARTITION" \
            "${TEST_PROCEDURE_ARG:+--test-procedure ${TEST_PROCEDURE_ARG}}"

    elif [[ "$ABLATION" == "A3" || "$ABLATION" == "A4" ]]; then
        # ── A3 / A4: shared script, --ablation flag selects the variant ───────
        submit_single_job \
            "$ABLATION" \
            "$SCRIPT_PATH" \
            "$EVAL_OUT_BASE/$ABLATION" \
            "$TIME" \
            "$MEM" \
            "$EFFECTIVE_PARTITION" \
            "--ablation ${ABLATION}"

    elif [[ "$ABLATION" == "A7" || "$ABLATION" == "A8" ]]; then
        # ── A7 / A8: shared script, --ablation flag selects the variant ───────
        submit_single_job \
            "$ABLATION" \
            "$SCRIPT_PATH" \
            "$EVAL_OUT_BASE/$ABLATION" \
            "$TIME" \
            "$MEM" \
            "$EFFECTIVE_PARTITION" \
            "--ablation ${ABLATION}"

    elif [[ "$ABLATION" == "A10" || "$ABLATION" == "A11" || "$ABLATION" == "A12" ]]; then
        submit_single_job \
            "$ABLATION" \
            "$SCRIPT_PATH" \
            "$EVAL_OUT_BASE/$ABLATION" \
            "$TIME" \
            "$MEM" \
            "$EFFECTIVE_PARTITION" \
            "--ablation ${ABLATION}"

    elif [[ "$ABLATION" == "steps_M0" ]]; then
        # ── steps_M0: MAML adaptation steps sweep (paper curve, no training) ──
        # Alpha fixed to Trial 89 HPO best. No --sweep-alpha needed.
        submit_single_job \
            "steps_M0" \
            "$SCRIPT_PATH" \
            "$EVAL_OUT_BASE/steps_sweep/M0" \
            "$TIME" \
            "$MEM" \
            "$EFFECTIVE_PARTITION" \
            "--model-type maml --ablation-id M0 --checkpoint $STEPS_M0_CHECKPOINT --alpha $STEPS_M0_ALPHA"

    elif [[ "$ABLATION" == "steps_A2" ]]; then
        # ── steps_A2: A2 trained inline (~2 min), then adaptation steps sweep ──
        # No --checkpoint — the Python script trains A2 from scratch and saves
        # the checkpoint to the output dir as a side effect.
        # ft_lr defaults to 0.001 in the sweep script (SUPERVISED_FT_LR).
        submit_single_job \
            "steps_A2" \
            "$SCRIPT_PATH" \
            "$EVAL_OUT_BASE/steps_sweep/A2" \
            "$TIME" \
            "$MEM" \
            "$EFFECTIVE_PARTITION" \
            "--model-type a2 --ablation-id A2 --ft-mode $RESOLVED_FT_MODE"

    elif [[ "$ABLATION" == "steps_A7" ]]; then
        # ── steps_A7: supervised adaptation steps sweep (paper curve, no training) ──
        # ft_lr defaults to 0.001 in the sweep script (SUPERVISED_FT_LR).
        submit_single_job \
            "steps_A7" \
            "$SCRIPT_PATH" \
            "$EVAL_OUT_BASE/steps_sweep/A7" \
            "$TIME" \
            "$MEM" \
            "$EFFECTIVE_PARTITION" \
            "--model-type supervised --ablation-id A7 --checkpoint $STEPS_A7_CHECKPOINT --ft-mode $RESOLVED_FT_MODE"

    elif [[ "$ABLATION" == "steps_A11" ]]; then
        # ── steps_A11: Meta pretrained steps sweep (paper curve, no training) ──
        # No --checkpoint arg — MetaEMGWrapper loads from its hardcoded path.
        # ft_lr defaults to 0.01 in the sweep script (A11_FT_LR).
        submit_single_job \
            "steps_A11" \
            "$SCRIPT_PATH" \
            "$EVAL_OUT_BASE/steps_sweep/A11" \
            "$TIME" \
            "$MEM" \
            "$EFFECTIVE_PARTITION" \
            "--model-type a11 --ablation-id A11 --ft-mode $RESOLVED_FT_MODE"

    else
        # ── All other ablations: single job, no extra CLI args ─────────────────
        submit_single_job \
            "$ABLATION" \
            "$SCRIPT_PATH" \
            "$EVAL_OUT_BASE/$ABLATION" \
            "$TIME" \
            "$MEM" \
            "$EFFECTIVE_PARTITION"
    fi
done

echo ""
echo "Done. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Log locations:"
echo "  $LOG_DIR/eval_<ID>_<jobid>.out"
echo ""