#!/bin/bash
# eval_ablation_launcher.sh
# =========================
# Submit final evaluation jobs for one or more ablation IDs.
#
# Most ablations are a single (non-array) SLURM job calling their dedicated
# Python script. Two exceptions spawn multiple parallel jobs:
#
#   A5   : sweeps num_experts over 8 values -> 8 jobs (see A5_EXPERT_COUNTS)
#   grid : k-shot x n-way grid -> 9 jobs (see GRID_K_SHOTS / GRID_N_WAYS)
#
# Script mapping:
#   M0   → M0_full_model.py
#   A1   → A1_no_maml_moe.py
#   A2   → A2_no_maml_no_moe.py
#   A4   → A3_A4_maml_no_moe.py
#   A5   → A5_expert_count_sweep.py      [one job per expert count]
#   A7   → A7_A8_subject_specific.py
#   A8   → A7_A8_subject_specific.py
#   A11  → A10_A11_A12_meta_pretrained.py
#   grid → fewshot_grid.py               [one job per (k_shot, n_way) cell]
#
# Usage:
#   bash eval_ablation_launcher.sh M0                    # eval M0
#   bash eval_ablation_launcher.sh A1 A2 A4              # multiple ablations
#   bash eval_ablation_launcher.sh all                   # all ablations (no grid)
#   bash eval_ablation_launcher.sh grid                  # k-shot/n-way grid (M0)
#   bash eval_ablation_launcher.sh all grid              # everything
#   bash eval_ablation_launcher.sh M0 --dry-run          # print without submitting
#   bash eval_ablation_launcher.sh A1 --debug            # debug partition, 15 min limit
#   bash eval_ablation_launcher.sh A5 --partition commons
#
# Output layout:
#   $EVAL_OUT_BASE/<ID>/              (M0, A1, A2, A4, A7, A8, A11)
#   $EVAL_OUT_BASE/A5/E<N>/           (A5, one subdir per expert count)
#   $EVAL_OUT_BASE/grid/k<K>_n<N>/   (grid, one subdir per cell)
#   $LOG_DIR/eval_<ID>_<jobid>.out

set -euo pipefail

# =============================================================================
# Paths — keep in sync with hpo_ablation_launcher.sh
# =============================================================================
CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
ABLATIONS_DIR="$CODE_DIR/system/NOTS/paper/ablations"
EVAL_OUT_BASE=/scratch/my13/kai/runs/paper/ablations/eval
LOG_DIR=/scratch/my13/kai/runs/paper/ablations/eval/logs

ENV_PATH=/projects/my13/kai/meta-pers-gest/envs/fl-torch

mkdir -p "$EVAL_OUT_BASE" "$LOG_DIR"

# =============================================================================
# A5 expert count sweep definition.
# MUST be kept in sync with EXPERT_COUNTS in A5_expert_count_sweep.py.
# =============================================================================
A5_EXPERT_COUNTS=(4 8 12 16 20 24 32 40)

# =============================================================================
# Few-shot grid definition.
# MUST be kept in sync with GRID_K_SHOTS / GRID_N_WAYS in fewshot_grid.py.
# =============================================================================
GRID_K_SHOTS=(1 3 5)
GRID_N_WAYS=(3 5 10)

# =============================================================================
# Ablation ID -> Python script mapping.
# A4, A7/A8, and A11 share scripts — the Python scripts read
# config["ablation_id"] to select the right variant.
# A5 and grid are handled separately in the submission loop.
# =============================================================================
declare -A ABLATION_SCRIPT
ABLATION_SCRIPT[M0]="M0_full_model.py"
ABLATION_SCRIPT[A1]="A1_no_maml_moe.py"
ABLATION_SCRIPT[A2]="A2_no_maml_no_moe.py"
ABLATION_SCRIPT[A4]="A3_A4_maml_no_moe.py"
ABLATION_SCRIPT[A5]="A5_expert_count_sweep.py"
ABLATION_SCRIPT[A7]="A7_A8_subject_specific.py"
ABLATION_SCRIPT[A8]="A7_A8_subject_specific.py"
ABLATION_SCRIPT[A11]="A10_A11_A12_meta_pretrained.py"
ABLATION_SCRIPT[grid]="fewshot_grid.py"

# "all" expands to the standard ablation set only — grid is opt-in.
VALID_ABLATIONS=(M0 A1 A2 A4 A5 A7 A8 A11)
ALL_TOKENS=(M0 A1 A2 A4 A5 A7 A8 A11 grid)  # for usage string

# =============================================================================
# Parse args
# =============================================================================
ABLATIONS=()
DRY_RUN=false
DEBUG=false
OVERRIDE_PARTITION=""

i=0
args_array=("$@")

while [[ $i -lt ${#args_array[@]} ]]; do
    arg="${args_array[$i]}"
    case "$arg" in
        --dry-run)   DRY_RUN=true ;;
        --debug)     DEBUG=true ;;
        --partition) i=$((i+1)); OVERRIDE_PARTITION="${args_array[$i]}" ;;
        all)         ABLATIONS+=("${VALID_ABLATIONS[@]}") ;;
        -*)          echo "WARNING: Unknown flag '$arg' -- ignoring." ;;
        *)           ABLATIONS+=("$arg") ;;
    esac
    i=$((i+1))
done

if [[ ${#ABLATIONS[@]} -eq 0 ]]; then
    echo "ERROR: No ablations specified."
    echo "Usage: bash eval_ablation_launcher.sh [$(IFS='|'; echo "${ALL_TOKENS[*]}")|all] [--dry-run] [--debug] [--partition PARTITION]"
    echo "       'all' expands to: ${VALID_ABLATIONS[*]}  (grid is opt-in, not included in 'all')"
    exit 1
fi

# Validate all requested tokens upfront so we fail fast before submitting anything.
for ABL in "${ABLATIONS[@]}"; do
    if [[ -z "${ABLATION_SCRIPT[$ABL]+x}" ]]; then
        echo "ERROR: Unknown ablation '$ABL'. Valid tokens: ${ALL_TOKENS[*]}"
        exit 1
    fi
done

# =============================================================================
# Cluster defaults
# =============================================================================
PARTITION=commons
CPUS=10
MEM_DEFAULT=32G
TIME_DEFAULT="08:00:00"

# =============================================================================
# Per-ablation resource overrides.
# Supervised ablations (A1, A2, A7, A11) are much faster than MAML ablations.
# MAML+MoE ablations (M0, A5, A8, grid) are the slowest.
# Adjust TIME_* based on observed wall times from HPO runs.
# =============================================================================
TIME_A1="01:00:00";   MEM_A1=24G
TIME_A2="01:00:00";   MEM_A2=16G
TIME_A7="01:00:00";   MEM_A7=16G
TIME_A8="01:00:00";   MEM_A8=16G
TIME_A11="01:00:00";  MEM_A11=24G
# Uncomment and tune once you have wall-time data from HPO:
# TIME_M0="08:00:00";   MEM_M0=32G
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

    EFFECTIVE_PARTITION="$PARTITION"
    if [[ "$DEBUG" == true ]]; then
        EFFECTIVE_PARTITION=debug
        TIME="00:15:00"
    fi
    if [[ -n "$OVERRIDE_PARTITION" ]]; then
        EFFECTIVE_PARTITION="$OVERRIDE_PARTITION"
    fi

    if [[ "$ABLATION" == "A5" ]]; then
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
        # ── grid: one job per (k_shot, n_way) cell ────────────────────────────
        local_n_cells=$(( ${#GRID_K_SHOTS[@]} * ${#GRID_N_WAYS[@]} ))
        echo ""
        echo "##################################################"
        echo "  Few-Shot Grid: ${local_n_cells} jobs"
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
echo "Aggregate A5 mountain curve results:"
echo "  python -c \""
echo "    import json, glob"
echo "    files = sorted(glob.glob('$EVAL_OUT_BASE/A5/E*/A5_E*_final_results.json'))"
echo "    for f in files:"
echo "      r = json.load(open(f))"
echo "      print(f\"{r['num_experts']:>4} experts  top_k={r['top_k']}  acc={r['test_results']['mean_acc']*100:.2f}%\")"
echo "  \""
echo ""
echo "Aggregate few-shot grid results:"
echo "  python -c \""
echo "    import json, glob"
echo "    files = sorted(glob.glob('$EVAL_OUT_BASE/grid/k*_n*/grid_*_final_results.json'))"
echo "    for f in files:"
echo "      r = json.load(open(f))"
echo "      print(f\"k={r['k_shot']} n={r['n_way']}  acc={r['test_results']['mean_acc']*100:.2f}%\")"
echo "  \""