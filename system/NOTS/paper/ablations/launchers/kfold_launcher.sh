#!/bin/bash
# kfold_launcher.sh
# =================
# Submit evaluation jobs for the kfold7 protocol (28/4 outer split, 7 folds of 20/4/4;
# see system/fixed_user_splits/kfold28_hpo4_split.json). One SLURM job per (token, fold),
# each calling the token's script with --fold-idx <k>. Replaces the L2SO fan-out in
# eval_launcher.sh, which is LEGACY (do not use it for new numbers).
#
# Usage:
#   bash kfold_launcher.sh TOKEN [TOKEN ...] [--folds 0 1 2 ...] [--dry-run] [--debug]
#                          [--partition P] [--cluster NOTS|RANGE]
#
# Examples:
#   bash kfold_launcher.sh M0 --folds 0 --debug --dry-run   # inspect the sbatch command
#   bash kfold_launcher.sh M0 --folds 0 --debug             # 15-min debug-partition smoke, fold 0
#   bash kfold_launcher.sh M0 A1 A2 A3 A4                   # all 7 folds of each (full run)
#
# Tokens (script, extra args):
#   M0 A1 A2 A3 A4 A7 A8
#   A13_both A13_emg_only A13_imu_only          (modality ablation, 3-way default)
#   A14_noenc                                   (ProtoNet/kNN/PCA raw+PCA tracks, no checkpoint)
#   A14                                         (adds proto_encoded; needs M0 fold checkpoint, see below)
#   A15_N4 A15_N8 A15_N12 A15_N16 A15_N20       (meta-training-user sweep; N=20 = full pool)
#   portB_head                                  (MoEMeta Port B; trains, 1 config)
#   grid_M0 grid_A2 grid_A4                     (k in {1,3,5} x n in {3,5,10} = 9 cells per fold;
#                                                63 jobs per token with all folds -- pass --folds!)
#
# NOT wired yet (use only after the pending work in docs/FULL_RUNNING_LOG.md):
#   A10/A11/A12/A11b (2 kHz + Kaifosh, waits on the O1 design), A16 (needs fold checkpoints
#   manifest), A5, portA, steps sweeps (must use the dev users P005 P131 P116 P109).
#
# A14 with encoder / portA need the M0 checkpoint of the SAME fold. Output layout:
#   $EVAL_OUT_BASE/kfold7/<TOKEN>/fold<k>/        (results JSON + checkpoints)
#   $LOG_DIR/kfold7_<TOKEN>_f<k>_<jobid>.out

set -euo pipefail

CLUSTER="NOTS"
DRY_RUN=false
DEBUG=false
PARTITION_ARG=""
TOKENS=()
FOLDS=()
N_FOLDS=7

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --debug)     DEBUG=true; shift ;;
        --partition) PARTITION_ARG="$2"; shift 2 ;;
        --cluster)   CLUSTER="$2"; shift 2 ;;
        --folds)
            shift
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do FOLDS+=("$1"); shift; done
            ;;
        -*) echo "ERROR: unknown flag $1"; exit 1 ;;
        *)  TOKENS+=("$1"); shift ;;
    esac
done

[[ ${#TOKENS[@]} -gt 0 ]] || { echo "ERROR: no tokens given. See header for usage."; exit 1; }
if [[ ${#FOLDS[@]} -eq 0 ]]; then
    for ((k = 0; k < N_FOLDS; k++)); do FOLDS+=("$k"); done
fi
for k in "${FOLDS[@]}"; do
    (( k >= 0 && k < N_FOLDS )) || { echo "ERROR: fold $k not in [0, $N_FOLDS)"; exit 1; }
done

if [[ "$CLUSTER" == "NOTS" ]]; then
    CODE_DIR=/projects/my13/kai/meta-pers-gest/pers-gest-cls
    DATA_DIR=/scratch/my13/kai/meta-pers-gest/data
    EVAL_OUT_BASE=/scratch/my13/kai/runs/paper/ablations/eval
    LOG_DIR=$EVAL_OUT_BASE/logs
    ENV_PATH=/projects/my13/kai/meta-pers-gest/envs/fl-torch
elif [[ "$CLUSTER" == "RANGE" ]]; then
    CODE_DIR=/home/km82/pers-gest-cls
    DATA_DIR=/home/km82/pers-gest-cls/dataset/meta-learning-sup-que-ds
    EVAL_OUT_BASE=$SHARED_SCRATCH/$USER/runs/paper/ablations/eval
    LOG_DIR=$EVAL_OUT_BASE/logs
    ENV_PATH=/home/km82/envs/fl_torch
else
    echo "ERROR: --cluster must be NOTS or RANGE, got '$CLUSTER'"; exit 1
fi
MODULE_LOAD_BLOCK='source /etc/profile.d/modules.sh
module purge
module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh'

ABL_DIR=$CODE_DIR/system/NOTS/paper/ablations/test_eval_files
CPUS=10

# token -> "script|extra args|time|mem"     (--fold-idx <k> is appended per job)
declare -A SPEC
SPEC[M0]="M0_full_model.py||12:00:00|32G"
SPEC[A1]="A1_no_maml_moe.py||02:00:00|24G"
SPEC[A2]="A2_no_maml_no_moe.py||02:00:00|16G"
SPEC[A3]="A3_A4_maml_no_moe.py|--ablation A3|12:00:00|24G"
SPEC[A4]="A3_A4_maml_no_moe.py|--ablation A4|12:00:00|24G"
SPEC[A7]="A7_A8_subject_specific.py|--ablation A7|03:00:00|16G"
SPEC[A8]="A7_A8_subject_specific.py|--ablation A8|03:00:00|16G"
SPEC[A13_both]="A13_modality_ablation.py|--condition both|12:00:00|32G"
SPEC[A13_emg_only]="A13_modality_ablation.py|--condition emg_only|12:00:00|32G"
SPEC[A13_imu_only]="A13_modality_ablation.py|--condition imu_only|12:00:00|32G"
SPEC[A14_noenc]="A14_protonet_baseline.py|--no-encoder|03:00:00|32G"
SPEC[A14]="A14_protonet_baseline.py|@CKPT|03:00:00|32G"
for n in 4 8 12 16 20; do
    SPEC[A15_N$n]="A15_metatrain_user_sweep.py|--n-users $n|12:00:00|32G"
done
SPEC[portB_head]="portB_local_adaptation.py|--local-module head|12:00:00|32G"
SPEC[grid_M0]="fewshot_grid.py|@GRID|12:00:00|32G"
SPEC[grid_A2]="fewshot_grid_A2.py|@GRID|02:00:00|16G"
SPEC[grid_A4]="fewshot_grid_A4.py|@GRID|12:00:00|32G"

submit_job() {
    local token="$1" fold="$2" script="$3" extra="$4" time="$5" mem="$6" tag="$7" ckpt_glob="${8:-}"
    local ckpt_block=""
    if [[ -n "$ckpt_glob" ]]; then
        ckpt_block="CKPT=\$(ls -t $ckpt_glob 2>/dev/null | head -n 1)
[[ -n \"\$CKPT\" ]] || { echo \"JOB_FAILED: no M0 checkpoint matching $ckpt_glob\"; exit 3; }
echo \"CKPT      : \$CKPT\""
    fi
    local out_dir="$EVAL_OUT_BASE/kfold7/$token/fold$fold"
    local name="kfold7_${token}${tag}_f${fold}"
    local partition="${PARTITION_ARG:-commons}"
    if [[ "$DEBUG" == true ]]; then partition="debug"; time="00:15:00"; fi

    local wrap_body
    wrap_body=$(cat <<WRAPEOF
$MODULE_LOAD_BLOCK
mamba activate $ENV_PATH
export RUN_DIR=$out_dir
mkdir -p "\$RUN_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "JOB_START host=\$(hostname) date=\$(date) jobid=\${SLURM_JOB_ID}"
echo "GIT_REV   : \$(cd $CODE_DIR && git rev-parse --short HEAD)"
echo "TOKEN     : $token  fold=$fold  tag=$tag"
echo "SCRIPT    : $ABL_DIR/$script $extra --fold-idx $fold"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
$ckpt_block
set +e
python -u $ABL_DIR/$script $extra --fold-idx $fold
PY_RC=\$?
set -e
echo "PYTHON_EXIT_CODE=\${PY_RC}"
echo "JOB_END date=\$(date) rc=\${PY_RC}"
[[ \${PY_RC} -ne 0 ]] && echo "JOB_FAILED: python exited \${PY_RC} -- no valid results."
exit \${PY_RC}
WRAPEOF
)
    local cmd=(sbatch --job-name="$name" --partition="$partition" --nodes=1 --ntasks=1
        --cpus-per-task="$CPUS" --mem="$mem" --time="$time" --gres=gpu:1
        --output="$LOG_DIR/%x_%j.out"
        --export="ALL,CODE_DIR=$CODE_DIR,DATA_DIR=$DATA_DIR,PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-}"
        --wrap="$wrap_body")
    echo "== $name  [$partition, $time, $mem]  -> $out_dir"
    if [[ "$DRY_RUN" == true ]]; then
        echo "   [DRY RUN] ${cmd[*]:0:11} ... (python -u $script $extra --fold-idx $fold)"
        [[ "${SHOW_WRAP:-0}" == 1 ]] && echo "$wrap_body"
    else
        mkdir -p "$LOG_DIR" "$out_dir"
        "${cmd[@]}"
    fi
}

for token in "${TOKENS[@]}"; do
    [[ -n "${SPEC[$token]:-}" ]] || { echo "ERROR: unknown token '$token'. See header."; exit 1; }
    IFS='|' read -r script extra time mem <<< "${SPEC[$token]}"
    for fold in "${FOLDS[@]}"; do
        if [[ "$extra" == "@GRID" ]]; then
            for k in 1 3 5; do for n in 3 5 10; do
                submit_job "$token" "$fold" "$script" "--k-shot $k --n-way $n" "$time" "$mem" "_k${k}n${n}"
            done; done
        elif [[ "$extra" == "@CKPT" ]]; then
            # M0 checkpoint of the same fold; fail the job loudly if absent (no silent fallback).
            ckpt_glob="$EVAL_OUT_BASE/kfold7/M0/fold$fold/M0_*_kfold7_fold${fold}_seed42_best.pt"
            submit_job "$token" "$fold" "$script" '--checkpoint "$CKPT"' "$time" "$mem" "" "$ckpt_glob"
        else
            submit_job "$token" "$fold" "$script" "$extra" "$time" "$mem" ""
        fi
    done
done
