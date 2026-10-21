#!/bin/bash
# kfold_launcher.sh
# =================
# Submit evaluation jobs for the kfold7 protocol (28/4 outer split, 7 folds of 20/4/4;
# see system/fixed_user_splits/kfold28_hpo4_split.json). One SLURM job per (token, fold),
# each calling the token's script with --fold-idx <k>. Replaces the L2SO fan-out in
# eval_launcher.sh, which is LEGACY (do not use it for new numbers).
#
# Usage:
#   bash kfold_launcher.sh TOKEN [TOKEN ...] --hpo-config {legacy|frozen|tuned} [--folds 0 1 2 ...]
#                          [--dry-run] [--debug]
#                          [--partition P] [--cluster NOTS|RANGE]
#
# Examples:
#   bash kfold_launcher.sh M0 --folds 0 --debug --dry-run   # inspect the sbatch command
#   bash kfold_launcher.sh M0 --folds 0 --debug             # 15-min debug-partition smoke, fold 0
#   bash kfold_launcher.sh M0 A1 A2 A3 A4                   # all 7 folds of each (full run)
#
# Tokens (script, extra args):
#   M0 A1 A2 A3 A4
#   A13_both A13_emg_only A13_imu_only          (modality ablation, 3-way default)
#   A6_dollar_b A6_dollar_b_user A6_knn_l1 A6_proto_euclid A6_minirocket A6_riemann_mdm A6_riemann_ts
#                                               (classical baselines on the cached episodes, CPU only,
#                                                all 9 cells; need pyriemann=0.9 and aeon in the env)
#   A15_N4 A15_N8 A15_N12 A15_N16 A15_N20       (meta-training-user sweep; N=20 = full pool)
#   portB_head                                  (MoEMeta Port B; trains, 1 config)
#   grid_M0 grid_A2 grid_A4                     (k in {1,3,5} x n in {3,5,10} = 9 cells per fold;
#                                                63 jobs per token with all folds -- pass --folds!)
#
#   A14_M0 A14_A4                               (ProtoNet / kNN readout on the fold's frozen M0 / A4 backbone;
#                                                submit AFTER that fold's M0 / A4 job has finished)
# NOT on the shared episode cache yet, so deliberately NOT launchable (the aggregator would reject
# their results): A7/A8 (subject-specific), portA, A16.
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
HPO_CONFIG_ARG=""
EMG_REPR_ARG="env20"   # env20 | fs64 (needs segfilt_fs64_tensor_dict.pkl next to the env20 dict)
TOKENS=()
FOLDS=()
N_FOLDS=7

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --debug)     DEBUG=true; shift ;;
        --partition) PARTITION_ARG="$2"; shift 2 ;;
        --cluster)   CLUSTER="$2"; shift 2 ;;
        --hpo-config) HPO_CONFIG_ARG="$2"; shift 2 ;;
        --emg-repr)  EMG_REPR_ARG="$2"; shift 2 ;;
        --folds)
            shift
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do FOLDS+=("$1"); shift; done
            ;;
        -*) echo "ERROR: unknown flag $1"; exit 1 ;;
        *)  TOKENS+=("$1"); shift ;;
    esac
done

case "$HPO_CONFIG_ARG" in legacy|frozen|tuned) ;; *)
    echo "ERROR: --hpo-config legacy|frozen|tuned is required (legacy = Trial-89, smoke only)"; exit 1 ;;
esac
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
    # sub2 = the resubmission runs; HPO lives next to it in /scratch/my13/kai/runs/paper/sub2/hpo
    EVAL_OUT_BASE=/scratch/my13/kai/runs/paper/sub2/eval
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

# Launchers are COPIED to scratch and run from there, but the repo lives under projects ($CODE_DIR).
# Provenance (git revision) is read from the repo at $CODE_DIR, at submit time, and exported into
# the job (compute nodes have no git). A stale scratch copy of this script is refused.
REPO_LAUNCHER="$CODE_DIR/system/NOTS/paper/ablations/launchers/$(basename "${BASH_SOURCE[0]}")"
if ! cmp -s "${BASH_SOURCE[0]}" "$REPO_LAUNCHER"; then
    echo "ERROR: this launcher differs from the repo copy (stale?). Refresh it with:"
    echo "  cp $REPO_LAUNCHER $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"
    exit 1
fi
GIT_REV="$(git -C "$CODE_DIR" rev-parse --short HEAD)" || { echo "ERROR: cannot read git revision of $CODE_DIR"; exit 1; }
if ! git -C "$CODE_DIR" diff --quiet HEAD; then GIT_REV="${GIT_REV}-dirty"; fi
echo "Submitting revision: $GIT_REV"

ABL_DIR=$CODE_DIR/system/NOTS/paper/ablations/test_eval_files
CPUS=10

# token -> "script|extra args|time|mem[|cpu]"   (--fold-idx <k> is appended per job; `cpu` = no GPU)
declare -A SPEC
SPEC[M0]="M0_full_model.py||12:00:00|32G"
SPEC[A1]="A1_no_maml_moe.py||02:00:00|24G"
SPEC[A2]="A2_no_maml_no_moe.py||02:00:00|16G"
SPEC[A3]="A3_A4_maml_no_moe.py|--ablation A3|12:00:00|24G"
SPEC[A4]="A3_A4_maml_no_moe.py|--ablation A4|12:00:00|24G"
SPEC[A13_both]="A13_modality_ablation.py|--condition both|12:00:00|32G"
SPEC[A13_emg_only]="A13_modality_ablation.py|--condition emg_only|12:00:00|32G"
SPEC[A13_imu_only]="A13_modality_ablation.py|--condition imu_only|12:00:00|32G"
for n in 4 8 12 16 20; do
    SPEC[A15_N$n]="A15_metatrain_user_sweep.py|--n-users $n|12:00:00|32G"
done
# ProtoNet / kNN readout on the frozen backbone of the SAME fold's checkpoint (run after M0 / A4).
SPEC[A14_M0]="A14_protonet_encoder.py|@CKPT:M0|03:00:00|32G"
SPEC[A14_A4]="A14_protonet_encoder.py|@CKPT:A4|03:00:00|32G"
SPEC[portB_head]="portB_local_adaptation.py|--local-module head|12:00:00|32G"
# Classical / non-parametric baselines on the cached episodes (CPU only, all 9 cells).
for m in dollar_b dollar_b_user knn_l1 proto_euclid; do
    SPEC[A6_$m]="A6_classical_baselines.py|--method $m --cells all|03:00:00|16G|cpu"
done
SPEC[A6_minirocket]="A6_classical_baselines.py|--method minirocket --cells all|06:00:00|16G|cpu"
SPEC[A6_riemann_mdm]="A6_classical_baselines.py|--method riemann_mdm --cells all|23:00:00|16G|cpu"
SPEC[A6_riemann_ts]="A6_classical_baselines.py|--method riemann_ts --cells all|23:00:00|16G|cpu"
SPEC[grid_M0]="fewshot_grid.py|@GRID|12:00:00|32G"
SPEC[grid_A2]="fewshot_grid_A2.py|@GRID|02:00:00|16G"
SPEC[grid_A4]="fewshot_grid_A4.py|@GRID|12:00:00|32G"

submit_job() {
    local token="$1" fold="$2" script="$3" extra="$4" time="$5" mem="$6" tag="$7" ckpt_glob="${8:-}" kind="${9:-gpu}"
    local ckpt_block=""
    if [[ -n "$ckpt_glob" ]]; then
        ckpt_block="CKPT=\$(ls -t $ckpt_glob 2>/dev/null | head -n 1)
[[ -n \"\$CKPT\" ]] || { echo \"JOB_FAILED: no M0 checkpoint matching $ckpt_glob\"; exit 3; }
echo \"CKPT      : \$CKPT\""
    fi
    local out_dir="$EVAL_OUT_BASE/kfold7/$token/fold$fold"
    local name="kfold7_${token}${tag}_f${fold}"
    local partition="${PARTITION_ARG:-commons}"
    if [[ "$DEBUG" == true ]]; then
        partition="debug"; time="00:15:00"
        # --smoke (1 epoch x 48 episodes) so train -> val -> TEST -> save all run inside 15 min
        case "$script" in M0_full_model.py|A3_A4_maml_no_moe.py) extra="$extra --smoke" ;; esac
    fi

    local wrap_body
    wrap_body=$(cat <<WRAPEOF
$MODULE_LOAD_BLOCK
mamba activate $ENV_PATH
export RUN_DIR=$out_dir
mkdir -p "\$RUN_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "JOB_START host=\$(hostname) date=\$(date) jobid=\${SLURM_JOB_ID}"
export GIT_REV=$GIT_REV
export HPO_CONFIG=$HPO_CONFIG_ARG
export EMG_REPR=$EMG_REPR_ARG
echo "GIT_REV   : \$GIT_REV"
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
    local gres=(--gres=gpu:1)
    if [[ "$kind" == "cpu" ]]; then gres=(); fi
    local cmd=(sbatch --job-name="$name" --partition="$partition" --nodes=1 --ntasks=1
        --cpus-per-task="$CPUS" --mem="$mem" --time="$time" "${gres[@]}"
        --output="$LOG_DIR/%x_%j.out"
        --export="ALL,CODE_DIR=$CODE_DIR,DATA_DIR=$DATA_DIR,PYTHONPATH=$CODE_DIR:$CODE_DIR/system/MAML:$CODE_DIR/system/MOE:${PYTHONPATH:-}"
        --wrap="$wrap_body")
    echo "== $name  [$partition, $time, $mem]  -> $out_dir"
    if [[ "$DRY_RUN" == true ]]; then
        echo "   [DRY RUN] ${cmd[*]:0:11} ... (python -u $script $extra --fold-idx $fold)"
        if [[ "${SHOW_WRAP:-0}" == 1 ]]; then echo "$wrap_body"; fi
    else
        mkdir -p "$LOG_DIR" "$out_dir"
        "${cmd[@]}"
    fi
}

for token in "${TOKENS[@]}"; do
    [[ -n "${SPEC[$token]:-}" ]] || { echo "ERROR: unknown token '$token'. See header."; exit 1; }
    IFS='|' read -r script extra time mem kind <<< "${SPEC[$token]}"
    kind="${kind:-gpu}"
    for fold in "${FOLDS[@]}"; do
        if [[ "$extra" == "@GRID" ]]; then
            for k in 1 3 5; do for n in 3 5 10; do
                submit_job "$token" "$fold" "$script" "--k-shot $k --n-way $n" "$time" "$mem" "_k${k}n${n}" "" "$kind"
            done; done
        elif [[ "$extra" == @CKPT:* ]]; then
            # checkpoint of the same fold from the named run; the job fails loudly if absent.
            src="${extra#@CKPT:}"
            ckpt_glob="$EVAL_OUT_BASE/kfold7/$src/fold$fold/${src}_*_kfold7_fold${fold}_seed42_best.pt"
            submit_job "$token" "$fold" "$script" "--model $src --cells all --checkpoint \"\$CKPT\"" "$time" "$mem" "" "$ckpt_glob" "$kind"
        else
            submit_job "$token" "$fold" "$script" "$extra" "$time" "$mem" "" "" "$kind"
        fi
    done
done
