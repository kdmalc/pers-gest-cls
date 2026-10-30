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
#   A6_td_knn A6_td_lda A6_td_lda_pool          (Hudgins time-domain features; use the td64 EMG dict)
#                                               (classical baselines on the cached episodes, CPU only,
#                                                all 9 cells; need pyriemann=0.9 and aeon in the env)
#   A15_N4 A15_N8 A15_N12 A15_N16 A15_N20       (meta-training-user sweep; N=20 = full pool)
#   portB_head                                  (MoEMeta Port B; trains, 1 config)
#   grid_M0 grid_A2 grid_A4                     (k in {1,3,5} x n in {3,5,10} = 9 cells per fold;
#                                                63 jobs per token with all folds -- pass --folds!)
#
#   A14_M0 A14_A4                               (ProtoNet / kNN readout on the fold's frozen M0 / A4 backbone;
#                                                submit AFTER that fold's M0 / A4 job has finished)
#   V_<token>                                   (architecture-exploration finalists: V_X2 ... V_M0_ANIL, see
#                                                ARCH_VARIANT_TOKENS below; arch_variant_eval.py, all 9 cells)
#   Z_all Z3                                    (zero-shot cross-user; Z3 after the fold's M0 job)
#   A16_A2 A16_M0                               (upper bound; after the fold's A2 / M0 job)
#
# --seeds 42 43 44 : one job per seed for the scripts that take --seed (headline 3-seed runs).
# --emg-repr env20 | fs64 | raw3s (raw3s = native-rate EMG through the learned stem; EMG-RAW ablation)
# NOT on the shared episode cache yet, so deliberately NOT launchable (the aggregator would reject
# their results): A7/A8 (subject-specific), portA, A16.
# NOT wired yet (use only after the pending work in docs/archive/FULL_RUNNING_LOG.md):
#   A10/A11/A12/A11b (2 kHz + Kaifosh, waits on the O1 design), A16 (needs fold checkpoints
#   manifest), A5, portA, steps sweeps (must use the dev users P005 P131 P116 P109).
#
# A14 with encoder / portA need the M0 checkpoint of the SAME fold. Output layout:
#   $EVAL_OUT_BASE/kfold7/<TOKEN>/<EMG_REPR>/fold<k>/   (results JSON + checkpoints; EMG_REPR = --emg-repr,
#                                                     or the token's own --emg-repr, e.g. td64 for A6_td_*)
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
SEEDS=()
N_FOLDS=7

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --debug)     DEBUG=true; shift ;;
        --partition) PARTITION_ARG="$2"; shift 2 ;;
        --cluster)   CLUSTER="$2"; shift 2 ;;
        --hpo-config) HPO_CONFIG_ARG="$2"; shift 2 ;;
        --emg-repr)  EMG_REPR_ARG="$2"; shift 2 ;;
        --seeds)
            shift
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do SEEDS+=("$1"); shift; done
            ;;
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
[[ ${#SEEDS[@]} -gt 0 ]] || SEEDS=(42)

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
SPEC[A13_both]="A13_modality_ablation.py|--condition both --cells all|12:00:00|32G"
SPEC[A13_emg_only]="A13_modality_ablation.py|--condition emg_only --cells all|12:00:00|32G"
SPEC[A13_imu_only]="A13_modality_ablation.py|--condition imu_only --cells all|12:00:00|32G"
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
# Reviewer-requested classical baselines on Hudgins time-domain features (needs segfilt_td64_tensor_dict.pkl).
for m in td_knn td_lda td_lda_pool; do
    SPEC[A6_$m]="A6_classical_baselines.py|--method $m --emg-repr td64 --cells all|03:00:00|16G|cpu"
done
# Architecture-exploration tokens (arch_variants/registry.py ARCH_VARIANTS; finalists of the dev-user screen only).
# Launcher token V_<token>; tests/test_arch_variants.py checks this list against the registry.
ARCH_VARIANT_TOKENS="X2 X3 X4 X5 X6 F1 F2 F3 T0 T2 T4 P1 P1_moe P1_match P1_rel P1_ridge P1_feat P3 A4_ANIL M0_ANIL"
for t in $ARCH_VARIANT_TOKENS; do
    SPEC[V_$t]="arch_variant_eval.py|--token $t --cells all|12:00:00|32G"
done
SPEC[V_P3]="arch_variant_eval.py|--token P3 --cells all|03:00:00|24G"
# Zero-shot cross-user (SPEC_zeroshot.md): Z1 + Z2 + Z4 train their own models; Z3 reads the fold's M0 checkpoint.
SPEC[Z_all]="Z_zeroshot_crossuser.py|--models Z1 Z2 Z4|04:00:00|32G"
SPEC[Z3]="Z_zeroshot_crossuser.py|@CKPT:M0 --models Z3|02:00:00|32G"
# Upper bound (ARaC Q2): fine-tune the fold's A2 / M0 checkpoint on 9 of each test user's reps.
SPEC[A16_A2]="A16_upper_bound_kfold.py|@CKPT:A2 --base A2|06:00:00|32G"
SPEC[A16_M0]="A16_upper_bound_kfold.py|@CKPT:M0 --base M0|12:00:00|32G"
# Scripts that take --seed (kfold_launcher --seeds S1 S2 ... fans out one job per seed); the others are
# deterministic or read a seed-42 checkpoint and are submitted once.
SEEDABLE_SCRIPTS=" M0_full_model.py A1_no_maml_moe.py A2_no_maml_no_moe.py A3_A4_maml_no_moe.py A13_modality_ablation.py arch_variant_eval.py Z_zeroshot_crossuser.py "
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
    # the EMG representation is part of the output path: env20 and fs64 runs of one token must never share a folder
    local repr="$EMG_REPR_ARG"
    if [[ "$extra" =~ --emg-repr[[:space:]]+([a-z0-9]+) ]]; then repr="${BASH_REMATCH[1]}"; fi
    local out_dir="$EVAL_OUT_BASE/kfold7/$token/$repr/fold$fold"
    local name="kfold7_${token}${tag}_${repr}_f${fold}"
    local partition="${PARTITION_ARG:-commons}"
    if [[ "$DEBUG" == true ]]; then
        partition="debug"; time="00:15:00"
        # --smoke (1 epoch x 48 episodes) so train -> val -> TEST -> save all run inside 15 min
        case "$script" in M0_full_model.py|A3_A4_maml_no_moe.py|arch_variant_eval.py|Z_zeroshot_crossuser.py|A16_upper_bound_kfold.py) extra="$extra --smoke" ;; esac
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
    IFS='|' read -r script extra0 time mem kind <<< "${SPEC[$token]}"
    kind="${kind:-gpu}"
    if [[ "$SEEDABLE_SCRIPTS" == *" $script "* ]]; then token_seeds=("${SEEDS[@]}"); else token_seeds=(42); fi
    if [[ "$SEEDABLE_SCRIPTS" != *" $script "* && "${SEEDS[*]}" != "42" ]]; then
        echo "NOTE: $token ($script) takes no --seed; submitted once"
    fi
    for seed in "${token_seeds[@]}"; do
    extra="$extra0"; seed_tag=""
    if [[ "$SEEDABLE_SCRIPTS" == *" $script "* ]]; then extra="$extra --seed $seed"; seed_tag="_s$seed"; fi
    for fold in "${FOLDS[@]}"; do
        if [[ "$extra0" == "@GRID" ]]; then
            for k in 1 3 5; do for n in 3 5 10; do
                submit_job "$token" "$fold" "$script" "--k-shot $k --n-way $n" "$time" "$mem" "_k${k}n${n}" "" "$kind"
            done; done
        elif [[ "$extra0" == @CKPT:* ]]; then
            # checkpoint of the same fold (seed 42) from the named run; the job fails loudly if absent.
            # "@CKPT:SRC" alone = the A14 argument form; "@CKPT:SRC <args>" = <args> --checkpoint <ckpt>.
            # reads the seed-42 checkpoint: submitted once, not once per --seeds entry
            [[ "$seed" == "${token_seeds[0]}" ]] || continue
            rest="${extra0#@CKPT:}"; src="${rest%% *}"; args=""
            [[ "$rest" == *" "* ]] && args="${rest#* }"
            ckpt_glob="$EVAL_OUT_BASE/kfold7/$src/$EMG_REPR_ARG/fold$fold/${src}_*_kfold7_fold${fold}_seed42_best.pt"
            if [[ -z "$args" ]]; then args="--model $src --cells all"; fi
            submit_job "$token" "$fold" "$script" "$args --checkpoint \"\$CKPT\"" "$time" "$mem" "" "$ckpt_glob" "$kind"
        else
            submit_job "$token" "$fold" "$script" "$extra" "$time" "$mem" "$seed_tag" "" "$kind"
        fi
    done
    done
done
