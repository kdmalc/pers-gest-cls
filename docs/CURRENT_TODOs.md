# CURRENT TODOs
*Short list of open threads only; finished work goes to `docs/LOG.md`. Run tracker: `docs/REBUTTAL_CHECKLIST.md`. Top-level plan: `docs/PLAN_resubmission.md`. Docs index: `docs/README.md`. Architecture work: `docs/specs/SPEC_architecture_exploration.md`. Updated 2026-10-30, evening.*

**Goal:** resubmit EncoderMoE (rejected NeurIPS'26) on one clean protocol: 28 users, 7 folds, cached episodes, 10-way 1-shot primary. Add the reviewer-requested baselines and ablations. **Venue: ICML (Jan) or TMLR, decided after the main runs finish.**

## 0. What Claude is doing now
- **Nothing running.** Everything below is built and tested locally (386 tests pass); none of it has run on the cluster.
- **Next code, waiting for Kai's go:** nothing queued. Open options: F4, T1/T3, CNP, the GNI side of the raw-EMG comparison, ports of A7/A8/portA and Kaifosh (A10-A12) to the cache.

## 1. Plan (Kai, 2026-10-30 evening)
1. **Settle the MoE and fusion architecture first, on the dev users. No final HPO yet.**
   - Read the routing diagnostics (X1 vs X2) to see whether the MoE routes or just averages experts.
   - **A4 and A2 tuning stay on hold until EncoderMoE is final.**
   - The screen therefore freezes the shared training HPs at the **legacy Trial-89 values** (`system/NOTS/paper/ablations/screen_fixed/`), the same for every variant.
   - M0 and A4 are screened with the same frozen values, as the references.
2. **Budgets:**
   - **≥ 24 trials** per screened variant.
   - Finalists: 100 or 60 trials, decided from the debug timings (§2B).
3. **E and top-k:** a grid around the best MoE config, not TPE.
4. **PI sign-offs:**
   - fs64 for the main table;
   - building td64 (command in `LOG.md`, 2026-10-25; then the tokens `A6_td_knn A6_td_lda A6_td_lda_pool`);
   - reporting `$B` if it wins.

## 2. Cluster commands (Kai runs these)
**Every session starts with:**
```bash
cd /projects/my13/kai/meta-pers-gest/pers-gest-cls && git pull && git status --short
L=/projects/my13/kai/meta-pers-gest/pers-gest-cls/system/NOTS/paper/ablations/launchers
cp $L/kfold_launcher.sh $L/hpo_ablation_launcher.sh /scratch/my13/kai/runs/paper/sub2/
cd /scratch/my13/kai/runs/paper/sub2
module purge && module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh && source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba activate /projects/my13/kai/meta-pers-gest/envs/fl-torch
```

**A. Classical baselines (launched 2026-10-30, CPU).** When `squeue --me` shows no `kfold7_A6_*` job:
```bash
for f in /scratch/my13/kai/runs/paper/sub2/eval/logs/kfold7_A6_*.out; do rc=$(grep -o "PYTHON_EXIT_CODE=[0-9]*" "$f"); [[ "$rc" == "PYTHON_EXIT_CODE=0" ]] || echo "${rc:-NO_EXIT_LINE(timeout/killed)}  $f"; done   # empty = all OK
sacct -u km82 -S 2026-10-30 -X --format=JobID,JobName%34,State,Elapsed | grep -E "TIMEOUT|FAILED|OUT_OF_ME"                                          # empty = none timed out
```
Then copy the results to the laptop:
```bash
# on the laptop (Git Bash)
mkdir -p ~/gest_results/A6_cluster && scp -r "km82@<NOTS login host>:/scratch/my13/kai/runs/paper/sub2/eval/kfold7/A6_*" ~/gest_results/A6_cluster/
```
Claude then runs notebook 02 with `A6_RESULTS=~/gest_results/A6_cluster`.

**B. Debug trials of the new code: RE-RUN after this `git pull`.**
- **Bug found 2026-10-30 evening:** the HPO driver never applied `--emg-repr` to the training config, so the first debug batch (and any earlier "fs64" study) trained on the **env20** dict.
- That batch still shows crashes, but its epoch times are not fs64 times. Paste me its output anyway, then re-run on the fixed code.
```bash
bash hpo_ablation_launcher.sh A4 M0 X2 X3 X4 X6 T2 F1 F2 F3 T0 T4 P1 P1_moe P1_rel P1_feat P3 A4_ANIL --emg-repr fs64 --debug
```
When they finish, paste Claude the output of:
```bash
LOGS=/scratch/my13/kai/runs/paper/sub2/hpo/logs
for f in $(ls -t $LOGS/hpo_*.out | head -18); do echo "== $(basename $f)"; grep -m1 "emg_in_ch:" $f; grep -m1 "build_model\]" $f; grep -m2 -E "Epoch completed in|\[metric:.*epoch 1:" $f; grep -m1 -E "Traceback|Error" $f; done
```
- **Pass:** every log shows `emg_in_ch: 112` (fs64), `[build_model] ... parameters` and at least one epoch line, with no Traceback.

**C. Eval-path smoke tests of the new scripts** (debug partition, fold 0, legacy HPs; checks train → test → save end to end):
```bash
bash kfold_launcher.sh V_X2 V_P1 V_P3 Z_all M0 --folds 0 --debug --hpo-config legacy --emg-repr fs64
# after that M0 smoke finishes (they read its fold-0 checkpoint):
bash kfold_launcher.sh Z3 A16_M0 --folds 0 --debug --hpo-config legacy --emg-repr fs64
```
Pass: every `kfold7_*_f0_*.out` ends with `PYTHON_EXIT_CODE=0`.

**C2. The architecture screen**, after B passes. It uses ≥ 24 trials and the legacy shared HPs frozen.
```bash
F=/projects/my13/kai/meta-pers-gest/pers-gest-cls/system/NOTS/paper/ablations/screen_fixed
bash hpo_ablation_launcher.sh M0 A4 X2 X3 X4 X5 X6 F1 F2 F3 --emg-repr fs64 --n-trials 24 --study-suffix screen --fixed-params $F/legacy_shared_maml.json --dry-run
bash hpo_ablation_launcher.sh M0 A4 X2 X3 X4 X5 X6 F1 F2 F3 --emg-repr fs64 --n-trials 24 --study-suffix screen --fixed-params $F/legacy_shared_maml.json
bash hpo_ablation_launcher.sh T0 T2 T4 --emg-repr fs64 --n-trials 24 --study-suffix screen --fixed-params $F/legacy_inner_only.json
bash hpo_ablation_launcher.sh P1 P1_moe P1_match P1_rel P1_ridge P1_feat P3 --emg-repr fs64 --n-trials 24 --study-suffix screen
```

**C3. EMG-RAW dict (laptop; raw CSVs in Box):** about 0.6 GB.
```bash
cd ~/Repos/pers-gest-cls; PY=~/anaconda3/envs/fl_torch/python.exe
$PY system/universal_preprocessing/EMG_preprocessing/build_raw3s_tensor_dict.py --root "/c/Users/kdmen/Box/Yamagami Lab/Data/2024_UIST_dataset/upload/segmented_raw_data" --fused-dict dataset/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl --out dataset/meta-learning-sup-que-ds/segfilt_raw3s_tensor_dict.pkl
scp dataset/meta-learning-sup-que-ds/segfilt_raw3s_tensor_dict.pkl km82@<NOTS login host>:/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/
```
Then any token runs on it with `--emg-repr raw3s`.

**D. Status, timing and GPU-hours of any HPO, at any time:**
```bash
python /projects/my13/kai/meta-pers-gest/pers-gest-cls/system/NOTS/paper/ablations/hpo_tools.py report --db-dir /scratch/my13/kai/runs/paper/sub2/hpo/optuna_dbs
# per-trial detail for one study:  ... report --db-dir ... --trials ablation_A4_10w1s_kfold7_fs64_v1
# from the logs instead:           grep -h HPO_TRIAL_TIMING /scratch/my13/kai/runs/paper/sub2/hpo/logs/*.out
sacct -u km82 -S 2026-10-29 -X -n -P --format=JobID,ElapsedRaw,AllocTRES | \
  awk -F'|' '{ if (match($3,/gres\/gpu=[0-9]+/)) { n=substr($3,RSTART+9,RLENGTH-9); s+=$2*n } } END {printf "%.1f GPU-hours\n", s/3600}'
```

**E. Housekeeping still open from the 10/29 debug run:**
- run one `--smoke` M0 on fs64: `bash kfold_launcher.sh M0 --folds 0 --debug --hpo-config legacy --emg-repr fs64`;
- delete the non-smoke A1/A2 fold-0 JSONs under `.../eval/kfold7/{A1,A2}/fold0/`.

## 3. Code status (all local-tested; none cluster-tested yet)
| Item | Where | Launch |
|---|---|---|
| X2-X6, T2 (MoE variants in M0's class) | `MOE/MOE_encoder.py` | HPO tokens `X2`..`X6 T2`; eval `V_<token>` |
| F1-F3, T0, T4 | `arch_variants/flexnet.py` | `F1 F2 F3 T0 T4`; `V_<token>` |
| ProtoNet / Matching / Relation / R2-D2 / FEAT; ADANN; ANIL | `arch_variants/metric_learners.py`, `adann.py` | `P1* P3 *_ANIL`; `V_<token>` |
| One registry for every token (HPO, eval, launcher) | `arch_variants/registry.py` | |
| Routing diagnostics | `MOE/routing_metrics.py` (per HPO trial; per fold on val in `V_*`) | |
| Timing, `hpo_tools.py`, frozen HPs, study suffix, E×k enqueue | `ablation_hpo.py` | `--fixed-params --study-suffix --enqueue-json` |
| 7-fold eval of any token | `test_eval_files/arch_variant_eval.py` | `kfold_launcher.sh V_<token>` |
| EMG-RAW stem + raw3s builder | `pretrain_models.EMGRawStem`, `build_raw3s_tensor_dict.py` | `--emg-repr raw3s` (both launchers) |
| Zero-shot Z1-Z4 | `test_eval_files/Z_zeroshot_crossuser.py` | `Z_all`, `Z3` |
| Upper bound (kfold7, leave-one-rep-out) | `test_eval_files/A16_upper_bound_kfold.py` | `A16_A2`, `A16_M0` |
| `--seed` (M0, A1, A2, A3/A4, A13, V_*, Z) | `ablation_config.add_seed_arg` | `kfold_launcher.sh --seeds 42 43 44` |
| A13 at 10-way, all 9 cells | `A13_modality_ablation.py` | `A13_both A13_emg_only A13_imu_only` |
| **Fixes:** HPO ignored `--emg-repr`; A2 tuned at a different width than evaluated; collapse penalty zeroed balanced trials | | |

**Not built:**
- F4 and T1/T3;
- CNP;
- learned `$B` (only if R4 triggers);
- the GNI side of the raw-EMG comparison;
- ports of A7/A8/portA and Kaifosh A10-A12 to the cache.

## 4. Results so far
**Dev EMG decision (notebook 01):** fs64. At 10-way 1-shot, EMG-only gains +28.9 (proto) / +17.2 (1-NN) points over env20, on 4/4 users.

**28 users, 10-way 1-shot, fused input (notebook 02):**

| method | env20 | fs64 |
|---|---|---|
| kNN-L1 | 58.0 [54.0, 62.1] | **72.9** [69.4, 76.3] |
| ProtoNet (raw features) | 51.1 [46.8, 55.7] | 65.6 [61.5, 69.7] |

- fs64 is better for 28/28 users.
- **Bar for learned models:** about 78% on the dev users (1-NN L1).
- **R4 risk:** old M0 scored 67.7% (old protocol), which is below the untrained kNN.

**Uniqueness (notebook 03):**
- Cross-user zero-shot gets 13-16%, against 62-78% within-user.
- Executions recur across users under different labels.
- The ability-group effect is weak (Mantel r 0.05-0.09).

**Body-region Gate 0 (dev users, 1-NN, `specs/SPEC_body_regions.md` §4):** region experts dropped by the pre-stated rule. Anatomical groups do no better than random groups (81.3 vs 80.4, 94th percentile). Per-user selection helps (+5.3 over one shared subset), but the support-only selector tried fails. Sensor map: `system/arch_variants/body_regions.py`.

**On hold (Kai, 2026-10-30):** the originality track and MoEMeta-TS are held; do not build them without Kai's go. Body-region is closed and low priority. The 2026-11-20 Track B date no longer applies.

## 5. Open issues
- **IMU time axis:** fs64 still uses the dict's IMU, which carries the env20 anomaly. Rebuilding the IMU from the raw CSVs costs about a day; decide before the finalist HPO.
- **fs64 standardisation** uses the 4 dev users only. That is fine, but state it in the paper.
- **Rebuttal commitments:**
  - EMGBench EMG-only with gate statistics;
  - the user sweep (A15);
  - the modality ablation;
  - E=8 vs 22 (the §1 grid);
  - citations: Rahimian 2021, Antoniou 2019, Proroković 2020, Raghu (ANIL);
  - remove the NOTS mention;
  - relabel the Côté-Allard / Kaifosh rows.
- **Ability story** is weak at the data level; plan the framing for a null.
- **Venue / second dataset:** decide after the `DECISION_method.md` verdict. ICML needs HPO started by about mid-November.
- **Rules:**
  - no silent fallbacks;
  - never commit `.pkl` files or reviewer/paper PDFs (`docs/papers/NeurIPS-2025-moemeta.pdf` is untracked; leave it out);
  - only Kai launches jobs;
  - refresh the scratch launchers after every `git pull`.
