# CURRENT TODOs
*Short list of open threads only; finished work goes to `docs/LOG.md`. Run tracker: `docs/REBUTTAL_CHECKLIST.md`. Top-level plan: `docs/PLAN_resubmission.md`. Docs index: `docs/README.md`. Architecture work: `docs/specs/SPEC_architecture_exploration.md`. Updated 2026-10-30, late evening.*

**Goal:** resubmit EncoderMoE (rejected NeurIPS'26) on one clean protocol: 28 users, 7 folds, cached episodes, 10-way 1-shot primary. Add the reviewer-requested baselines and ablations. **Venue: ICML (Jan) or TMLR, decided after the main runs finish.**

## 0. START HERE on Monday 2026-11-02
**Read `docs/analysis/COVARIANCE_TUTORIAL.md` §0 (executive summary) first.** The 2026-10-30 classical results changed the plan: training-free Riemannian MDM scores 91.4% at 10-way 1-shot over the 28 users (IMU covariance does the work), 20+ points above any learned model we have. Consequences and the Monday plan are in that file:
- §9 the finding ("for user-defined gestures the useful prior is geometric, not semantic");
- §10 which covariance ideas augment EncoderMoE (dead end) vs build on MDM (go);
- §11 finding paper (TMLR / TNSRE) vs method paper (ICML), and why the premise and the clustering/ability story survive;
- **§12 the planned analyses (new, nothing implemented):** §12.1 the pooled-learned-tangent-metric gate decides whether any meta-learning story is left; §12.3 eight cheap CPU tests;
- §13 what the pending fs64 sanity rows (other instance) will be used for.
Venues under consideration: **ICML, TMLR, possibly TNSRE / a domain journal** (Kai, 2026-10-30).
Nothing deep gets built before the §12.1 gate. Data is on the other laptop and the cluster, not this one (hold).

## 0b. What Claude is doing now
- **Riemannian track (2026-10-30 night, PAUSED by Kai; pick up here):** all code is built and tested (`specs/SPEC_riemannian.md`; 468 tests pass). The laptop results are in `~/gest_results/riemann_analysis/` (env20 only so far):
  - **Done:** distance bank `dist_env20.npz`; `gesture_descriptors.csv` (320 cells, raw CSVs); `fa_env20/` (failure analysis); `cov_cluster_gate_env20.json`.
  - **Stopped half-way:** the fs64 distance bank. Re-run, then the fs64 gate and failure analysis:
    `PY=~/anaconda3/envs/fl_torch/python.exe; O=~/gest_results/riemann_analysis; export CODE_DIR=$PWD`
    `$PY system/analysis/trial_distances.py --emg-repr fs64 --out $O/dist_fs64.npz` (~30-40 min)
    `$PY system/analysis/failure_analysis.py --dist $O/dist_fs64.npz --descriptors $O/gesture_descriptors.csv --out $O/fa_fs64`
    `$PY system/analysis/cov_cluster_gate.py --emg-repr fs64 --out $O/cov_cluster_gate_fs64.json`
  - **Still to do:** compute P4 (seg_mdm beats mdm on ≥ 20% of cells) from `fa_*/user_gesture_acc.csv`. Kai: cluster batch B + C2 + C4 below include the 10 new tokens and the A6 Riemannian methods.
- **Nothing running.** Waiting on Kai's cluster batch B (debug HPO re-run on fs64, now including the Q1 family) and C (eval-path smoke tests). Results of the 2026-10-30 batch: §4.
- **Next code, waiting for Kai's go:** nothing queued. Open options: F4, T1/T3, CNP, the GNI side of the raw-EMG comparison, ports of A7/A8/portA and Kaifosh (A10-A12) to the cache, EMGBench EMG-only (promised to fkxS).
- The Q1 family is in the B command below; after B it goes into the screen with the other P-tokens (no frozen MAML HPs: they are metric learners).

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

**A. Classical baselines: done 2026-10-30** (all 7 tokens x 7 folds x {env20, fs64}; the four `$B` fs64 folds that hit 3 h finished at 12 h in ~1.5 h each). Results in §4; copies on the lab desktop in `~/gest_results/r1030/eval/kfold7/`.

**B. Debug trials of the new code: RE-RUN (not done yet).**
- The first batch (jobs 2475010-27, 2026-10-30 14:00-16:45) is read in §4. The debug jobs run one at a time and each loads the code when it starts; the cluster repo pulled the `--emg-repr` fix at about 16:25, so A4 through T4 trained on **env20** (`emg_in_ch: 16`) and only P1, P1_moe, P1_rel, P1_feat, P3, A4_ANIL on fs64. No Tracebacks. M0, X2, X3, X4, X6, T2 and F3 hit the 15-min debug limit inside trial 0 (fine for this check).
- HPO logs never print `PYTHON_EXIT_CODE`; judge them by the pass rule below, not by an exit line.
```bash
bash hpo_ablation_launcher.sh A4 M0 X2 X3 X4 X6 T2 F1 F2 F3 T0 T4 P1 P1_moe P1_rel P1_feat P3 A4_ANIL Q1 Q1_nola Q1_mlp Q1_uni Q1_ce --emg-repr fs64 --debug
# the Riemannian track (specs/SPEC_riemannian.md):
bash hpo_ablation_launcher.sh X7 X7T X8 X8W X9 A4_cov R1 R1_A4 P1_cov P1_spd --emg-repr fs64 --debug
```
When they finish, paste Claude the output of:
```bash
LOGS=/scratch/my13/kai/runs/paper/sub2/hpo/logs
for f in $(ls -t $LOGS/hpo_*.out | head -33); do echo "== $(basename $f)"; grep -m1 "emg_in_ch:" $f; grep -m1 "build_model\]" $f; grep -m2 -E "Epoch completed in|\[metric:.*epoch 1:" $f; grep -m1 -E "Traceback|Error" $f; done
```
- **Pass:** every log shows `emg_in_ch: 112` (fs64), `[build_model] ... parameters` and at least one epoch line, with no Traceback.
- X8 / X8W also print `[init_cov_centroids] ... cluster sizes [...]` before training.

**C. Eval-path smoke tests of the new scripts** (debug partition, fold 0, legacy HPs; checks train → test → save end to end):
```bash
bash kfold_launcher.sh V_X2 V_P1 V_P3 Z_all M0 --folds 0 --debug --hpo-config legacy --emg-repr fs64
# after that M0 smoke finishes (they read its fold-0 checkpoint):
bash kfold_launcher.sh Z3 A16_M0 --folds 0 --debug --hpo-config legacy --emg-repr fs64
```
Pass: every `kfold7_*_f0_*.out` ends with `PYTHON_EXIT_CODE=0`, and every new result JSON records `"emg_repr": "fs64"`.
- **Bug fixed 2026-10-30 late evening:** eval scripts applied `EMG_REPR` per fold only, so they trained on fs64 but `save_results()` recorded `emg_repr: env20` from the base config (seen in the 11:21 M0 fs64 smoke). `make_base_config()` now applies `EMG_REPR`. Any eval JSON written before this fix under a `.../fs64/` folder is mislabelled; the A6 results are not affected (A6 sets the representation itself).

**C2. The architecture screen**, after B passes. It uses ≥ 24 trials and the legacy shared HPs frozen.
```bash
F=/projects/my13/kai/meta-pers-gest/pers-gest-cls/system/NOTS/paper/ablations/screen_fixed
bash hpo_ablation_launcher.sh M0 A4 X2 X3 X4 X5 X6 F1 F2 F3 --emg-repr fs64 --n-trials 24 --study-suffix screen --fixed-params $F/legacy_shared_maml.json --dry-run
bash hpo_ablation_launcher.sh M0 A4 X2 X3 X4 X5 X6 F1 F2 F3 --emg-repr fs64 --n-trials 24 --study-suffix screen --fixed-params $F/legacy_shared_maml.json
bash hpo_ablation_launcher.sh T0 T2 T4 --emg-repr fs64 --n-trials 24 --study-suffix screen --fixed-params $F/legacy_inner_only.json
bash hpo_ablation_launcher.sh X7 X7T X8 X8W X9 A4_cov R1 R1_A4 --emg-repr fs64 --n-trials 24 --study-suffix screen --fixed-params $F/legacy_shared_maml.json
bash hpo_ablation_launcher.sh P1 P1_moe P1_match P1_rel P1_ridge P1_feat P3 Q1 Q1_nola Q1_mlp Q1_uni Q1_ce P1_cov P1_spd --emg-repr fs64 --n-trials 24 --study-suffix screen
```

**C4. Riemannian classical extensions (CPU, any time):** all 9 cells x 7 folds, env20 and fs64.
```bash
bash kfold_launcher.sh A6_riemann_seg_mdm A6_riemann_rc_pool A6_riemann_cluster_pool --hpo-config legacy --emg-repr fs64
bash kfold_launcher.sh A6_riemann_seg_mdm A6_riemann_rc_pool A6_riemann_cluster_pool --hpo-config legacy --emg-repr env20
```
- `M0` and every `V_*` eval now also save per-query test logits (`*_query_logits.npz`, next to the results JSON). Copy them back with the results; they feed the hybrid (H1) and the failure analysis.

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
- the `--smoke` M0 on fs64 is covered by C (the 11:21 run trained but mislabelled its JSON, see C);
- delete the non-smoke A1/A2 fold-0 JSONs under `.../eval/kfold7/{A1,A2}/fold0/`.

**F. Send results back to Claude (any batch).** `touch ~/batch_mark` right before launching; afterwards:
```bash
cd /scratch/my13/kai/runs/paper/sub2
{ echo "### sacct"; sacct -u km82 -S $(date -r ~/batch_mark +%Y-%m-%dT%H:%M) -X --format=JobID,JobName%40,State,Elapsed
  echo "### HPO report"; python $CODE/system/NOTS/paper/ablations/hpo_tools.py report --db-dir hpo/optuna_dbs; } > status_batch.txt 2>&1
find eval hpo status_batch.txt -newer ~/batch_mark -type f ! -name '*.pt' ! -name '*.pkl' > files_batch.txt
tar czf ~/results_batch.tgz -T files_batch.txt && ls -lh ~/results_batch.tgz
# laptop / lab desktop (Git Bash):
mkdir -p ~/gest_results/<name> && scp "km82@<NOTS login host>:~/results_batch.tgz" ~/gest_results/<name>/
```
(`$CODE` = `/projects/my13/kai/meta-pers-gest/pers-gest-cls`; run with the mamba env active or `hpo_tools` fails on `numpy`.)

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
| MoEMeta-TS (faithful port; margin loss, local adaptation, MoE relation meta) | `arch_variants/moemeta_ts.py` | `Q1 Q1_nola Q1_mlp Q1_uni Q1_ce`; eval `V_Q1` etc. |
| Gate-0 harness: any support-only channel-weighting idea vs uniform 1-NN and a permutation null | `nonparametric/gate0.py`, `docs/specs/SPEC_gate0.md` | laptop: `python system/nonparametric/gate0.py --candidate ...` |
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

**Classical baselines, 28 users, 10-way 1-shot, final (cluster A6, all 7 folds):** mean ± SD over users.

| method | env20 | fs64 |
|---|---|---|
| Riemannian MDM | 90.4 ± 10.3 | **91.4 ± 8.6** [88.0, 94.2] |
| Riemannian tangent space + LR | 86.8 ± 12.5 | 88.8 ± 9.4 |
| `$B` (PCA + L1 1-NN) | 78.1 ± 11.2 | 80.4 ± 8.9 |
| MiniRocket + ridge | 78.8 ± 9.4 | 79.5 ± 9.6 |
| `$B`-user | 58.6 ± 10.8 | 74.0 ± 9.4 |
| kNN-L1 | 58.0 ± 11.2 | 72.9 ± 9.5 |
| ProtoNet (raw features) | 51.1 ± 12.4 | 65.6 ± 11.4 |

- MDM fs64 beats every other row (paired Wilcoxon, Holm p < 0.001) except MDM env20 (+1.1, p = 0.14).
- **The bar for learned models is MDM, ~91%, not `$B`.** See `analysis/COVARIANCE_TUTORIAL.md`.

**HPO debug batch 1 (2026-10-30, one trial each, dev users; plumbing only, not results):**
- fs64: P1 ProtoNet 76.8 after 1 epoch, P1_moe 67.3, P3 63.8, P1_feat 38.9, P1_rel 32.2, A4_ANIL 22.1 (low; check after B).
- env20 (pre-fix): A4 34.7, F1 36.5, F2 36.0, T0 12.0, T4 11.2.
- Epoch time (env20): ~130-145 s for the E=22 MoE variants (M0, X2, X4, X6, T2), 55 s X3, 35 s A4, 40-53 s F1-F3. A4_ANIL on fs64: 22 s. Use the B re-run for the fs64 trial-budget estimate.

**Uniqueness (notebook 03):**
- Cross-user zero-shot gets 13-16%, against 62-78% within-user.
- Executions recur across users under different labels.
- The ability-group effect is weak (Mantel r 0.05-0.09).

**Body-region Gate 0 (dev users, 1-NN, `specs/SPEC_body_regions.md` §4):** region experts dropped by the pre-stated rule. Anatomical groups do no better than random groups (81.3 vs 80.4, 94th percentile). Per-user selection helps (+5.3 over one shared subset), but the support-only selector tried fails. Sensor map: `system/arch_variants/body_regions.py`.

**Kai, 2026-10-30 evening:** MoEMeta-TS is a go (built, see §3). The originality track is exploratory: test concrete, cheap ideas through `SPEC_gate0.md` when they come up; otherwise spend the time on the main runs. Body-region is closed. Venue (ICML vs TMLR) follows the results. The `.md` specs are suggestions, not rules: prefer a better solution when there is one.

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
