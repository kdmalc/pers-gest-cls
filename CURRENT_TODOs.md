# CURRENT TODOs (short; top-level plan: `docs/PLAN_resubmission.md`; history: `LOG.md`, `docs/FULL_RUNNING_LOG.md`)
*Updated 2026-10-30.*

## 1. Main goal
Resubmit EncoderMoE (rejected NeurIPS'26). Rerun everything on one clean protocol: 28 users, 7 folds, shared cached episodes, 10-way 1-shot primary. Add the baselines and ablations the reviewers asked for. Fix the EMG representation. Pick the venue after the reruns (`PLAN_resubmission.md` §6).

## 2. Order of work
1. **EMG representation:** done on the dev users (the rule says **fs64**, §4). **Classical baselines:** full grid on the cluster (§3A); quick laptop numbers are in §4.
2. **Kai + PI:** sign off on fs64 for the main table.
3. **Cluster:** check the debug results (§5A). Then reply "hpo ok" / "start hpo". HPO runs at 10-way 1-shot on the dev users with `emg_repr=fs64`.
4. **Code, in parallel with HPO:** transformers, the MoE X-matrix (uniform-gate control), zero-shot, the upper bound, the seed flag, A13 at 10-way (§6). The transformers come first because they need their own HPO.
5. **Cluster:** 7-fold runs (tuned configs) → aggregate → `DECISION_method.md` verdict → venue.

**Are the runs final-paper results?** Yes, if all of these hold:
- the run is on the kfold7 split;
- it uses the final `emg_repr` and `--hpo-config tuned` (or `frozen` for the frozen ablation rows);
- its code is committed (the JSON records `git_rev`);
- it is not a smoke or debug run.

The aggregator rejects smoke runs and mismatched episodes. The classical-baseline runs from §3 are final-paper results as they are: they are deterministic, read the same episodes, and are reported for both representations. Rerun anything whose code changes later.

## 3. Classical baselines: cluster for the long runs, laptop for quick checks (updated 2026-10-30)
**Rule of thumb:** anything over ~5 h goes to the cluster. The full 3x3 grid for all 7 baselines x 2 EMG versions is roughly 30+ CPU-hours on the laptop, but on the cluster it is 98 parallel CPU jobs (7 tokens x 7 folds x 2), done in a few hours.

**A. Cluster (Kai): every classical baseline, all 9 cells, env20 and fs64.**
```bash
cd /projects/my13/kai/meta-pers-gest/pers-gest-cls && git pull
git status --short          # should be clean; if the tracked .pyc shows up: git restore system/__pycache__/__init__.cpython-310.pyc
L=/projects/my13/kai/meta-pers-gest/pers-gest-cls/system/NOTS/paper/ablations/launchers
cp $L/kfold_launcher.sh $L/hpo_ablation_launcher.sh /scratch/my13/kai/runs/paper/sub2/

# one-time: MiniRocket needs aeon, Riemannian needs pyriemann (your dry run was clean)
module purge && module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh && source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba install -p /projects/my13/kai/meta-pers-gest/envs/fl-torch -c conda-forge pyriemann=0.9 aeon pytest

cd /scratch/my13/kai/runs/paper/sub2
B="A6_dollar_b A6_dollar_b_user A6_knn_l1 A6_proto_euclid A6_minirocket A6_riemann_mdm A6_riemann_ts"
bash kfold_launcher.sh $B --hpo-config legacy --emg-repr env20 --dry-run | head -20   # check paths: .../kfold7/<token>/env20/fold<k>
bash kfold_launcher.sh $B --hpo-config legacy --emg-repr env20
bash kfold_launcher.sh $B --hpo-config legacy --emg-repr fs64
```
- CPU-only jobs on `commons`, 10 CPUs and 16 GB each. Time limits: 3 h for `$B`/kNN/ProtoNet, 6 h for MiniRocket, 23 h for Riemannian.
- `--hpo-config legacy` is only there because the launcher requires the flag. Classical methods take no tuned hyperparameters.
- Output: `/scratch/my13/kai/runs/paper/sub2/eval/kfold7/A6_<method>/<env20|fs64>/fold<k>/`. Since 2026-10-30, each results JSON also records `emg_repr`.

When the jobs finish, check for failures and copy the results to the laptop:
```bash
# run only after squeue --me is empty (a running job has no exit line yet, so it would look "failed")
for f in /scratch/my13/kai/runs/paper/sub2/eval/logs/kfold7_A6_*.out; do rc=$(grep -o "PYTHON_EXIT_CODE=[0-9]*" "$f"); [[ "$rc" == "PYTHON_EXIT_CODE=0" ]] || echo "${rc:-NO_EXIT_LINE(timeout/killed)}  $f"; done   # empty = all OK
sacct -u km82 -S 2026-10-30 -X --format=JobID,JobName%34,State,Elapsed | grep -E "TIMEOUT|FAILED|OUT_OF_ME"   # empty = none timed out
```
```bash
# on the laptop (Git Bash); small JSONs only
mkdir -p ~/gest_results/A6_cluster && scp -r "km82@<NOTS login host>:/scratch/my13/kai/runs/paper/sub2/eval/kfold7/A6_*" ~/gest_results/A6_cluster/
```
Then run notebook 02 with `A6_RESULTS=~/gest_results/A6_cluster` (or edit `RESULTS` in its first code cell).

**B. Laptop (done 2026-10-30 by Claude, git `58eea72`):** kNN-L1 and ProtoNet-Euclid at 10-way 1-shot, both representations, in `~/gest_results/A6_primary/`. Notebook 02 runs end to end on them. Results are in §4.
- **Kai's laptop loop started 11:10 the same day** (`$B` first) writes into the same folder.
  - Its fold-0 `$B` file predates the `emg_repr` fix. Its later files record `git_rev 80fc244` but ran the edited working tree, so their recorded revision is wrong.
  - If that loop continues, its kNN/ProtoNet runs will duplicate mine, and the aggregator rejects duplicates.
  - **Stop it (Ctrl+C) and delete `~/gest_results/A6_primary/env20/A6_dollar_b_*`.** The cluster run (A) covers `$B` and the rest.

**C. Time-domain / LDA baselines (laptop build, then cluster).** These need the td64 dict, built from the raw CSVs in Box. Needs your OK; it was a PI question.
```bash
cd ~/Repos/pers-gest-cls; PY=~/anaconda3/envs/fl_torch/python.exe
$PY system/universal_preprocessing/EMG_preprocessing/build_td64_tensor_dict.py   --root "/c/Users/kdmen/Box/Yamagami Lab/Data/2024_UIST_dataset/upload/segmented_raw_data"   --fused-dict dataset/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl   --out dataset/meta-learning-sup-que-ds/segfilt_td64_tensor_dict.pkl
scp dataset/meta-learning-sup-que-ds/segfilt_td64_tensor_dict.pkl km82@<NOTS login host>:/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/
```
Then, on the cluster: `bash kfold_launcher.sh A6_td_knn A6_td_lda A6_td_lda_pool --hpo-config legacy`. Their output lands under `.../td64/` automatically.

**D. Notebooks** (`system/stats/`; open them in Jupyter with the `fl_torch` kernel):
- `01_emg_repr_decision_dev.ipynb`: **run** (outputs saved). Dev users only; the decision rule is in the notebook.
- `02_classical_baselines_28users.ipynb`: reads any folder of A6 results and groups them by the recorded `emg_repr`. It gives the summary table with CIs, the full grid, and the paired fs64-vs-env20 comparison.
- `03_gesture_uniqueness.ipynb`: **run** (all 32 users, no training). Results are in §4.

## 4. Results so far (first looks, laptop, 2026-10-29)
- **EMG representation (dev users, notebook 01).** The pre-registered rule says **adopt fs64**. At 10-way 1-shot, EMG-only fs64 beats env20 by +28.9 pts (proto) and +17.2 pts (1-NN L1), on 4/4 users. Balanced fused fs64+IMU beats env20+IMU by +10.8 / +12.5. The env20 time-axis anomaly points the same way.
- **Classical baselines, 28 users, 10-way 1-shot (fused EMG+IMU, notebook 02, 2026-10-30).** Reportable numbers, not used for the representation decision.

  | method | env20 | fs64 |
  |---|---|---|
  | kNN-L1 | 58.0 [54.0, 62.1] | **72.9** [69.4, 76.3] |
  | ProtoNet-Euclid (raw) | 51.1 [46.8, 55.7] | 65.6 [61.5, 69.7] |

  fs64 − env20 is +14.9 (kNN) and +14.5 (ProtoNet) pts, 28/28 users better, Holm p < 1e-7.
  - **Warning for `DECISION_method.md` R4:** old EncoderMoE was 67.7% at 10-way 1-shot (old protocol and split, so not comparable). A no-training kNN on fs64 is already above that number. The rerun M0 must beat these baselines on the same episodes.
- **Uniqueness (notebook 03, 32 users, model-free).**
  - Cross-user zero-shot (nearest label centroid of the other 31 users) gets 13-16% (chance 10%), against 62-78% for within-user 1-shot. That is under the 20% "keep the user-defined framing" threshold in `SPEC_zeroshot.md`. It is not the pre-registered Z test, which needs trained models.
  - A user's gesture prototype has an other-user nearest neighbour with the same label 13-19% of the time (null 10%, p ≤ .035). That is weak but real shared label semantics.
  - Cross-user "motif" matches are above the channel-permuted null for fs64 (13.2 vs 6.8 per prototype) and fused (11.2 vs 6.7), but only about 10% of them share the label. **So executions recur across users under different labels**, which is the sample-level story in the study plan.
  - Ability group barely predicts sharing: Mantel r = 0.05-0.09, p = 0.05-0.17. The ability-cluster story is weak at the data level.

## 5. Cluster
**A. Debug runs (launched 2026-10-29): DONE, plumbing OK, numbers meaningless (legacy HPs).** Details and the issue list: `docs/10_29_26_debug_run_results.md`. Fixed the same day (pushed): `pretrain_trainer` ignored the config's `earlystopping_*`/`gradient_clip_max_norm` keys (ran patience 12 / clip 1.0, now 8 / 10.0); legacy Kapanji / L2SO / hpo_test_split code removed from M0, `ablation_config`, the A11 sweep. Still to do:
- [x] The cleanup commits pass the full local test suite (233 tests, laptop, 2026-10-30).
- [ ] On the cluster: after the §3A `git pull`, re-run one `--smoke` M0 (`bash kfold_launcher.sh M0 --folds 0 --debug --hpo-config legacy --emg-repr fs64`). This also checks M0 on fs64.
- [ ] Delete or move the non-smoke A1/A2 fold-0 JSONs (legacy HPs + old patience/clip) from `.../eval/kfold7/{A1,A2}/fold0/`.
- [ ] HPO search space: cap `label_smooth` (debug trial sampled 0.42), sanity-range `ft_lr`/`ft_steps` (the 46.8% val came from FT, pretrain-only val was 21.8%, train acc 97%, best epoch 4). Update `SPEC_hpo.md`.
- [ ] MoE logging: entropy is saturated at ln(top_k) (use pre-top-k softmax), imbalance ratios are div-by-0 artifacts, "Routing by Participant" prints `?` (PIDs not passed in the supervised path).
- [ ] Cosmetic: mojibake `Â±` in FINAL lines; HPO launcher header prints literal `$(hostname)`/`$(date)`; prune per-trial 35 MB checkpoints.
- [ ] Remaining legacy L2SO/hpo_test_split code: `launchers/eval_launcher.sh` (now incompatible with M0, delete?), A10-A12, A1, A13, A16, fewshot_grid_A4, portB, `tests/tiny_config.py`.

**B. Done:** the dict hash matches the laptop (`eb60c8af…`), and the fs64 dict is copied. The `mamba install` of pyriemann/aeon is now part of §3A.

**C. GPU-hour accounting.** From now on, count paper compute from 2026-10-29:
```bash
sacct -u km82 -S 2026-10-29 -X -n -P --format=JobID,ElapsedRaw,AllocTRES | \
  awk -F'|' '{ if (match($3,/gres\/gpu=[0-9]+/)) { n=substr($3,RSTART+9,RLENGTH-9); s+=$2*n } } END {printf "%.1f GPU-hours\n", s/3600}'
```

## 6. Code still to write (Opus/Sonnet; can start now, in parallel with HPO)
- [ ] Transformers T0-T3 (`SPEC_moe_compute.md` §3) + their HPO space.
- [ ] MoE X-matrix: X2 uniform gate (key control), X3, X4 task-conditioned routing.
- [ ] Zero-shot Z1-Z4 script (`SPEC_zeroshot.md`).
- [ ] Upper bound (ARaC Q2): fine-tune on all non-test reps of the test user.
- [ ] `--seed` flag through the eval scripts and launcher (3 seeds for headline methods).
- [ ] A13 modality ablation at 10-way (currently 3-way default); add an fs64 run.
- [x] HPO launcher now requires `--emg-repr env20|fs64` (exports EMG_REPR; study name, journal and output dir include it). Not yet exercised on the cluster: `--debug --dry-run` first.
- [ ] Episodically trained ProtoNet; ANIL (= portB) checked on the cache.
- [ ] Port A7/A8/portA/A16 and Kaifosh (A10-A12) to the cache; FS-HGR / EMG-CrossFormer (Tier 3).

## 7. Open issues (tracked, not urgent tonight)
- **PI sign-offs:** fs64 for the main table (replaces the "1 kHz" policy draft in `SPEC_2khz_emg.md` §8), building td64, reporting `$B` if it wins.
- **The env20 anomaly.** The old 20 Hz EMG (and the dict's IMU) time axis does not reproduce from the raw CSVs. fs64 is built from the raw EMG but **still uses the dict's IMU**. Consider rebuilding the IMU from the raw CSVs before HPO (it costs a day; otherwise the anomaly stays in the IMU half of every model).
- **fs64 standardisation** constants come from the 4 dev users only. That is fine for test-user validity, but state it in the paper.
- **HPO code never ran on the cluster.** The debug trial in §5A is the first check of pruning, replay and export.
- **Rebuttal commitments to keep:** EMGBench EMG-only with gate statistics (fkxS); a user sweep to 20 users (A15); modality ablation under the paired design; E=8 vs E=22; citations (Rahimian 2021 FS-HGR, Antoniou ICLR 2019, Proroković 2020, Raghu/ANIL); remove the NOTS/Rice mention (anonymity); relabel the Côté-Allard / Kaifosh rows.
- **The ability story is weak at the data level** (§4 Mantel). The per-group gain analysis (`SPEC_group_analysis.md`) after the reruns is the real test; plan the framing for a null.
- **Venue / second dataset:** decide after the `DECISION_method.md` verdict. ICML needs HPO started by about mid-November.
- **Alternative-method track** (`PLAN_resubmission.md` §5): explore on dev users only.
- **Rules:** no silent fallbacks; never commit `.pkl` or reviewer PDFs; only Kai launches cluster jobs; submit from `/scratch/my13/kai/runs/paper/sub2/`; refresh the scratch launchers after every `git pull`.
