# CURRENT TODOs (short; top-level plan: `docs/PLAN_resubmission.md`; history: `LOG.md`, `docs/FULL_RUNNING_LOG.md`)
*Updated 2026-10-29 (evening).*

## 1. Main goal
Resubmit EncoderMoE (rejected NeurIPS'26). Rerun everything on one clean protocol: 28 users, 7 folds, shared cached episodes, 10-way 1-shot primary. Add the baselines and ablations the reviewers asked for. Fix the EMG representation. Pick the venue after the reruns (`PLAN_resubmission.md` §6).

## 2. Order of work
1. **Laptop (§3):** confirm the EMG representation on the dev users (done tonight: the rule says **fs64**, see §4), run the classical baselines on the 28 users, and run the uniqueness notebook.
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

## 3. Laptop runs (Kai, 2026-10-30)
Use Git Bash. All of these run on the CPU.

**A. Classical baselines, primary cell (10-way 1-shot), both representations.** About 5 min per method per representation for the fast ones. Riemannian takes about 20 min per fold, so run it last or overnight.
```bash
cd ~/Repos/pers-gest-cls && git pull
cd system/NOTS/paper/ablations/test_eval_files
export CODE_DIR=~/Repos/pers-gest-cls HPO_CONFIG=legacy GIT_REV=$(git rev-parse --short HEAD)
PY=~/anaconda3/envs/fl_torch/python.exe
for r in env20 fs64; do
  export RUN_DIR=~/gest_results/A6_primary/$r
  for m in dollar_b dollar_b_user knn_l1 proto_euclid minirocket riemann_mdm riemann_ts; do
    for k in 0 1 2 3 4 5 6; do $PY A6_classical_baselines.py --method $m --emg-repr $r --fold-idx $k; done
  done
done
```
- `HPO_CONFIG=legacy` is only there because the runner requires it to be set. Classical methods have no hyperparameters from HPO.
- Keep a separate `RUN_DIR` per representation, because the file names don't record it.
- Never run the same method/cell twice into one folder. The aggregator rejects duplicate episodes.

**B. Full 3x3 grid (later or overnight; several hours for `$B` and Riemannian).** Same loop, with `RUN_DIR=~/gest_results/A6_grid/$r` and `--cells all` added.

**C. Time-domain / LDA baselines.** These first need the td64 dict, which is built from the raw CSVs in Box (needs your OK, it was a PI question):
```bash
cd ~/Repos/pers-gest-cls
$PY system/universal_preprocessing/EMG_preprocessing/build_td64_tensor_dict.py \
  --root "/c/Users/kdmen/Box/Yamagami Lab/Data/2024_UIST_dataset/upload/segmented_raw_data" \
  --fused-dict dataset/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl \
  --out dataset/meta-learning-sup-que-ds/segfilt_td64_tensor_dict.pkl
cd system/NOTS/paper/ablations/test_eval_files; export RUN_DIR=~/gest_results/A6_primary/td64
for m in td_knn td_lda td_lda_pool; do for k in 0 1 2 3 4 5 6; do
  $PY A6_classical_baselines.py --method $m --emg-repr td64 --fold-idx $k; done; done
```
Then `scp` the td64 `.pkl` to the cluster, next to the fs64 file.

**D. Notebooks** (`system/stats/`; open them in Jupyter with the `fl_torch` kernel):
- `01_emg_repr_decision_dev.ipynb`: **already run** (outputs saved). Uses dev users only, with the decision rule written in the notebook.
- `02_classical_baselines_28users.ipynb`: run it after A finishes. It reads `~/gest_results/A6_primary/` (set `A6_RESULTS` to use `A6_grid`). It gives the summary table with CIs, the full grid, and the paired fs64-vs-env20 comparison.
- `03_gesture_uniqueness.ipynb`: **already run** (all 32 users, no training). Results are in §4.

## 4. Results so far (first looks, laptop, 2026-10-29)
- **EMG representation (dev users, notebook 01).** The pre-registered rule says **adopt fs64**. At 10-way 1-shot, EMG-only fs64 beats env20 by +28.9 pts (proto) and +17.2 pts (1-NN L1), on 4/4 users. Balanced fused fs64+IMU beats env20+IMU by +10.8 / +12.5. The env20 time-axis anomaly points the same way.
- **kNN-L1 (fused), 28 users, 10-way 1-shot**, from testing notebook 02: env20 ≈ 59%, fs64 ≈ 73% (fold means 53-63% vs 71-77%). This is a reportable baseline, but it was not used for the decision.
- **Uniqueness (notebook 03, 32 users, model-free).**
  - Cross-user zero-shot (nearest label centroid of the other 31 users) gets 13-16% (chance 10%), against 62-78% for within-user 1-shot. That is under the 20% "keep the user-defined framing" threshold in `SPEC_zeroshot.md`. It is not the pre-registered Z test, which needs trained models.
  - A user's gesture prototype has an other-user nearest neighbour with the same label 13-19% of the time (null 10%, p ≤ .035). That is weak but real shared label semantics.
  - Cross-user "motif" matches are above the channel-permuted null for fs64 (13.2 vs 6.8 per prototype) and fused (11.2 vs 6.7), but only about 10% of them share the label. **So executions recur across users under different labels**, which is the sample-level story in the study plan.
  - Ability group barely predicts sharing: Mantel r = 0.05-0.09, p = 0.05-0.17. The ability-cluster story is weak at the data level.

## 5. Cluster
**A. Debug runs (launched 2026-10-29): DONE, plumbing OK, numbers meaningless (legacy HPs).** Details and the issue list: `docs/10_29_26_debug_run_results.md`. Fixed the same day (pushed): `pretrain_trainer` ignored the config's `earlystopping_*`/`gradient_clip_max_norm` keys (ran patience 12 / clip 1.0, now 8 / 10.0); legacy Kapanji / L2SO / hpo_test_split code removed from M0, `ablation_config`, the A11 sweep. Still to do:
- [ ] On the cluster: `git pull`, `git restore system/__pycache__/__init__.cpython-310.pyc`, refresh the scratch launcher copies, re-run `--smoke` M0 and `pytest` (the cleanup was never executed: no Python on the Claude side).
- [ ] Delete or move the non-smoke A1/A2 fold-0 JSONs (legacy HPs + old patience/clip) from `.../eval/kfold7/{A1,A2}/fold0/`.
- [ ] HPO search space: cap `label_smooth` (debug trial sampled 0.42), sanity-range `ft_lr`/`ft_steps` (the 46.8% val came from FT, pretrain-only val was 21.8%, train acc 97%, best epoch 4). Update `SPEC_hpo.md`.
- [ ] MoE logging: entropy is saturated at ln(top_k) (use pre-top-k softmax), imbalance ratios are div-by-0 artifacts, "Routing by Participant" prints `?` (PIDs not passed in the supervised path).
- [ ] Cosmetic: mojibake `Â±` in FINAL lines; HPO launcher header prints literal `$(hostname)`/`$(date)`; prune per-trial 35 MB checkpoints.
- [ ] Remaining legacy L2SO/hpo_test_split code: `launchers/eval_launcher.sh` (now incompatible with M0, delete?), A10-A12, A1, A13, A16, fewshot_grid_A4, portB, `tests/tiny_config.py`.

**B. Done:** the dict hash matches the laptop (`eb60c8af…`), and the fs64 dict is copied. **Still to do:** the `mamba install` of pyriemann/aeon. It's only needed if classical baselines are ever run on the cluster, because the laptop already covers them.

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
- [ ] **Blocker for "start hpo":** `hpo_ablation_launcher.sh` has no `--emg-repr` option (checked 2026-10-29), so HPO would run on env20. Add the flag, plus a test that the HPO config picks up fs64.
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
