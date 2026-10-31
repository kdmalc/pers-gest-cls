# LOG (context and design decisions that are not immediately critical)
Full history: `docs/archive/FULL_RUNNING_LOG.md`. Status table: `docs/archive/HIGH_LEVEL_STATUS.md`. Immediate blockers (long form): `docs/archive/KAI_TODO.md`.

## 2026-10-30 night, Riemannian track: first laptop results (env20; rules pre-stated in `specs/SPEC_riemannian.md`, commit fc3130a)
- MDM sanity on fs64 (50 episodes/user): fused 91.5, shuffled labels 9.6, EMG-only 76.6, IMU-only 91.0.
- **Distance bank:** reproduces `classical.py` exactly at 1-shot. The whole 28-user analysis takes ~5 min instead of hours. On env20, IMU-only MDM (91.0) beats fused MDM (90.4).
- **Cluster gate (env20):**
  - G1 identifiability: 1.00, pass.
  - G2 silhouette: 0.24 vs null 95th percentile 0.11 (K=3), pass.
  - **G3 split-half ARI: 0.09, fail.**
  - G4 same-referent rate: 0.14 vs the 0.2 bar, fail.
  - Verdict: users have stable, unique covariance signatures, but the clusters are not stable across halves. So use continuous routing (X7 / X7T, "routing by user signature") rather than discrete clusters (X8, `cluster_pool`). This waits on fs64.
- **Failure analysis (env20, 504k queries):**
  - **P1 not supported.** Oracle-or with MDM reaches 93.6 (`$B`) and 94.9 (MDM-IMU), but no equal-weight z-fusion beats MDM (best: mdm+td_knn 90.0).
  - **P2 not supported as stated.** Every method drops on low-movement gestures (MDM 94.5 high vs 84.7 low). The EMG-IMU gap is the same in both tertiles (-15.5 / -16.6). On finger-like gestures EMG holds up better (-13.0 vs -17.9 elsewhere).
  - **P3 supported for `seg_mdm` only** (leave-one-user-out rho 0.19, p 0.01): gesture properties predict where segment covariances help.
  - `td_knn` (mean/std features) reaches 84.8, so the per-channel means carry signal that the centred covariance drops.
  - P4 not computed yet.

## 2026-10-30, cluster batch read (late evening)
- Results came back as one tarball (`find ... -newermt` minus `*.pt`/`*.pkl`, 3.3 MB), unpacked on the lab desktop in `~/gest_results/r1030/`; recipe in `CURRENT_TODOs.md` §2F.
- A6 complete; `$B` fs64 = 80.4 ± 8.9 after the 12 h relaunch. Riemannian MDM fs64 (91.4) is the top classical baseline.
- HPO debug batch 1 was mixed env20/fs64 because the code was pulled mid-batch and each job imports at start time. Lesson: pull only between batches, or check `emg_in_ch` per log.
- Eval mislabel bug fixed (`make_base_config()` now applies `EMG_REPR`); test `test_base_config_records_launcher_emg_repr`.

## 2026-10-30, body-region line
- Sensor map from Kai's `docs/emg-sensor-placements.png`, in `system/arch_variants/body_regions.py`. IMU order comes from recording slots 1-8, 9, 11, 13, 15; pods 9-12 have two EMG leads. Side assignment verified on the signals; position within a side was not.
- Gate 0 rules were committed in `398078a` before the first run. Result: anatomical groups barely beat random groups, so deep region experts were dropped. `docs/specs/SPEC_body_regions.md` §4.

## 2026-10-25
- Created `CURRENT_TODOs.md` (short handoff) and this file. Existing `docs/archive/KAI_TODO.md` remains the detailed version; keep the two consistent.
- Why the TD baselines: peer feedback says journal reviewers will ask for LDA / time-domain / PCA+KNN baselines. `$B` is PCA + L1 1-NN, so label it that way in the paper. `td_lda_pool` is the only variant using the fold's train users (pooled within-class covariance, Ledoit-Wolf).
- Building the td64 dict (laptop, raw CSVs from Box): `python system/universal_preprocessing/EMG_preprocessing/build_td64_tensor_dict.py --root <Box segmented_raw_data dir> --fused-dict dataset/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl --out dataset/meta-learning-sup-que-ds/segfilt_td64_tensor_dict.pkl`, then scp like the fs64 file. `.pkl` files stay git-ignored.
- Project rules that still apply: no silent fallbacks (raise); never commit reviewer PDFs or `.pkl`; only Kai launches cluster jobs; submit from `/scratch/my13/kai/runs/paper/sub2/`.

## 2026-10-30 (moved out of CURRENT_TODOs.md when it was shortened)
- **Laptop A6 run (Claude, git `58eea72`):** kNN-L1 and ProtoNet-Euclid at 10-way 1-shot, env20 and fs64, in `~/gest_results/A6_primary/`; notebook 02 verified end to end on them. Kai's laptop loop (started 11:10, `$B` first) wrote into the same folder with a wrong recorded `git_rev`; Claude stopped it and moved its `$B` files to `~/gest_results/_discarded/`. The cluster run covers `$B` and everything else.
- **Cluster A6 launch (Kai, 2026-10-30):** 7 tokens x 7 folds x {env20, fs64} on commons (CPU). kNN / ProtoNet / `$B`-user finished first; MiniRocket, Riemannian, `$B` ran longer.
- **Debug runs of 2026-10-29: done**, plumbing OK, numbers meaningless (legacy HPs). Details: `docs/run_reports/10_29_26_debug_run_results.md`. Fixed the same day: `pretrain_trainer` honours `earlystopping_*` / `gradient_clip_max_norm`; legacy Kapanji / L2SO code removed; `label_smooth` capped at 0.2; HPO launcher requires `--emg-repr` and prints real hostname/date.
- **Superseded plan:** the "weekend launch of A1/A2/A4/M0 HPO at 60 trials" (old §3b) was put on hold after Kai's point that M0 and the param-matched controls change if the MoE scheme changes. Replaced by the staged plan in `docs/specs/SPEC_architecture_exploration.md` §3 (A4/A2 first, architecture screen with frozen shared HPs, full HPO only for finalists).
- **Architecture-exploration code (Claude, Opus, 2026-10-30):** X2-X6 and T2 in `MOE_encoder.DeepCNNLSTM_EncoderMOE` (defaults reproduce M0 bit for bit, tested); `system/arch_variants/` (functional attention, FlexNet for F1-F3/T0/T4, metric learners P1 family, ADANN P3); `MOE/routing_metrics.py` (replaces the saturated entropy / div-by-0 diagnostics, debug issues 5-7); per-trial timing in the journal plus `hpo_tools.py`; frozen-HP screening (`--fixed-params`); E x k grid enqueueing. Bug found and fixed on the way: the HPO collapse penalty used an absolute 0.80 selection-frequency threshold, which scored every balanced trial with top_k/E >~ 0.7 as 0 (and would have zeroed every uniform-gate trial). Tests 233 -> 356.

## 2026-10-30 (evening): paper-run code
- **Kai's plan:** settle the MoE and fusion architecture on the dev users first, and read the X1-vs-X2 routing diagnostics. A4 and A2 tuning is held until EncoderMoE is final. So the screen freezes the legacy Trial-89 training HPs (`screen_fixed/`) and screens M0 and A4 as references, with ≥ 24 trials per variant. 100 vs 60 trials for the finalists is decided from the debug timings.
- **Built (local tests only):**
  - EMG-RAW stem and the raw3s builder (IMU deliberately unchanged from the fused dict, so only the EMG input differs from fs64);
  - `arch_variants/registry.py` (one token registry for HPO, eval and launcher);
  - `arch_variant_eval.py` (7-fold eval of any token, `V_<token>`);
  - zero-shot Z1-Z4 (`Z_all`, `Z3`);
  - A16 upper bound on kfold7 (`A16_A2`, `A16_M0`);
  - `--seed` through M0 / A1 / A2 / A3-A4 / A13 / V_* / Z, with `kfold_launcher.sh --seeds`;
  - A13 at 10-way with all 9 cells.
  - Tests 356 → 386.
- **Bugs found and fixed:**
  1. The HPO driver never applied `--emg-repr`: "fs64" studies, and the first debug batch, trained on env20.
  2. A2 was tuned at width 64 but evaluated param-matched. A2's HPO now uses the A4 widths.
  3. A tuned override from another EMG representation now raises (the provenance records `emg_repr`).

## 2026-10-30 (late evening): MoEMeta-TS and the Gate-0 harness (Fable)
- **Kai's decisions:** MoEMeta-TS and a reusable Gate 0 are a go; the originality track is exploratory (cheap ideas via Gate 0, no deadline); `DECISION_method.md` pre-registration is not wanted (if MoEMeta-TS beats EncoderMoE and that cannot be fixed, EncoderMoE is not publishable); the `.md` specs are suggestions, not rules; venue follows the results.
- **MoEMeta-TS built** (`system/arch_variants/moemeta_ts.py`, tokens `Q1 Q1_nola Q1_mlp Q1_uni Q1_ce`, eval `V_Q1*`). Read from arXiv 2510.23013v4 (the repo still has only a README). Triplet (h, r, t) -> (user centroid from the support, class, window). MoE relation meta per class (Eq 5-8, top-N without renormalisation, 1/N), zero-initialised task vectors p_h/p_r/p_t and the R_c adapted on the support margin loss (Eq 9-15), translational L2 score (Eq 12), margin loss over the wrong classes (Eq 13/16; the ranked slot is corrupted, as in the paper). Outer loop trains encoder + experts + gate through the inner step (second order by default). Controls follow the paper's ablation: w/o L.A, w/o MoE (one MLP), plus a uniform gate and a CE-loss sensitivity. Test: with one identity expert at 1-shot the score is exactly 1-NN (`ANALYSIS_moemeta.md` §3). HPO space tunes M, N, hidden, inner lr and steps (the paper does not give alpha or the step count).
- **Gate-0 harness** (`system/nonparametric/gate0.py`, `docs/specs/SPEC_gate0.md`): any support-only channel-weighting idea vs uniform 1-NN L1 on the dev episodes and a per-episode permutation null; rules H1 (>= 2 pts, >= 3/4 users) and H2 (> p95 of the null). The candidate receives a support-only view, so the body-region oracle leak cannot recur. Built-ins: uniform, emg_only, imu_only, support_spread.
- **Laptop env:** this machine had no torch; `.venv/` (git-ignored) from `requirements.txt` with CPU torch 2.15. Suite: 385 passed, 1 skipped, **6 pre-existing failures** that also fail on the clean tree with this env (scipy 1.17 / pingouin 0.7 / numpy 2.4): `test_stats_utils::test_mde_matches_textbook_dz_for_n28` (scipy brentq NaN in the power solver), three `test_group_analysis` tests, two `test_eval_builds::test_zero_shot_detects_transfer_only_when_it_exists`. Follow-up: pin or fix for the new scipy; they do not affect the cluster env.
- **Cluster ask (Kai):** add `Q1 Q1_mlp` to the batch-B debug re-run, then all five `Q1*` to the P-token screen line (`CURRENT_TODOs.md` §2 B/C2 already updated).
- **Covariance discussion (Kai + Fable, late evening):** Riemannian MDM 91.4 (fs64) / 90.4 (env20), IMU-only 91.0, shuffled labels 9.5, rep-gap 95 -> 77. Written up as `COVARIANCE_TUTORIAL.md` §0 and §9-13: the finding (geometric prior transfers, semantic does not), the idea spectrum (augment EncoderMoE = dead end; build on MDM = go), finding vs method paper and venues (ICML / TMLR / TNSRE only), the clustering story as SPD clusters of user signatures, and the planned analyses, of which §12.1 (pooled learned tangent metric; **new, not implemented, not attempted by the first instance**) is the gate that decides whether a meta-learning story remains. Data copy to this laptop: on hold. fs64 sanity rows: pending from the other instance.
