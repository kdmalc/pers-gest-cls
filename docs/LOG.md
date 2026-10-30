# LOG (context and design decisions that are not immediately critical)
Full history: `docs/archive/FULL_RUNNING_LOG.md`. Status table: `docs/archive/HIGH_LEVEL_STATUS.md`. Immediate blockers (long form): `docs/archive/KAI_TODO.md`.

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
