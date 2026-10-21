# FULL_RUNNING_LOG — EncoderMoE resubmission

Newest entries at the bottom of each section. Human-readable summary: `HIGH_LEVEL_STATUS.md`. Old docs (`LIMITATIONS.md`, `PLAN.md`, `READING_LIST.md`, `VENUE_STRATEGY.md`, `HANDOFF.md`, rebuttal PDFs): `archive/`.

**Rules:** Kai alone launches cluster jobs. Per task: Claude writes code + debug script -> Kai runs, reports -> Claude writes full launcher -> Kai runs, reports. No number enters docs without a committed results JSON. Trust order: Kai's chat > newer docs > older docs; flag conflicts to Kai.

## Key inputs / decisions from Kai
- **2026-10-21** Rerun basically everything on one split so numbers stop conflicting.
- **2026-10-21** Split: fixed 28/4 outer split; the 4 are HPO-val only. Within the 28, 7 folds of 20/4/4. Re-stratify as needed. (Replaces 24/4/4 and 32-fold L2SO.)
- **2026-10-21** EMG should be 2 kHz. Raw 2 kHz blows up EncoderMoE memory (all experts computed densely). Options: better expert implementation (maybe 1 expert per sample) or Khushaba spectral-moment features (not the fully decimated 20 Hz); worry that CNN/LSTM on 1D features is odd. Pinned for Opus design work.
- **2026-10-21** Reviewers want comparisons vs SOTA EMG-IMU classifiers, SOTA few-shot/meta-learning, and transformers (swap EncoderMoE for a transformer; transformer experts; transformer instead of LSTM; both). Doable without the PI.
- **2026-10-21** EncoderMoE may lose to `$B`; want to beat it or know. Not wed to EncoderMoE; pursue fixing it and alternatives in parallel.
- **2026-10-21** Work split: Sonnet = code, reruns, scripts; Opus = 2 kHz design, expert restructure, SOTA/alternatives choice. Docs: archive old ones, keep this log + status.
- **2026-10-22** (pending) PI meeting; venue/framing outcome to be recorded here.

## Plan (approved 2026-10-21)
Source: planning-session exploration. Findings that shaped it:
- Protocols were mixed: default for M0/A1/A2/A3_A4/A7/A8/A10-12 is 32-fold L2SO (`ablation_config.py:169`); A13/A14/A15/A11b/portA/B were fixed-split; fewshot_grid JSONs mislabel themselves "L2SO"; A15 mixes protocols; A4=81.75 protocol unverified; EncoderMoE 5-way is 82.9 vs 78.2 in two places.
- Val/test leaks: `maml_eval_hp_sweep.py:115`, `A11_eval_hpo_extended.py:329` default val PIDs include P105 (old test user); `num_eval_steps_sweep` defaults to val+test; `eval_knn_proto` defaults to 4kfcv val users; A14 doc'd test PIDs look like 4kfcv's, not kapanji's (verify).
- 2 kHz: 2 kHz dict (4300 samples, EMG only, no IMU); fused model concatenates EMG+IMU on channels so equal T needed; `front_end_stride` only in `DeepCNNLSTM_EncoderMOE`, applied after concat. OOM root cause: `torch.stack` over all E experts at `MOE_encoder.py:710,949,1145,1337,1618,1643`.
- MoE: experts are CNN-only, LSTM shared after mix. No transformer anywhere in MoE. Sparse dispatch hits B14 (dead gradients under MAML).
- Baselines: ProtoNet/KNN/PCA in `nonparametric/eval_knn_proto.py` and A14; `$B` faithful only in `run_$B.ipynb`; A6 stub; `metric_models.py`, `metric_train.py` (broken import) and `AttentiveMatchingNetwork` unused.

### Sonnet lane
- S1 unify protocol (new split in all scripts, `--test-procedure` on A1/A3_A4/A7_A8, stamp protocol in result JSONs, fix leaks, tests, rerun launcher + `RERUN_CHECKLIST.md`, aggregator with paired stats)
- S2 baselines on same split: `$B` (port from notebook) / A6, Riemannian tangent-space, MiniRocket, ProtoNet, Matching Net, Reptile/ANIL, Proto-MAML, MAML-no-MoE, NeuralSubspace / CovarianceEmbedding, ADANN (after O3)
- S3 transformers: param-matched TST (~5.75M) + naive preset; T1-T4; transformer-expert / CNN-Transformer / dual (after O2 spec)
- S4 running-sum expert mix (+ optional checkpoint/bf16), equivalence test, peak-memory debug script
- S5 2 kHz data plumbing per O1 spec (joined EMG-2kHz+IMU dict, envelope-rate sweep dicts, windowed Khushaba tensors)
- S6 keep docs honest

### Opus lane
- O1 2 kHz EMG representation (modality-specific stem vs windowed Khushaba sequences vs envelope-rate sweep)
- O2 expert restructuring (chunking, shared early layers, top-1/top-k small, Soft-MoE) and transformer variant specs
- O3 SOTA literature review -> `SOTA_COMPARISONS.md` (verify via web; do not trust memory)
- O4 go/no-go on a better method than EncoderMoE (learned `$B`, covariance embedding, Riemannian)
- O5 statistics design; O6 results/claims consistency review

### Order
1) no-cluster work: S1, S4, S2 code, O3, O1/O2 specs. 2) Kai cluster round 1 (debug partition), round 2 (full rerun + classical baselines). 3) O4 go/no-go on matched numbers. 4) S3 + S5 after specs.

## Change log
### 2026-10-21
- Docs: moved `HANDOFF/LIMITATIONS/PLAN/READING_LIST/VENUE_STRATEGY`, the two 10_19 copies and the two PDFs into `docs/archive/`; created `HIGH_LEVEL_STATUS.md` and this file. (Root copies of those md files were already deleted in the working tree; nothing committed yet.)
- Split: added `system/fixed_user_splits/make_28_4_7fold_split.py` (seed 0), `kfold28_hpo4_split.json`, `kfold_split.py` (stdlib loader), `tests/test_splits.py` (4 tests pass with `pytest tests/test_splits.py --noconftest`; full suite needs torch, which is not installed in the local anaconda env).
  - HPO-val 4 = P005, P131, P116, P109 (old kapanji val users; one NDK, two DK, one DNK).
  - Fold test blocks: 0 P102,P108,P126,P132 | 1 P103,P115,P123,P124 | 2 P006,P107,P110,P114 | 3 P010,P105,P112,P127 | 4 P011,P118,P119,P122 | 5 P004,P106,P111,P125 | 6 P008,P104,P121,P128. Each fold has exactly 2 Disabled Non-Kapanji test users; val of fold k = test block of fold (k+1)%7.
  - Caveat: old HPO trained on users that are now test users (no label leakage; hyperparameters picked on val users only). Re-HPO is optional, Kai's call.

## Next
S1: replace L2SO / `hpo_test_split` in `ablation_config.py` and the eval scripts with `kfold_split.get_fold`, add `--fold-idx` in 0..6, then S4.

### 2026-10-21 (later) — S1 started
- `ablation_config.py`: default `test_procedure` is now `"kfold7"`; default train/val/test PIDs = fold 0; `all_PIDs` = the 28 evaluated users (was 32); new `build_kfold_folds()`, `apply_kfold_fold(config, i)`, `get_hpo_split()` (train on the 28, val on the 4 HPO users), config keys `fold_idx`, `hpo_val_PIDs`; `assert_protocol_consistent` knows `kfold7`. Legacy `L2SO` / `hpo_test_split` remain selectable and restore the old PIDs themselves. `tests/golden_base_config_keys.json` edited by hand (+2 keys, 118) — **not regenerated; needs torch to run `tests/regen_golden.py` and verify**.
- `M0_full_model.py`: `--test-procedure kfold7` (default) with `--fold-idx 0..6`; no `--fold-idx` runs all 7 sequentially.
- NOT yet done for S1: A1, A2, A3_A4, A7_A8, A10-A12 (own L2SO loops); A13/A14/A15/A16/A11b/portA/B/fewshot grids/sweeps (hard-coded old split or val PIDs); `eval_launcher.sh` (fold fan-out is hard-coded 32 PIDs; needs 7); `eval_knn_proto` eval PIDs; sweep val-PID leak fixes; protocol stamp in results; aggregator. Nothing here has been run — local env has no torch.

### 2026-10-21 (evening) — Kai's inputs, audit, S1 wiring
**Key inputs from Kai**
- Redo ALL HPO on the new split. HPO the transformers too, for a fair comparison. Consider HPO for the ablation models as well; maybe run both "frozen M0 hyperparameters" and "re-tuned per ablation" and report how much it matters.
- Wire the remaining scripts. Mark clearly which tasks need Opus.
- Co-advisor asked for a read-only evaluation-pipeline audit (+ label-permutation / zero-shot question). **Deliverable: `EVAL_PIPELINE_AUDIT.md`. Code changes the audit proposes are NOT made until Kai/co-advisor approve.** Coding rule from the co-advisor for later changes: no silent fallbacks (no `dict.get` defaults, no swallowing `except`, no silently skipping missing users/files; raise instead).

**HPO design decision (Claude, to be confirmed by Kai)**
- HPO split = fold 0's 20 train users for training, the 4 dev users (P005, P131, P116, P109) for validation, no test users (`ablation_config.get_hpo_split()`, used by `ablation_hpo.apply_fold_to_config`). Matches the per-fold training-set size so hyperparameters transfer.
- Two tiers for ablations: Tier A = every headline comparator (M0, A1, A2, A4, transformers, MAML-no-MoE, ProtoNet/other learned baselines) gets its own equal-budget HPO; Tier B = the same ablation run with M0's frozen hyperparameters. Report both for A1-A4 to quantify how much tuning matters; the remaining component ablations use Tier B only (cost).
- NOT done yet: plumbing from HPO results back into eval scripts (best params live in Optuna journals; `make_base_config` still hard-codes legacy Trial 89 values). Needs the real param-name mapping from the journals after the first HPO run; see "Pending".

**Done (S1 wiring, all untested on cluster; local env has no torch, so only `py_compile` + pyflakes + torch-free tests)**
- `ablation_config.py`: `add_fold_arg`, `resolve_fold_ids`, `run_kfold`, `kfold_tag`, `get_hpo_split`, `apply_kfold_fold`.
- Scripts now kfold7-only with `--fold-idx` (0..6; omitted = all 7 sequentially): A1, A2, A3/A4, A7/A8 (evaluates the fold's 4 test users; all 28 if omitted). M0 default kfold7 (legacy L2SO/hpo_test_split kept).
- Single-fold scripts with REQUIRED `--fold-idx`: A13, A14, A15 (pool = the fold's 20 train users; sweep N in {4,8,12,16,20}), A16, fewshot_grid / _A2 / _A4, portA, portB.
- Selection discipline: `num_eval_steps_sweep.py` now requires `--eval-pids` and refuses non-dev users; `M0_inner_steps_eval_sweep.py` uses the dev users; `maml_eval_hp_sweep.py` / `A11_eval_hpo_extended.py` require `--val-pids` (no default). `eval_knn_proto.py` requires `eval_PIDs` explicitly.
- `ablation_hpo.py`: HPO split = 20 train / 4 dev val / no test.
- New launcher `system/NOTS/paper/ablations/launchers/kfold_launcher.sh` (one job per token x fold; `--folds`, `--debug`, `--dry-run`; `SHOW_WRAP=1` prints the job body). Verified with `bash -n` and dry-runs only.
- Tests: `tests/test_splits.py` (passes locally), `tests/test_kfold_config.py` (needs torch; unrun).

**Not wired yet (deliberately)**
- A10/A11/A12/A11b (2 kHz + Kaifosh): will raise "Unknown test_procedure 'kfold7'" if run. Waits on the O1 design (**Opus**), since those are being redesigned.
- A16 needs a manifest of per-fold checkpoints (`find_a16_checkpoints.py` still assumes L2SO); do after the reruns exist.
- A5 (expert-count sweep), portA launcher token, steps sweeps in the new launcher; A6 stub.
- Legacy `eval_launcher.sh` is untouched and now LEGACY.

## Pending (who does what)
| Task | Model | Notes |
|---|---|---|
| O3 SOTA literature review | **Opus** | web search; defines baselines for S2/S3 |
| O1 2 kHz EMG representation + A10-A12/A11b redesign | **Opus** | unblocks S5 |
| O2 expert restructuring, transformer-expert specs, transformer HPO space | **Opus** | unblocks S3 |
| O4 go/no-go on better method than EncoderMoE | **Opus** | after first matched numbers |
| Audit-driven fixes (label permutation at test, shared episode generator, rep ids, leak assert on test path, zero-shot script) | Sonnet | **only after approval** of EVAL_PIPELINE_AUDIT.md; the shared episode generator affects every baseline so design review (Opus) first |
| HPO-result plumbing (Optuna journal -> config), per-tier runs | Sonnet | after first HPO run produces journals |
| S4 running-sum expert mixing + peak-memory debug script | Sonnet | next |
| S2 `$B` port, Riemannian, MiniRocket, Matching Net, Reptile/ANIL, learned-`$B` | Sonnet | next; needs O3 list for the rest |
| S3 transformer baselines (param-matched) + HPO | Sonnet | after O2 spec |

### 2026-10-21 (late) — co-advisor recap of the audit
**Co-advisor decisions (relayed by Kai)**
- Item 4: permute labels at test AND remove the `use_label_shuf_meta_aug` flag entirely (do not flip it at ~13 call sites). Flat-global-head baselines lose their cross-user label prior; re-initialise the head or map it through the episode permutation, and say so in the paper. MAML pre-adaptation accuracy is chance by construction, so it is not a zero-shot measure.
- Item 6: shared episode cache is TOP PRIORITY; finding 6.3 means current baseline comparisons are invalid. Every method consumes identical saved episodes (fold, user, seed, rep ids).
- Finding 4.4 (rep order/sessions) is the biggest real-world vulnerability: resolve sessions/contiguity/donning from collection records; ordered support/query (earliest K support) becomes a MAIN-table result; cross-session evaluation if any participant has >1 session. [OVERRIDDEN by Kai 2026-10-21, see below: no ordered split; one session, no cross-session possible.]
- Zero-shot test: pooled + per-gesture accuracy + representation-level nearest-centroid version; decide the framing consequence BEFORE seeing the number.
- >=3 seeds for main methods and the strongest 1-2 baselines on all 7 folds. Stats on per-user means (28 users), paired Wilcoxon vs best baseline; never episodes as independent.
- Finding 1.3: re-run a small architecture/expert-count check on the 4 dev users (preferred) or disclose.
- Add a test: build a fold, generate episodes, assert user disjointness (train/val/test/dev), support/query rep disjointness via rep_idx in the real test path, and identical episode-file hash across every method's results JSON.
- Keep the 20/4/4-with-dev-users-excluded split; state in the paper that dev users are excluded from all results.
- Recommended order: (1) sessions question from raw records, (2) code hygiene, (3) episode generator/cache + port all baselines, (4) zero-shot experiment then commit to framing, (5) final dev-only HPO + architecture check, (6) full 7-fold runs with 3 seeds, (7) rewrite results from scratch.

**Kai's question on query size.** Re-checked in code: `q_query` is nominal; train slices `trial_indices[K:K+q_query]`, eval ignores it (`all_remaining`), so realised Q = 10-K (9/7/5). Support and query are disjoint slices of one shuffled list, so there is NO overlap/leak; it is under-sampling at larger K versus the configured 9 (archived LIMITATIONS B2, `MITIGATED` by disclosure; `episode_shape_log` records the realised counts; mislabelled comment at `fewshot_grid.py` was corrected earlier). Not changed in this session; report realised Q in captions. With ordered sampling Q stays 10-K.
Related, unfixed: B3 — at N=10 every eval episode uses one identical class set and label map; permuting labels at test (item 4) addresses the label-map half.
Old notes (D7) say one session per participant; unconfirmed. Added `system/universal_preprocessing/inspect_rep_structure.py` to see what the metadata file can show.

**Docs added:** `OPUS_GUIDE.md` (share with Opus), `KAI_TODO.md` (immediate blockers only).

**Ready now vs needs Kai (Claude's call)**
| Item | Status |
|---|---|
| Running-sum memory rewrite (S4) | Safe, can start now, BUT I cannot run the equivalence test without torch locally (KAI_TODO 3). Peak-memory check needs Kai on the debug partition later. |
| Audit hygiene fixes (raise on missing PIDs, drop `.get` defaults, disjointness asserts, rep ids, remove permutation flag) | Approved by co-advisor; waiting for Kai's "go fixes" + local torch so I can test. |
| `$B` port, Riemannian, MiniRocket | Classifier cores (support/query arrays in, predictions out) can be written now; the runners must wait for the episode cache (Opus P1 spec) or they would repeat finding 6.3. Needs `pyriemann`, `aeon`/`sktime` on the cluster env (check). |
| Shared episode cache | Opus design first (P1), then Sonnet. |
| Zero-shot experiment | Opus design/framing first (P3), then Sonnet. |
| Any cluster run | Kai only; see KAI_TODO. |

### 2026-10-21 (night)
**Kai's answers about the data:** one session per participant; sensors never removed; later-indexed reps were recorded later; reps probably back-to-back (unsure); gestures 0.5-3 s; EMG recorded at 2048 Hz. Implications: no cross-session evaluation possible; ordered support/query is a real within-session temporal split. Flagged: code uses `FS=2000` and 2 kHz-dict trial lengths (0.8-12.2 s) exceed the stated gesture durations; recorded in `OPUS_GUIDE.md` for P4.

**Environment:** the laptop has a conda env `fl_torch` (Python 3.10.10, torch 2.7.1+cu128, CUDA available) at `C:\Users\kdmen\anaconda3\envs\fl_torch`. Installed `pytest` into it; the full suite (110 tests incl. the new split/kfold tests) passes there, so local torch is no longer a blocker. `pyriemann`/`aeon` NOT installed locally yet (a dry run would bump numpy 1.24.3 -> 1.26.4 and numba; will install when S2 starts). Added `pyriemann`, `aeon`, `pytest` to `fl_torch_NOTS.yaml`. The cluster has mamba only and its env is the prefix env `/projects/my13/kai/meta-pers-gest/envs/fl-torch`, so it gets the packages via `mamba install -p <prefix> -c conda-forge ...` (dry-run first; command in KAI_TODO 2a) rather than rebuilding from the yaml.

**q_query cleanup (done, tests pass).**
- Finding: with 10 reps and `q_query=9`, the cap never binds for K>=1 (10-K <= 9), so `q_query` was a no-op in training and eval in every published run; realised Q = 10-K = 9/7/5. The cap only binds for deliberate small values (HPO uses `q_query=3`, some exploratory scripts 5).
- Removed the `q_query_eval_mode` flag (its "fixed" branch was dead and a trap) from `MetaGestureDataset`, `get_maml_dataloaders`, `make_base_config`, the tiny test config and the golden keys file (117 keys). Eval now has exactly one behaviour: score all non-support reps.
- Documented `q_query` as a TRAINING-ONLY cap in the module/class docstrings; rewrote the misleading "q_query is held fixed across all grid cells (standard practice)" comments in `fewshot_grid.py`, `fewshot_grid_A2.py`, `fewshot_grid_A4.py` and `desk_checks.py`.
- Tests: replaced the fixed-mode test with `test_training_q_query_is_a_cap_and_eval_ignores_it` and `test_q_query_eval_mode_switch_is_gone`.
- Decision: do NOT hold Q fixed across K. A fixed Q (e.g. 5) would equalise estimator precision across K but throw away data; everything is scored on all non-support reps and the paper reports realised Q.
- Left alone (legacy, off the new path): `pretraining/pretrain_finetune._sample_episode` (own sampler with explicit rep pools; asserts loudly if too few reps) and `contrastive_data_pipeline.py`. The ~13 `config.get("q_query", None)` call sites belong to the pending `.get`-removal in the audit fixes.

### 2026-10-21 (late night) — Kai's decisions + audit fixes implemented
**Kai's decisions**
- "All the audit fixes look good" -> approved and implemented (below).
- **No temporal/ordered split.** If the 10 reps were back-to-back they were recorded within ~1-5 minutes, so rep order carries no meaningful drift; ordering support before query would not test anything real. Random support/query reps stay the protocol (this overrides the co-advisor's suggestion that ordered be a main result). The paper states the limitation instead: single session, minutes-scale, no cross-session claims.

**Implemented (117 tests pass in the `fl_torch` env; cluster untested)**
- Label permutation: `use_label_shuf_meta_aug` removed everywhere on the new path (dataset class, `get_maml_dataloaders`, `make_base_config`, eval functions, sweeps, A13, portA, HPO, tests, golden keys). Classes are drawn in random order, so each episode's label map is a fresh permutation in training, validation AND test. N=10 still has exactly one class *set* (irreducible, B3) but now many label maps; the old identity-map test was replaced. Flat-global-head baselines (A1/A2) already re-initialise their head at eval (`replace_head_for_eval`); wording for the paper is pending Opus P2.
- Missing users now raise `KeyError` (`MetaGestureDataset`) instead of being silently dropped.
- Rep ids: every sample carries `rep_idx` (tensor row = rep number - 1); `maml_mm_collate` stacks it; `global_class` is now read strictly (the `.get(..., label)` fallback is gone).
- Always-on check: `mamlpp.assert_support_query_rep_disjoint` runs at the top of every `mamlpp_adapt_and_eval` call and in the supervised test eval (not only in debug mode).
- Disjointness: `kfold_split.assert_fold_disjoint` (train/val/test pairwise + dev users excluded) is called by `apply_kfold_fold`; `get_maml_dataloaders` now checks train/val, train/test and val/test on the configured splits; `run_episodic_test_eval` / `run_supervised_test_eval` refuse test users that are also train/val users.
- `.get` defaults removed on the eval/data path: `modality_mask`, `strict_n_way`, `subject_specific_model`, debug flags (`get_maml_dataloaders`), `q_query` at 7 eval/HPO/sweep call sites. NOT removed (out of scope, legacy or non-eval): `config.get` throughout `mamlpp.py`, `MOE_encoder.py`, `eval_knn_proto.py` internals, legacy `early/` and `exploratory/` scripts and `mpp_run.py` (those legacy entry points will now raise `KeyError` for the keys above if run; they are off the kfold7 path).
- New tests: `tests/test_eval_protocol.py` (real fold PIDs end to end: fold users disjoint and exclude dev, support/query rep disjointness at K=1/3/5 through the collate, labels permuted per episode, `mamlpp_adapt_and_eval` refuses overlapping reps, missing user raises) and `tests/kfold_split_loader.py` helper.
- Still pending from the co-advisor's list: shared episode cache + identical-episode-hash test (needs Opus P1 spec), zero-shot experiment (Opus P3), >=3 seeds plumbing, per-user val aggregation in `shared_maml.meta_evaluate`, dev-only architecture check, statistics module.

### 2026-10-21 (Opus session 1) — design specs
Kai merged the branch into `main` (work continues on `main`) and switched to Opus for the OPUS_GUIDE tasks while running the cluster smoke tests.

**Written (all in `docs/`):**
- `SPEC_episode_cache.md` (P1): committed, hashed, data-free episode files per fold (test 200 / val 50 eps per user per cell) + dev (100); N in {3,5,10} x K in {1,3,5}; per-(user,cell) seeded streams; torch + numpy adapters; per-episode results contract (`n_correct`, ids, sha256); aggregator asserts identical hash and id sets across methods; migration order; A7/A8 re-specified as train-on-support per episode (20-episode subset).
- `SPEC_protocol.md` (P2): **recommends 10-way as the primary task (10-way 1-shot = primary cell) — needs Kai's OK**; Methods paragraph; scope/limitations (single session, minutes-scale, pre-segmented, one montage) with suggested text; "user-defined" wording gated on zero-shot.
- `SPEC_zeroshot.md` (P3): Z1 supervised 10-way, Z2/Z3 nearest-centroid in Z1 / M0 embeddings, Z4 Riemannian+LR; outcome thresholds and framing consequences fixed now (<=20% keep framing; 20-40% soften; >40% drop idiosyncratic framing and make zero-shot a primary baseline).
- `SPEC_stats.md` (P8): user as unit; bootstrap CI over users; paired Wilcoxon (Pratt) + Holm over a pre-registered primary family; fold-cluster bootstrap sensitivity; 3 seeds for main/best methods; MDE reporting; `system/analysis/` tooling (not `system/stats/`, which is Kai's notebook folder).
- `SOTA_COMPARISONS.md` (P6, web-verified): no single EMG+IMU SOTA exists; best reimplementable SOTA fusion model is EMG-CrossFormer (Atzori group, Jul 2026, MIT code); FS-HGR, ADANN, TEMGNet as recognisable EMG baselines; Riemannian, MiniRocket, `$B`, ProtoNet, fine-tune (dev-tuned budget) as Tier 1; key framing citations (Eddy et al. 2024 zero-shot cross-user works for fixed gestures; ASSETS '23 ability-specific gestures; Sarwat et al. 2024 one-shot ProtoNet in stroke with EMG+FMG+IMU).
- `SPEC_2khz_emg.md` (P4): the OOM is a representation problem: compress EMG before the experts. Primary: EMG-FS64 (7 windowed spectral/TD features x 16 ch on the 64 IMU-aligned steps; FS=2048; zero-phase band-pass + 60 Hz notch). Secondary: EMG-RAW learned strided stem on a 3 s native-rate crop. Classical baselines get their natural representation (Riemannian on native-rate covariance). Envelope-rate sweep dropped. A10-A12/A11b demoted to Tier 3; A12 retired.
- `SPEC_moe_compute.md` (P5): only the exact running-sum rewrite (+ optional checkpointing) for memory; X0-X4 MoE matrix with the key control X2 = uniform-mixture (is the gate doing anything?); task-conditioned routing X4 is new code (the archived `MOE_task_routing_mode` flag never existed); transformer variants T0-T3 with a functional attention implementation (no `nn.MultiheadAttention`), param-matched presets.
- `SPEC_hpo.md` (P7): dev-user objective at the primary cell; equal 60-trial budgets per family; MedianPruner; top-3 x 3-seed re-evaluation; widened expert/top-k space; eval-time inner steps tuned on dev (replaces test-user sweeps); tuned-vs-frozen table for A1-A4; architecture sanity check; `best_hpo_configs.json` export and required `--hpo-config` flag.
- `DECISION_method.md` (P9): go/no-go rules R1-R5 and venue mapping pre-registered; verdict empty until matched numbers exist.

### 2026-10-21 (Sonnet, implementing the Opus specs)
Kai agreed 10-way 1-shot is the ideal primary task (he only moved to 3-way because 10-way was so poor). Implemented, all tested locally in the `fl_torch` env (176 tests pass), pushed to `main`:
- **Shared episode cache** (`system/episodes/`): `episode_cache.py` (stdlib: generate/validate/hash/load), `make_episodes.py`, 15 committed files in `system/episodes/kfold7/` (1.3 MB; test 200 / val 50 episodes per user per cell, dev 100; all 9 cells N in {3,5,10} x K in {1,3,5}) + `MANIFEST.json` (the contract; `make_episodes.py --check` and a test regenerate them), `episode_torch.py` / `episode_numpy.py` adapters.
- **Cache-backed evaluation:** `run_episodic_test_eval` / `run_supervised_test_eval` now read the fold's cached TEST episodes and return `per_user_acc`, `mean_acc`, `per_cell`, `episode_source` (file + sha256) and `per_episode` rows; `get_maml_dataloaders` builds its val loader from `val_fold{k}` (`config["val_episode_source"]` = `val` | `dev` | `sampled`; HPO sets `dev`); `meta_evaluate` now selects on the per-user mean. Cells must use the model's trained `n_way` (head size).
- **Stats + aggregator** (`system/analysis/`): `stats_utils.py` (per-user accuracy, user bootstrap CI, paired Wilcoxon/Pratt + W/T/L, Holm, fold-cluster bootstrap, MDE) and `aggregate.py` (rejects wrong episode hash, differing id sets across methods, missing users; averages seeds per user).
- **Running-sum expert mixing** at all six sites; equal to the dense stack to 1e-10 (float64) forward + gradients. **Measured saving only ~14% peak memory** (laptop RTX 4070, `system/MOE/debug_peak_memory.py`): autograd keeps each expert's activations. **Per-expert checkpointing gives wrong gradients under `functional_call`** (12/57 differed once params were substituted) so it is not used. The real fix for 2 kHz EMG is Opus's representation change (experts at T=64).
- **Classical baselines on the cache** (`system/nonparametric/classical.py`, runner `A6_classical_baselines.py`, launcher tokens `A6_*`): `$B` (per-template PCA, verified bit-exact against a direct transcription of the notebook), `$B`-user-PCA variant, kNN-L1, ProtoNet-Euclid, Riemannian MDM, Riemannian tangent-space + LR, MiniRocket + RidgeCV. Old stub `A6_pca_knn_baseline.py` deleted. Costs (laptop CPU, per episode, 10-way 1-shot / 10-way 5-shot): `$B` 0.4/1.0 s, Riemannian ~1.6/2.9 s, MiniRocket 0.5/0.2 s, others <0.1 s; all 9 cells for Riemannian is a few hours per fold.
- **Launcher:** `A7`, `A8`, `A14` tokens REMOVED from `kfold_launcher.sh` because they still use their own samplers (their results would be rejected by the aggregator); `portA` and `A16` likewise. TODO: port them to the cache (A14 = ProtoNet on the M0 encoder; A7/A8 per `SPEC_episode_cache.md` item 4).
- **Env incident (my fault, repaired):** `pip install pyriemann aeon` into the laptop `fl_torch` env broke numpy (conda-installed numpy cannot be cleanly replaced by pip; numba/llvmlite also changed). Repaired by removing the leftover numpy/numba/llvmlite folders from site-packages and reinstalling `numpy==1.26.4`, `numba==0.61.2`, `llvmlite==0.44.0`, `pyriemann==0.9` (0.10+ needs numpy>=2 and fails here), `aeon 1.3`. torch 2.7.1+cu128 (CUDA True), scipy 1.10.1, sklearn 1.7.2, pandas 2.3.3 unchanged; full suite passes. `fl_torch_NOTS.yaml` now pins `pyriemann=0.9`. **The cluster install needs the dry run first (KAI_TODO): the solver may want to change numpy/numba; do not let it touch torch.**
- Not done yet: A14/A7/A8/portA/A16 on the cache; zero-shot script (`SPEC_zeroshot.md`); EMG-FS64 builder (`SPEC_2khz_emg.md`); HPO objective on dev episodes + export (`SPEC_hpo.md`); transformers/MoE-matrix (`SPEC_moe_compute.md`); `--hpo-config` plumbing.
