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
Kai agreed 10-way 1-shot is the ideal primary task (he only moved to 3-way because 10-way was so poor). Implemented, all tested locally in the `fl_torch` env (193 tests pass), pushed to `main`:
- **Shared episode cache** (`system/episodes/`): `episode_cache.py` (stdlib: generate/validate/hash/load), `make_episodes.py`, 15 committed files in `system/episodes/kfold7/` (1.3 MB; test 200 / val 50 episodes per user per cell, dev 100; all 9 cells N in {3,5,10} x K in {1,3,5}) + `MANIFEST.json` (the contract; `make_episodes.py --check` and a test regenerate them), `episode_torch.py` / `episode_numpy.py` adapters.
- **Cache-backed evaluation:** `run_episodic_test_eval` / `run_supervised_test_eval` now read the fold's cached TEST episodes and return `per_user_acc`, `mean_acc`, `per_cell`, `episode_source` (file + sha256) and `per_episode` rows; `get_maml_dataloaders` builds its val loader from `val_fold{k}` (`config["val_episode_source"]` = `val` | `dev` | `sampled`; HPO sets `dev`); `meta_evaluate` now selects on the per-user mean. Cells must use the model's trained `n_way` (head size).
- **Stats + aggregator** (`system/analysis/`): `stats_utils.py` (per-user accuracy, user bootstrap CI, paired Wilcoxon/Pratt + W/T/L, Holm, fold-cluster bootstrap, MDE) and `aggregate.py` (rejects wrong episode hash, differing id sets across methods, missing users; averages seeds per user).
- **Running-sum expert mixing** at all six sites; equal to the dense stack to 1e-10 (float64) forward + gradients. **Measured saving only ~14% peak memory** (laptop RTX 4070, `system/MOE/debug_peak_memory.py`): autograd keeps each expert's activations. **Per-expert checkpointing gives wrong gradients under `functional_call`** (12/57 differed once params were substituted) so it is not used. The real fix for 2 kHz EMG is Opus's representation change (experts at T=64).
- **Classical baselines on the cache** (`system/nonparametric/classical.py`, runner `A6_classical_baselines.py`, launcher tokens `A6_*`): `$B` (per-template PCA, verified bit-exact against a direct transcription of the notebook), `$B`-user-PCA variant, kNN-L1, ProtoNet-Euclid, Riemannian MDM, Riemannian tangent-space + LR, MiniRocket + RidgeCV. Old stub `A6_pca_knn_baseline.py` deleted. Costs (laptop CPU, per episode, 10-way 1-shot / 10-way 5-shot): `$B` 0.4/1.0 s, Riemannian ~1.6/2.9 s, MiniRocket 0.5/0.2 s, others <0.1 s; all 9 cells for Riemannian is a few hours per fold.
- **Launcher:** `A7`, `A8`, `A14` tokens REMOVED from `kfold_launcher.sh` because they still use their own samplers (their results would be rejected by the aggregator); `portA` and `A16` likewise. TODO: port them to the cache (A14 = ProtoNet on the M0 encoder; A7/A8 per `SPEC_episode_cache.md` item 4).
- **Env incident (my fault, repaired):** `pip install pyriemann aeon` into the laptop `fl_torch` env broke numpy (conda-installed numpy cannot be cleanly replaced by pip; numba/llvmlite also changed). Repaired by removing the leftover numpy/numba/llvmlite folders from site-packages and reinstalling `numpy==1.26.4`, `numba==0.61.2`, `llvmlite==0.44.0`, `pyriemann==0.9` (0.10+ needs numpy>=2 and fails here), `aeon 1.3`. torch 2.7.1+cu128 (CUDA True), scipy 1.10.1, sklearn 1.7.2, pandas 2.3.3 unchanged; full suite passes. `fl_torch_NOTS.yaml` now pins `pyriemann=0.9`. **The cluster install needs the dry run first (KAI_TODO): the solver may want to change numpy/numba; do not let it touch torch.**
- Not done yet: A14/A7/A8/portA/A16 on the cache; zero-shot script (`SPEC_zeroshot.md`); EMG-FS64 builder (`SPEC_2khz_emg.md`); HPO objective on dev episodes + export (`SPEC_hpo.md`); transformers/MoE-matrix (`SPEC_moe_compute.md`); `--hpo-config` plumbing.

### 2026-10-21 (Sonnet, after the first cluster smoke results)
**Cluster smoke results (old code, fold 0, debug partition; Kai's logs).** A1 and A2 finished; M0 and A4 started, trained, and hit the 15-min cap as expected (M0 reached epoch 2, A4 epoch 8), so the test-eval path of the MAML scripts was NOT exercised. Observations:
- `git: command not found` on compute nodes -> GIT_REV empty (fixed: the launcher resolves it at submit time and exports it; results now carry `git_rev`).
- The stale "Using fold 0: 24 train | 4 val | 4 test PIDs" line was a leftover print of the legacy constants (relabelled).
- 3-way 1-shot, fold 0, 4 test users: A1 head-only 62.1% / full-ft 76.1%, A2 head-only 48.4% / full-ft 79.2% (one fold, 4 users, one seed: not a result).
- M0 epoch = 422 s (train 229 + val 193) vs A4 108 s at 3-way; the 800-episode val pass was half of M0's epoch time (now 100 cached episodes).
- A flat 10-way classifier scores only ~15-25% on the val users (A1 best epoch 25.0%, A2 22.8%, chance 10%). This is a PREVIEW of the zero-shot question, not the pre-registered test (4 users, best-epoch selection biases it up, no 28-user CI), but it sits in or near the "soften the framing" band of `SPEC_zeroshot.md`.
- The untrained A4 already gets 66% 3-way after 10 inner steps at epoch 0 while untrained M0 is exactly chance (33.33% +/- 0): check that M0's inner loop adapts at all at initialisation.

**Kai's decisions.** Train and HPO at **10-way 1-shot** (not both 3- and 10-way; a 10-way model should generalise down to 3-way). Logs/outputs go to `/scratch/my13/kai/runs/paper/sub2/eval/...`, HPO to `.../sub2/hpo/`. The debug partition cap is actually 30 min; keep requesting 15 and use `--smoke` instead.

**Implemented (all tested locally)**
- **10-way default** (`n_way=10`, k=1). A larger head serves smaller-n episodes at eval via its first n outputs (`mamlpp.episode_n_way`, applied in `mamlpp_adapt`, `mamlpp_predict_with_params` and pre-adapt): ONE checkpoint gives the whole 3/5/10-way x 1/3/5-shot grid, no per-cell retraining. Supervised eval was already n-agnostic.
- **`--smoke`** (M0, A3/A4; auto-added by `--debug`): 1 epoch x 48 episodes, 20 val / 40 test episodes; result JSON labelled `smoke` and rejected by the aggregator. Should take ~5 min on an L40S. (A local 10-way M0 run was too heavy for the 8 GB laptop GPU and was killed; the plumbing is covered by unit tests instead.)
- **Launcher:** logs/outputs to `sub2/eval`; git revision resolved at submit time (`-dirty` suffix if tracked files differ); `--hpo-config {legacy|frozen|tuned}` REQUIRED (exported as `HPO_CONFIG`, read by `apply_kfold_fold`, raises if unset); `--emg-repr {env20|fs64}`; A14 tokens (`A14_M0`, `A14_A4`) are back, now on the cache and reading the same fold's checkpoint.
- **Hyperparameter modes** (`ablation_config.apply_hpo_config`): `legacy` = Trial 89 (smoke/regression only, labelled), `tuned` = the ablation's own exported HPO result, `frozen` = M0's applied to the ablation (structural keys protected; supervised ablations re-mirror the MAML eval budget). Reads `models/hpo_best_models/best_hpo_configs.json` (written by `hpo_export_best.py finalize`).
- **HPO ready for the cluster:** `ablation_hpo.py` trains on 20 users and validates on the 4 dev users via the cached dev episodes (first 25/user per epoch, 100/user for the final re-evaluation), 10-way 1-shot objective on the per-user mean, MedianPruner (`mamlpp_pretrain(epoch_callback=...)` added), widened expert/top-k spaces (E in {4,8,16,22,28,36}, top_k in {1,2,4,6,9,12} clipped to E), tuned fine-tune budget for supervised ablations, study name `ablation_<ID>_10w1s_kfold7_v1`, `--replay-trial T --seed S` for the top-3 x 3-seed re-evaluation, `hpo_export_best.py candidates|finalize`. `hpo_ablation_launcher.sh` points at `sub2/hpo`, 60 trials, 12 h per MAML trial. NOT yet exercised on the cluster; the pruning callback and replay mode are untested beyond import/lint.
- **EMG-FS64 (docs/SPEC_2khz_emg.md):** `emg_fs64.py` (filter -> 64 windowed spectral/TD features x 16 channels aligned to the IMU's normalised time), `raw_emg_io.py`, `build_fs64_tensor_dict.py`, `diagnose_2khz.py`; `emg_repr` config (`env20` 16 ch / `fs64` 112 ch) switches the tensor-dict file and `emg_in_ch`. Standardisation constants come from the 4 DEV users only (no fold information; the spec had said the fold's train users).
- **A14** ported to the cache (`A14_protonet_encoder.py`).

**The raw EMG IS on this laptop** (Box: `Yamagami Lab/Data/2024_UIST_dataset/upload/segmented_raw_data/<PID>/*.csv`, plus older preprocessing files in `Meta_Gesture_Project/`), so the 2 kHz preprocessing needs no cluster; only training does.

**Facts measured from the raw files (3,200 trials)**
- **Recording structure** (file-name timestamps, `inspect_rep_timing.py`): every gesture was recorded as ONE contiguous block of 10 repetitions, repetition numbers increase with time, median gap between consecutive reps 5 s (4-7 s); the first-to-last span of one gesture's reps is at most median 90 s (range 43-204 s); a participant's whole 10-gesture session takes a median 19.5 min (13.4-53.5). This replaces the earlier "1-5 minutes" estimate.
- **Segments** (`diagnose_2khz.py`, `docs/diag_2khz_all.json`): length median 3.18 s, p10 2.06 s, p90 4.82 s, range 0.77-12.2 s at 2048 Hz. Segments are tightly cut around the gesture (median active fraction 97%, median lead/trail rest 0.03 s). My earlier guess that long segments include rest margins was WRONG: they are long gestures. The old 4300-sample resample (2.1 s) therefore time-warps the median trial by ~1.5x.
- **No mains line** in the data (59-61 Hz power 1.4x its neighbourhood): the recording rate (2048 vs 2000 Hz) cannot be verified from the signal. Keep 2048 (lab spec); the 60 Hz notch is off by default (`NOTCH_HZ = ()`).
- Power above 450 Hz is 0.15% (hardware band-limited).

**EMG-FS64 first look (dev users only, NOT a result; `compare_emg_repr_dev.py`).** `segfilt_fs64_tensor_dict.pkl` (151 MB, git-ignored like all `.pkl`) was built locally from the 3,200 raw trials; finite, non-dev features have mean ~0.03 / std ~1.0 under the dev-user standardisation. Nearest-prototype (Euclidean) accuracy on the 4 dev users, 40 cached dev episodes per user, per-user mean:

| representation | 10-way 1-shot | 10-way 5-shot |
|---|---|---|
| EMG env20 only | 48.9% | 62.6% |
| **EMG FS64 only** | **77.8%** | **92.1%** |
| IMU only | 48.9% | 66.6% |
| env20 + IMU | 52.4% | 70.2% |
| FS64 + IMU | 70.3% | 87.2% |

A trivial classifier on spectral features of the native-rate EMG beats the 20 Hz envelope by ~29 points at 10-way 1-shot and beats IMU-only by ~29, i.e. EMG was being starved by its representation (archived LIMITATIONS A1 finding that "EMG collapses at 10-way" is plausibly an artefact of the 20 Hz envelope). FS64+IMU is below FS64 alone only because raw concatenation lets the 72 IMU channels dominate the Euclidean distance (a learned model handles this; the A13 modality ablation on FS64 is the real test, `SPEC_2khz_emg.md` E4). Caveats: 4 users, one trivial classifier, features standardised with the dev users' own constants (7 means/stds, negligible but non-zero advantage on exactly these users). Next: 28-user evaluation of the classical baselines on both representations, then M0/A4 on FS64.

**Data anomaly to confirm with Kai:** raw folder `P005/` contains 560 files all NAMED `P105` (different timestamps/bytes from the real `P105/` folder; none named P005). The original loader used the folder, so P005's tensors come from them; `raw_emg_io.KNOWN_NAME_MISMATCH` accepts exactly this pair. Asked in KAI_TODO.

**Why the previous turn did not start the porting/baselines/2 kHz/HPO work:** the cache + eval wiring had to land first (every later item consumes it); the HPO work needed Kai's 10-way decision; and I did not know the raw EMG was on this laptop, which is what made the 2 kHz work doable without a cluster round trip. All of these are now done or unblocked (see this entry).

### 2026-10-21 (late) - cluster layout, P005, and an open anomaly in the 20 Hz EMG
**Cluster layout (Kai).** Repos live under `/projects`, jobs must be submitted from `/scratch`, launchers are COPIED from the repo to `/scratch/my13/kai/runs/paper/sub2/`, data are in `rhf`. The first `kfold_launcher.sh` run failed because the git revision was read next to the scratch copy (not a repo). Fixed: the revision is read with `git -C $CODE_DIR` (the projects repo; `git` itself works on the login node), and both launchers now refuse a scratch copy that differs from the repo copy (prints the `cp` to refresh it). Git provenance is optional in principle but cheap; it is what lets a results JSON be tied to the code that produced it.

**Mamba dry run (Kai):** installs `pooch`, `lazy-loader`, `mne-base`, downgrades `pyriemann` 0.12 -> 0.9; no torch/cuda touched, so it is safe to run for real.

**P005 resolved.** Kai: the P005 folder's files being named P105 is a source quirk (the official manifest lists P105 names too); a derived dataset (parquet) stacked two sessions under P105 but was fixed; the BASE data in Box is right. I verified the tensor dict independently by spatial-signature retrieval against the raw folders: dict P005 matches the P005 folder (cosine 1.00, exact top-1 93%) and NOT the P105 folder (1%); dict P105 matches the P105 folder (93%, 40% exact top-1) and not P005's. So `segfilt_rts_tensor_dict.pkl` (sha256 `eb60c8af8294550f0cf407d472daa1df6679340de564ee299436e93df1fac1c4` for the local copy, built Mar 19 2026 from `filtered_datasets/metadata_IMU_EMG_allgestures_allusers.pkl`, dated 2024-06-06) is correct for P005. Whether the CLUSTER copy is the same file must be checked by hash (KAI_TODO).

**Identity check of the existing tensor dict against the raw CSVs.** Per-trial retrieval by the 16-channel spatial signature (std over time of the $B-normalised envelope) finds the right (class, rep) with exact top-1 of 85% (P004), 97% (P110), 27% (P102) against 1% chance, same-gesture 99/99/59%. So dict[pid][class][rep] is the same execution as the raw CSV (pid, gesture, rep): joining raw-derived EMG-FS64 with the dict's IMU by rep index is valid.

**OPEN ANOMALY (for Opus/Kai, not resolved).** The temporal profile of the existing 20 Hz EMG does NOT reproduce from the documented pipeline (BPF 20-450 Hz, rectify, 200-sample MAV windows step 100, np.interp to 64, $B): correlation of dict rep i with the replicated envelope of raw rep j is the same on and off the diagonal (about -0.3). Dict trials show a flat plateau (about 75% of the 64 steps at the rest level) followed by a burst at the end, whereas the raw CSV segments are 97% active. The dict IMU does not match a simple resample of the raw IMU CSV either (diagonal ~ off-diagonal ~ 0). The spatial structure matches (above) but the TIME axis of the 2024-era source file is different from the final segmented CSVs (different segmentation? time axis handling?). If the 20 Hz EMG's time course is effectively mis-segmented, that alone could explain why EMG contributed little (and why the spectral-feature EMG built from the tight CSV segments is 29 points better). Worth settling before the paper makes any claim about the old EMG input. Suggested check for Opus: reproduce the 2024 file's windowing from `expdef_segraw0_EMG_allgestures_allusers.json` (Mar 2026) and compare segment lengths with the CSVs.

### 2026-10-23 - per-ability-group analysis, TD/LDA baselines, deployment notes (Sonnet)
**Origin.** `10_23_26_ExtraIdeas.txt` idea 1 (within-dataset test of the mechanism: does the MoE-vs-no-MoE gain differ by ability group?) plus peer feedback that journal reviewers will ask for classical baselines (LDA, time-domain features, PCA+KNN) and an offline-vs-real-time discussion. Idea 2 (public-dataset heterogeneity comparison) is on hold, needs explicit approval.
- **Per-group analysis (done, unrun on real data).** `docs/SPEC_group_analysis.md` (pre-registered: M0 - A4 paired per user, primary cell 10-way 1-shot, groups 5/9/14 over the 28 tested users, effect sizes with bootstrap CIs not p-values, both outcomes pre-stated). `system/analysis/group_analysis.py`, CLI `run_group_analysis.py`, `tests/test_group_analysis.py` (7 tests, planted-effect recovery). No local per-user results exist, so no prelim number: legacy L2SO per-user JSONs (if on the cluster) would need an adapter and are tainted by the open 20 Hz EMG anomaly. Add to the rerun checklist: run it for env20 and the adopted EMG representation, 3 seeds.
- **Classical baselines.** PCA+KNN is covered by `dollar_b` / `dollar_b_user` / `knn_l1` (label `$B` as "PCA + L1 1-NN" in the paper). Added `td_knn`, `td_lda` (support-only; at K=1 reduces to nearest class mean because within-class scatter is zero) and `td_lda_pool` (pooled within-class covariance from the fold's TRAIN users, Ledoit-Wolf, then nearest mean in whitened space). Features: trial-level mean and std over time of each channel. New `emg_td.py` (Hudgins MAV, WL, ZC, SSC per 64 windows, fixed a-priori thresholds 0.1 in block-std units), `build_td64_tensor_dict.py` (shares `build_fs64_tensor_dict.build`), `emg_repr="td64"` (64 channels), launcher tokens `A6_td_knn A6_td_lda A6_td_lda_pool` (set `--emg-repr td64`). Runner `A6_classical_baselines.py` got `--emg-repr` and builds the whitener per fold. Needs `segfilt_td64_tensor_dict.pkl`, which has NOT been built yet (needs the raw CSVs on this laptop). Compare directly against M0 rather than adding to the Holm family.
- **Offline vs real-time.** `docs/SPEC_deployment.md`: states what is offline/non-causal in the pipeline and lists the latency/params/memory measurements to take on the final model.
- **EMG rate policy (decided with Kai, pending PI sign-off).** No sampling-rate sweep or tunable knob. Two tables: Meta comparison at native 2 kHz (2048 resampled to 2000), main table at one literature-justified rate fixed in advance (proposed 1 kHz). See `SPEC_2khz_emg.md` §8.
- **NOT done yet:** the learned raw stem (`emg_repr="raw3s"`, `emg_stem`), the raw builder, the A10/A11 rewiring, the 1 kHz FS64 builder option.
