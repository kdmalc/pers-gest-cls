# LIMITATIONS.md — EncoderMoE known limitations tracker

**What this is.** One row per known limitation of the paper and the codebase: what it
is, its status, the evidence, what would fix it, who raised it, and where it belongs in
the paper. This doubles as the resubmission checklist and the draft Limitations section.

**Rules.**
1. Every claim here is read out of this repo or a results artifact, not recalled. Cite
   `file:line` or a results path.
2. Where a planning doc and the code disagree, **the code wins** (the standing convention
   from `rebuttal/REBUTTAL_CODE_FINDINGS.md`).
3. A limitation is only `CLOSED` when there is a number in a committed results JSON or a
   passing test — not when the code exists.

**Status vocabulary:** `OPEN` · `IN PROGRESS` · `MITIGATED` (partially addressed, must
still be disclosed) · `CLOSED` · `IRREDUCIBLE` (cannot be fixed with this dataset;
disclose and move on) · `AUDIT` (a provenance question about an existing number).

**Companions.** `rebuttal/NeurIPS26_Rebuttal_Responses_DRAFT.md` (reviewer text +
drafted responses; Appendix A pending compute, B no-compute paper edits, C PI sign-off)
and `rebuttal/REBUTTAL_CODE_FINDINGS.md` (code audit). Reviewer labels below: **R1**
LPUC, **R2** XH89, **R3** ARAC (confidence 4), **R4** FKXS (originality 1), **MR**
meta-review.

---

## A. Architecture and modelling

### A1. Fusion is inert by construction — no mechanism can make modalities complement
`OPEN` · raised by R1 Q2, R2 W4 ("sharpest correct hit"), R4 W3/Q3, MR Q9

All paper models fuse by `torch.cat([x_emg, x_imu], dim=1)` at the raw input, feeding one
shared `Conv1d(88→64)` in which **82% of input channels are IMU** (72 of 88). There is no
per-modality branch, no modality dropout during training, no per-modality loss, and no
gradient balancing. The MoE router reads the *fused* tensor, so experts cannot specialise
by modality even in principle.

- Evidence: `system/MOE/MOE_encoder.py:912` (headline M0 `backbone`), and the identical
  two lines at `:698`, `:1133`, `:1328`, `:1606`;
  `system/pretraining/pretrain_models.py:142`, `:269`, `:443`. Channel counts at
  `.../test_eval_files/ablation_config.py` (`emg_in_ch=16`, `imu_in_ch=72`).
- Fix: `fusion_mode` flag with per-modality stems / gated / cross-attention /
  modality-expert variants (`system/MOE/fusion.py`), + modality dropout + per-modality
  auxiliary heads. Plan Phase 0.5–0.6.
- Paper: Contribution 1 as written is unsupported without this.

### A2. Per-channel scale is never normalised
`OPEN` · raised by MR Q9, R4 Q2

Normalisation divides each modality block by a **single scalar std** over the whole
flattened `(64, D)` trial, after per-channel demeaning. Each modality lands at unit
*aggregate* std, but a low-amplitude EMG channel stays low-amplitude. The classical
baseline does per-modality PCA specifically to avoid this problem, with a source comment
saying so — `system/nonparametric/eval_knn_proto.py:109-111` ("Prevents high-variance IMU
from dominating the shared filter space").

- Evidence: `_B_normalize_block` / `preprocess_df_B_by_gesture`,
  `system/universal_preprocessing/tensor_saving_with_morphology.ipynb` cell 1
  (`biosignal_switch_ix=72`).
- Fix: `input_norm_mode="learned_affine"` — per-channel z-score then a learnable
  per-channel gain/bias, in the model so it lives in the checkpoint. Note plain
  per-channel z-scoring is **wrong** for EMG: relative amplitude across channels *is*
  the class signal (which muscle is active), so equalise then re-learn the weighting.
- Paper: needs an explicit per-modality normalisation subsection (MR Q9 asks for exactly
  this and the current text does not make the procedure evaluable).

### A3. EMG is starved by preprocessing, not by the model
`OPEN` · not raised by a reviewer — found in the code audit

EMG: 2 kHz → 20–450 Hz Butterworth bandpass → rectify → 100 ms window / 50 ms step MAV →
resample to 64 steps. That is a **20 Hz envelope with a 10 Hz Nyquist**. IMU is 148 Hz →
64 steps with no bandpass. If EMG contributes nothing, this is the first suspect and it
is upstream of every architectural fix.

- Evidence: bandpass `system/universal_preprocessing/EMG_preprocessing/shared_processing.py:40-48`;
  windowing/MAV `.../from_original_repo/ppdsegraw_noFE_windowing.ipynb` cells 8–9, with
  the executed call in cell 11 using 100/50 ms against function defaults of 200/100 ms;
  IMU `FS = 148` in `.../IMU_preprocessing/imu_segraw_moments_pipeline.ipynb` cell 3.
- Fix: EMG envelope-rate sweep {20, 50, 100, 200} Hz at fixed IMU (plan A23). Requires
  regenerating the tensor pickle. Also delivers the 200 Hz matched-input cell already
  committed for camera-ready in rebuttal Block KAIF.

### A4. The MoE is an ensemble, not a mixture — routing does not specialise
`MITIGATED` (measured and reportable) · raised by R1 Q1/Q3, R2 Q4, MR Q6/Q10

Mean per-sample routing entropy is **2.158 nats against a ceiling of log(top_k)=2.197 —
98.2% of maximum flatness** — with 0 dead experts and 8 never-dominant. So M0 (86.78%) vs
A4 (81.75%) is an *ensembling* gap, not a specialisation gap. Two causes are in the
shipped config:

1. `MOE_gate_temperature = 1.529`. `MOEGate.forward` computes `logits / temperature`, so
   **τ > 1 actively flattens.** The HPO selected a flattening temperature because the
   objective rewarded ensemble-averaging.
2. **All three aux losses constrain batch marginals only** — `dense_MOE_aux_loss`
   (KL to uniform, `system/MOE/MOE_encoder.py:427`), `topk_MOE_aux_loss` (Switch,
   `:448`), `importance_loss` (CV², `:476`, and currently disabled at
   `MOE_importance_coeff = 0.0`). **Nothing rewards per-sample sharpness.**

- Evidence: `paper_figures/rebuttal/rebuttal_claims.json` (`H_mean 2.1584`,
  `log_k 2.1972`, `H_frac_logk 0.9824`, `n_dead_experts 0`, `n_never_dom 8`,
  `probe_gest 0.2397` vs chance 0.10, `probe_subj 0.2358` vs chance 0.03125).
- Fix: `routing_mutual_information_loss` (I(x;e) = H(E[w]) − E[H(w)] decomposes exactly
  into the balance term the code has and the sharpness term it lacks), gate-temperature
  annealing to τ<1, trained support-conditional routing, and `expert_ablation_curve` as
  the *causal* specialisation diagnostic.
- **Note for the response:** re-enabling `MOE_importance_coeff` is what reviewers asked
  for and should be done, but CV² is a **third balance term** and cannot fix flatness.
  Say so rather than letting a reviewer find it.
- **Acceptable negative result:** if sharpening raises routing MI but costs accuracy, the
  honest finding is "the MoE's benefit in this regime is ensembling, not specialisation."
  That is publishable and answers R2 Q4 / MR Q10 directly.
- **Now proven, not just argued** (`tests/test_moe_gating.py`): construct two batches with
  an identical per-expert batch marginal but opposite per-sample sharpness (all-uniform vs
  cycling one-hot). `dense_MOE_aux_loss` and `importance_loss` score them **identically**;
  `topk_MOE_aux_loss` scores the sharp one **worse**. Routing mutual information separates
  them cleanly (0 vs log E). The tests fail if a sharpness term is ever added, forcing this
  entry to be updated.

### A5. The 10-way collapse is (probably) a head-initialisation problem
`OPEN` · raised by R2 W1, MR ("3-way vs 10-way"), R3 Q2

1-shot accuracy: 3-way 88.4%, 5-way 82.9%, 10-way 67.7%; and 10-way barely responds to
shots (67.7 → 64.4 → 68.5 at K=1/3/5). MAML must fit a **randomly-initialised**
`128→64→N` MLP head from N support examples in 10 inner steps, using per-parameter
learning rates meta-tuned at **3-way**. At N=10, K=1 that is 10 examples. This also
explains the flat K-response: more shots do not help a random head fit in a fixed 10
steps, but they do help a prototype.

- Evidence: `head_type="mlp"` in `ablation_config.py`; inner loop
  `system/MAML/mamlpp.py:165`; `maml_inner_steps = maml_inner_steps_eval = 10`.
- Fix: prototype-initialised cosine head (ProtoMAML-style), `system/MAML/proto_init.py`.
  Sweep `maml_alpha_init_eval` alongside it — cosine logits scaled by τ≈10 make head
  gradients ~10× larger than the MLP head's.

### A6. Head-level MoE `CosineHead` has an unconstrained temperature
`OPEN` · not raised — found in design review

`system/MAML/MOE_CNN_LSTM.py:27` `CosineHead` keeps `tau` as a raw parameter (init 10.0).
One inner step at `maml_alpha_init_eval = 5.07e-3` can drive it negative and **invert the
classifier**. It is a shipped MoE expert-head option, so leave it alone, but do not build
on it — the new `CosineProtoHead` log-parameterises τ.

---

## B. Evaluation protocol and statistics

### B1. Every headline number is a single seed, and same-config spread is ~3 points
`OPEN` · not raised by a reviewer — this is the most dangerous open item

Three runs of the nominally identical fixed-split config produced **88.46 / 87.58 /
90.68%**. `NUM_FINAL_SEEDS = 5` exists in the config but no script uses it. **Any claimed
effect below ~3 points is currently noise, including the 5-point M0-vs-A4 gap that is the
paper's core evidence for the MoE.**

- Evidence: spread and diagnosis in the `set_seeds` docstring, `ablation_config.py`;
  single-seed admissions at `.../test_eval_files/M0_full_model.py:154`,
  `A5_expert_count_sweep.py:19-20`, `fewshot_grid.py:25`.
- **Root cause found:** `MetaGestureDataset.__getitem__` draws the training subject via
  the **global** `random` module over a PID list whose order comes from the config, so the
  same seed with a different list order gives a different trajectory. The eval path is
  already immune (private `random.Random`, `system/MAML/maml_data_pipeline.py:350-363`).
- Fix: `canonical_pid_order=True` + `episodic_train_rng="private"` (default-off), then
  3 seeds × 32 folds on headline conditions and a published variance decomposition
  (`cross_subject_sd` ≈10–15 pts dominant; `seed_sd`; 32-fold mean bootstrap CI ≈±0.9 pts).
  Note this spread is a **fixed-split artefact** — at 32 folds the mean's SEM is already
  ≈0.45 pts, so 5 seeds × everything (~10,000 GPU-h) is the wrong response.

### B2. Realised query count is 10−K, not the configured `q_query=9`
`MITIGATED` (disclose, do not revise) · raised by R2 W1 indirectly; found in the audit

The eval path ignores `q_query` (`q_query_eval_mode="all_remaining"`), so realised Q per
class is **9 / 7 / 5 at K = 1 / 3 / 5**. Support and query are disjoint slices of one
shuffled list, so **no leak and no published number changes** — but the K≥3 estimates are
less precise than the caption implies (at 10-way K=5 an episode is scored on 50 query
samples, not 90, consistent with ±10.3 vs ±2.9).

- Evidence: `system/MAML/maml_data_pipeline.py:445-455`; measured table in
  `rebuttal/REBUTTAL_CODE_FINDINGS.md` §2b; `episode_shape_log` now records realised counts.
- Paper edit: state realised Q in Table 4's caption and the §4 protocol text, **measured
  per episode rather than taken from the config**, and check the paper does not currently
  assert a fixed Q (the disclosure is not credible alongside the opposite claim). A false
  comment asserting fixed Q at `fewshot_grid.py:62-63` has been corrected in code.
- Do **not** "fix" it by switching to `q_query_eval_mode="fixed"`: no new information at
  K=1, and it breaks comparability with every published number.

### B3. At N=10 every eval episode uses one identical class set and label map
`IRREDUCIBLE` (class-set axis) / `OPEN` (the other two axes) · raised by R2 W1

With exactly 10 gestures, `sorted(rng.sample(all_10, 10))` and label-shuffle off at eval
admit **one** class set and **one** label map. Measured 6/6 identical episodes at N=10 vs
19/20 distinct at N=3. Two of the three episode axes are dead. This is an
evaluation-diversity and variance story, not a leak — the model cannot exploit a fixed
mapping because meta-training randomises the label map.

- Evidence: `maml_data_pipeline.py:350-363`, `:530-533`;
  `REBUTTAL_CODE_FINDINGS.md` §2c.
- **Class-set axis: irreducible.** C(10,10) = 1. No sampler, seed, or budget changes it.
  Anyone proposing "sample more classes" has not counted. Disclose it.
- **Support-rep axis: alive and exhaustible.** Support = rep r for all classes, r = 1..10
  → 10 episodes/user covers it completely and deterministically, hitting each of the 90
  ordered (support-rep, query-rep) pairs per class exactly once. *Strictly dominates* 500
  random episodes: full coverage, zero rep-sampling noise, 50× cheaper.
- **Label→head-unit permutation axis: alive and samplable.** 10! bijections exist;
  exactly one (identity) is currently evaluated.
- Fix: `eval_episode_design="exhaustive_rep"` × `eval_label_perm_mode="permute"`.

### B4. No macro-F1, no confusion matrix, no per-class metric
`OPEN` · relevant to R2 W1 and MR

Accuracy is the only reported metric. On a 10-class problem, accuracy alone **cannot
distinguish uniform degradation from three gestures collapsing into one** — which is
exactly the question the 10-way complaint asks. The only confusion-matrix code is in
`system/MAML/archive/diagnostics.py`, imported nowhere.

- `global_labels` is already plumbed through the collate fn
  (`maml_data_pipeline.py:113-118`), so this is cheap.
- Fix: `.../test_eval_files/eval_metrics.py`; macro-F1 from **pooled** per-class counts,
  not a mean of per-episode F1s. The 10×10 global confusion matrix is the figure that
  answers R2/MR.

### B5. Model selection and reporting use different estimators
`OPEN` · not raised

Checkpoint selection uses query-weighted micro accuracy
(`system/MAML/shared_maml.py:39-40`, `:67`); the reported number is per-episode micro →
unweighted mean over episodes within a user → unweighted mean over users. They coincide
only when Q and episode counts are equal, and diverge whenever a class is dropped or K>1.

- Fix: `model_select_metric="macro_user"` for new runs; replace the ambiguous `std_acc`
  with `cross_subject_sd`, `subject_mean_ci95_bootstrap`, `seed_sd`.
- Also: the reported ± is a cross-subject **population** std (ddof=0) — not an episode CI
  and not seed variance. The paper should say which.

### B6. Chance level is never recorded in the main results JSONs
`OPEN` · `run_episodic_test_eval` records no chance field, so 3-way and 10-way numbers sit
in the same table with no denominator. `A13_modality_ablation.py` does record it — copy
that.

### B7. The paired structure of the evaluation is unexploited
`OPEN` (opportunity, not a defect) · relevant to R2 and R4 crediting the rigor

Test episodes are **byte-identical across models and seeds**: `run_episodic_test_eval`
pins `seed=FIXED_SEED` and `use_label_shuf_meta_aug=False`, and
`_precompute_val_episodes` uses a private `random.Random(seed)` independent of the global
RNG. So every cross-model comparison is already paired at the episode level and nobody is
using it. Zero GPU cost to exploit.

- Fragility: pairing **breaks silently** if the `test_pids` list order changes. Fix by
  recording an `episode_fingerprint` in every results JSON and asserting equality before
  anything is compared as paired.
- Fix: `stats_utils.py` — paired bootstrap CIs over 32 subjects, Wilcoxon, Holm, Cohen's
  d_z, and a published minimum detectable effect.

### B8. Adaptation-budget sweep defaults to VAL **+ TEST** subjects
`OPEN` · not raised — would be a serious finding if a reviewer found it

`.../test_eval_files/num_eval_steps_sweep.py:60` defaults to sweeping
`maml_inner_steps_eval` over `VAL_PIDS + TEST_PIDS`. Defensible only as a diagnostic
figure; if its selected step count feeds any headline number, that is test-set tuning.
The clean sweeps (`M0_inner_steps_eval_sweep.py`, `maml_eval_hp_sweep.py`,
`A11_eval_hpo_extended.py`) are val-only.

- Fix: make `--eval-pids` required; refuse test PIDs without `--allow-test-pids`, which
  stamps `"tuned_on_test": true` into the output so any figure sourced from it self-labels.

### B9. No train ∩ test leakage assertion
`OPEN` · `get_maml_dataloaders` asserts train ∩ val disjointness
(`maml_data_pipeline.py:612-627`) but nothing asserts train ∩ test. L2SO folds are built
correctly; nothing checks it at load time. Add the assertion to
`run_episodic_test_eval` and `run_supervised_test_eval`.

### B10. Declared protocol has already diverged from executed protocol
`MITIGATED` · `fewshot_grid.py` and `fewshot_grid_A2.py` wrote `"test_procedure": "L2SO"`
into their JSONs while evaluating the fixed 4-subject split. Not a leak, but **any audit
keyed on `test_procedure` is wrong for shipped files.** `assert_protocol_consistent` now
stamps an `effective_protocol` inferred from `n_test_pids`/`n_folds`; the aggregator must
prefer it over the declared value.

### B11. Results aggregation is manual, out-of-tree, and partly hardcoded
`OPEN` · `paper_figures/L2SO/NeurIPS_Figures_L2SO.ipynb` reads from a hardcoded Windows
OneDrive path, and the few-shot grid figure holds its numbers as **Python literals**
(cells 30–33), so figure and JSONs can silently drift. Fix:
`.../ablations/aggregate_results.py` emitting `figure_data.json`.

### B12. Zero automated tests in the repo
`IN PROGRESS` — 93 tests, 2.9 s, CPU-only, no cluster data (2026-08-28).

Landed: `pyproject.toml`, `requirements.txt`, `tests/conftest.py` (env + sys.path before
any `ablation_config` import), `tests/synthetic.py` (synthetic `tensor_dict` factory),
`tests/tiny_config.py`, `tests/regen_golden.py`, and four test modules —
`test_episodes.py` (26), `test_config_defaults.py` (7), `test_moe_gating.py` (26),
`test_models_functional.py` (34). Still to write: `test_fusion.py`,
`test_proto_head.py`, `test_modality_dropout.py`, `test_aug.py`, `test_hybrid_np.py`,
`test_metrics.py`, `test_aggregate.py`, `test_smoke_train.py`.

Original state, for the record: no `pytest`, `conftest.py`, `pyproject.toml`, or
`requirements.txt` anywhere.
Verification is inline asserts (which are genuinely good) plus the hand-rolled
`.../test_eval_files/desk_checks.py`. The synthetic-data harness described in
`REBUTTAL_CODE_FINDINGS.md` §7 (20/20 assertions) **was never committed**. Fix: `tests/`
with a synthetic `tensor_dict` fixture; target <60 s on CPU with no data.

### B13. Four sampler keys were consumed but never declared
`CLOSED` (2026-08-28) · found by `tests/test_config_defaults.py`

`modality_mask`, `q_query_eval_mode`, `strict_n_way` and `subject_specific_model` were read
by `get_maml_dataloaders` / `MetaGestureDataset` via `.get(key, default)` but appeared
**nowhere** in `make_base_config()`. So the sampler's fallback was the only source of truth,
and a caller had to already know the key name to change one. This is the same
undeclared-key class that cost A13, A15 and portB cluster jobs to
`KeyError: 'seed'` *after* their pre-flight checks printed PASS.

Fixed by declaring all four in `make_base_config()` with values identical to the existing
`.get()` fallbacks (`"both"`, `"all_remaining"`, `False`, `False`), so the default path is
bit-for-bit unchanged. Guarded by
`test_config_defaults.py::test_tiny_config_covers_every_key_the_sampler_reads` and the
golden key baseline.

### B14. Unselected experts silently stop adapting under top-k routing
`OPEN` (design constraint to respect, not a bug) · confirmed by
`tests/test_models_functional.py`

With `top_k < num_experts`, an expert can receive zero gate mass across an entire batch.
Its weights then affect no output and receive no gradient. `torch.autograd.grad(...,
allow_unused=True)` returns `None` and the update step skips it — **silently**, with no
warning. Observed directly in the test suite: zeroing an unselected expert's weights leaves
the model output bit-identical.

Consequence: **true sparse dispatch (skipping expert compute, capacity-factor token
dropping) is a dead end under MAML.** It would sever the graph through the mask and stop
those experts adapting, without raising. Keep `MOEGate`'s soft-mask top-k. At M0's
`top_k=9 / E=22` this also means ~59% of experts are inert per sample, which is worth
stating alongside the "41% utilisation" figure in A.7.

---

## C. HPO and hyperparameter justification

### C1. The `num_experts` search space starts at 20 — the "plateau" claim needs re-basing
`OPEN` · raised by R1 Q1

The stage-2 space that produced the shipped E=22 is
`[20,22,24,25,26,27,28,30,32,36,40,44]`, so **the HPO could not have chosen E=8.** All
four studies in the repo use `suggest_categorical`, never `suggest_int` — so a Table 2 row
reading "int-uniform [4, 32]" would be factually wrong. Meanwhile in the A5 post-hoc
sweep **E=8 (.9379) is the maximum**, above E=24 (.9241) and E=22, and the spread across
the claimed "plateau" is 5–7 points — larger than the M0−A4 headline effect.

- Evidence: `.../ablations/ablation_hpo.py:399-403`; A5 sweep values in
  `paper_figures/fixed_trts_split/NeurIPS_Figures_trts.ipynb` cell 10;
  `desk_checks.py` already flags this internally.
- Paper edit: correct Table 2's stated space to what was actually searched, add the Fig. 3
  note that its grid is a separate post-hoc sweep, and rest the plateau claim on the A5
  sweep rather than on the search. Use only the ordinal or mechanistic framing for the
  MoEMeta expert-count parallel (see rebuttal Block MOEMETA's DO-NOT-SAY).

### C2. A5's HPO confounded expert count with routing density
`OPEN` · not raised

`ablation_hpo.py:399-405` samples `num_experts ∈ [4..40]` and `MOE_top_k ∈ [4..10]`
**independently**, and `MOEGate.forward` applies top-k only `if top_k < num_experts`. So
every `E=4` trial ran **dense** while `E=40` ran sparse. The A5 *sweep script* is fine
(`top_k = round(E/3)`); the A5 *HPO curve* is confounded. Fix: derive `top_k` from a
utilisation ratio, as `M0_MOE_hpo.py:270-273` already does.

### C3. The headline config was tuned for 1-shot 3-way only
`OPEN` · raised by R2 W1, MR

Trial 89 of `ablation_M0_1s3w_hpo_v1` (val 90.05%) is hardcoded into `ablation_config.py`.
**10-way is the deployment vocabulary and was never the HPO objective.** This is the
mechanical cause of the 10-way complaint. Fix: train at `n_way=10` and evaluate the
10-unit head at N ∈ {3,5,10} — one model, three numbers, one selection rule. (Multi-objective
Optuna over 3/5/10-way is a dead end: it yields a Pareto front you then pick from by hand,
re-introducing exactly the arbitrariness reviewers flagged.)

### C4. No pruner — roughly half the HPO GPU-hours are wasted
`OPEN` · No `trial.report` / `should_prune` anywhere. Fix: `MedianPruner` with
`n_warmup_steps=8` (**must exceed `HPO_BURNIN_EPOCHS=5`** or it kills trials on the
epoch-0 inflation the burn-in exists to ignore), wired via an `epoch_report_fn` kwarg on
`mamlpp_pretrain`. Saves ~42%.

### C5. `MOE_importance_coeff` is disabled
`OPEN` · raised by R1 Q1/Q3 indirectly

Set to `0.0` with the comment "until HPO tunes it", and the module header records
"REMOVED FOR NOW". The `topk_MOE_aux_loss` docstring documents a real observed failure it
was meant to catch: *"one expert can consistently rank first in top-k selection while
maintaining near-uniform soft weights (3x soft imbalance, 1300x dispatch imbalance).
Always pair with importance_loss()."* Re-enable and search it. See A4 for why it is not a
specialisation fix.

### C6. `A11` eval HPO hit both search boundaries in v1
`MITIGATED` · `sweeps/A11_eval_hpo_extended.py:26-38` records that v1's best `ft_lr` was
0.01 (upper bound) and best `ft_steps` was 100 (upper bound), and widened the space. Any
A11 number from v1 was selected at a boundary — check which version produced the
published row.

---

## D. Baselines and comparisons

### D1. No transformer comparison, and the available TST config is a straw man
`OPEN` · raised by R1 W2, R4 Q1, MR Q8

The paper asserts a recurrent inductive bias without showing it. A TST exists and was
trained (`system/pretraining/pretrain_models.py:355`; checkpoints in
`pretrain_outputs/checkpoints/`, Optuna studies in `dataset/optuna_dbs/NAS/`) but never
entered the ablation suite, and MoE-over-TST is blocked by a `ValueError` at
`system/NOTS/exploratory/pretraining/pretrain_hpo.py:296`.

**Quantified straw-man risk:** the shipped TST (`d_model=64, n_heads=4, n_blocks=3,
patch_len=8`) is **≈197k parameters against M0's 5,538,216 — a 28× deficit — and sees 9
tokens including CLS.** Publishing "transformers are worse" from that is exactly what R1
W2 will call out.

- Fix: T1–T4 (subject-specific / transfer / MAML / MAML+MoE) at a param-matched preset
  (`d_model=256, n_blocks=7, patch_len=4` ≈ 5.75M), with `assert_param_budget` enforced as
  an Optuna `TrialPruned` guard so fairness is structural; an independent equal HPO
  budget; and a TST search space that **includes `patch_len ∈ {1,2}`** — 8 gives 9 tokens
  and a 9-token "transformer" is not a test of self-attention. Ship the naive preset
  (T3s/T4s) *next to* the matched one; reporting only the matched one invites "you tuned
  it until it lost", reporting only the naive one is the straw man.
- Free half of the fix, do it regardless: **stop asserting the claim.** Soften the
  inductive-bias assertion to a labelled design hypothesis in Limitations/Future Work.

### D2. `A6_pca_knn_baseline.py` — the non-DL anchor — is a stub that raises
`OPEN` · raised by R1 W1, R4 W2

`raise NotImplementedError`; everything below is commented-out sketch and the
`__main__` block calls a `main` that does not exist. Meanwhile a nonparametric PCA+KNN
baseline is believed to be **competitive with or better than** EncoderMoE, and the real
classical suite (`system/nonparametric/eval_knn_proto.py`) is fully implemented.

- Fix: implement A6 properly and report it. Architecturally, the baseline's advantage is
  legible: it does **per-modality** PCA (see A2) and it is a **prototype** method, which
  is what a 1-shot high-way regime rewards (see A5). Both are addressable in the neural
  model. Insurance: `HybridProtoEnsembleHead` with a meta-learned mixing weight, which
  reduces to the baseline when the neural branch is useless and to the pure learned model
  when `w_np=0`, so it **cannot lose** and both endpoint table rows come from one model.
- **PI sign-off required** (rebuttal Appendix C.1) on whether to report the `$B` baseline
  at all. Non-negotiable floor: **do not criticise a method we did not measure.** The §2
  expressivity critique must be cut or supported.

### D3. Two external baselines are mislabelled as self-ablations
`OPEN`, no compute needed · raised by R1 W1, R4 W2, MR

| Current label | What it actually is |
|---|---|
| "No MAML, No MoE" | cross-subject supervised + 10-step FT = the Côté-Allard [7] transfer recipe |
| "Meta Pretrained EMG" | the pretrained model of Kaifosh & Reardon [11] ("meta-pretrained" parses wrongly in a MAML paper) |
| "Subject-Specific Transfer Learning" | performs **no** cross-subject transfer at all → `Subject-Specific Supervised (within-subject)` |

A presentation failure, not a missing experiment. Relabel all three.

### D4. The Kaifosh comparison as executed is not a fair evaluation of their model
`IN PROGRESS` · raised by R1 W3, R3 Q3, MR Q2/Q5

Four defects, **all biasing against their model**: their preprocessing was never applied;
our input arrives normalised to *signal* s.d. 1.0 while their pipeline expects *noise*
s.d. 1.0, so their internal `x/(32+|x|)` squash operates in its linear corner (measured
post-squash p99 = 0.106 for the condition that produced 62.1%/56.0%); their model is a
9-way multilabel **detector** scored by CLER over Needleman–Wunsch-matched events, not a
window classifier; and our stated parameter count was wrong (**~6.5M**, their discrete-gesture
decoder — the 60M figure is their *handwriting* conformer).

- Evidence: `rebuttal/REBUTTAL_CODE_FINDINGS.md` §3–§4 with the gain-sweep table;
  `V7_checkpoint_param_count.py` computes ≈6,482,953 analytically.
- **Do not say** anything about a literal `2.46e-6` multiply as though it happened — it
  appears nowhere in the repo. Mention it only as a reading that was checked and excluded.
- **Do not say** their gesture pipeline uses hand-crafted features — it does not; MPF
  features serve only their wrist and handwriting decoders.
- Fix: `A11b` gain sweep × {head_only, full} with activation statistics logged per
  condition. Sanity-check the `UNDERSCALED_P99=0.15` / `SATURATED_P99=0.90` thresholds
  against real activation statistics before any text uses the word "fair" — they were
  tuned on synthetic data at an assumed SNR. The noise-floor estimator biases *high* if
  gestures fill the trial, which under-gains and therefore under-states their model — the
  **same** direction as the original defect, so it is not a conservative fallback. Report
  the fixed-gain sweep alongside it.

### D5. Electrode topology mismatch makes the Kaifosh row unrepairable in principle
`IRREDUCIBLE` · raised by R1 W3, R3 Q3

Their filters learned a dense circumferential wrist array (20 mm within-pair, 10.6–15 mm
between channels); our electrodes are distributed across the upper body, so any channel
mapping is arbitrary. Their wrist and handwriting decoders include a rotational-invariance
module; **their gesture architecture has neither that nor rotational augmentation**, so
their filters cannot be realigned to our montage even in principle. Their analog front end
also applies a 20 Hz HP / 850 Hz LP ahead of digital filtering, so "matched preprocessing"
is matched digitally only.

- Fix: none. **Reframe** the row out of the main ablation table into a transfer-study
  subsection asking *does large-scale able-bodied wrist-sEMG pretraining transfer to this
  hardware and population?* — under which the topology mismatch and head conversion become
  part of the question rather than defects in a head-to-head. **PI sign-off required**
  (Appendix C.2).
- Add electrode topology to Limitations as a first-class confound.

### D6. MoEMeta (Wu & Yin, NeurIPS 2025) was not cited
`IN PROGRESS` · raised by R3 W1/Q1 (highest-confidence reviewer), MR Q1

Accepted in substance. MoE + meta-learning has a NeurIPS 2025 precedent, and their
per-relation gating figure is structurally the same analysis as our §4.6 gate-contrast
heatmap. A literal baseline is not runnable (it needs a graph, a candidate set and ranking
objective, symbolic-embedding experts, and its meta-test protocol is a different
adaptation regime by definition; most fundamentally their held-out axis is *labels* and
ours is *users*). Answer with **ports** of its two distinguishing decisions: Port A
support-derived task-level routing, Port B frozen expert bank + small task-conditional
adaptation.

- **Port B must be meta-trained *through* the restricted inner loop.** Meta-training with
  full MAML++ and freezing only at eval evaluates the model in a regime it was never
  optimised for, loses by construction, and R3 at confidence 4 will notice. If time forces
  the eval-only version, disclose the confound and do not lean on the margin.
- Port A is currently eval-time only on a query-routed checkpoint — the weakest version of
  the idea. Promote it to a trained condition (`MOE_task_routing_mode`).
- Contribution 1 must be reframed away from "combining MAML with MoE is novel."
  **PI sign-off required** (Appendix C.2).

### D7. Single dataset, single session, 10 repetitions
`IRREDUCIBLE` (mostly) · raised by R1 W4, R2 W3, MR

- **Cross-session:** one session per participant. Not answerable with the data in hand;
  requires re-recruiting this population. *Open sub-question: do any participants have
  repeat sessions? Even a few would be worth more than the text.*
- **Cross-device:** requires a second montage with the same participants and their same
  self-defined gestures. D5 suggests topology mismatch is a first-order effect.
- **Cross-task:** partially addressed by construction — gesture vocabularies are
  user-defined and differ across participants, so meta-test label semantics are unseen.
  Not the same as cross-*task-family* generalisation.
- **10 reps; fatigue, electrode shift, day-to-day variability:** not capturable. The
  repetition budget was set by participant burden, a binding constraint for this
  population rather than a design oversight.
- Declining a second dataset is defensible **on grounds of hypothesis, not effort**:
  Ninapro and [11]'s benchmark use fixed standardised vocabularies from able-bodied
  participants, instantiating neither targeted property. Never phrase this as
  infrastructure cost. A *cheaper and different* claim is available and worth making:
  "our fusion/head changes also help on Ninapro DB2/DB5" strengthens the method without
  touching the population argument.

### D8. Proroković et al. (MAML for EMG recalibration) is cited but not distinguished
`OPEN`, no compute · raised by R2 W2 · It is reference [17], cited at L115. The overlap is
real but partial: [17] is within-subject session-to-session recalibration on a **fixed**
vocabulary; ours is cross-subject with user-defined semantics. Make both the citation and
the distinction explicit in §2.

### D9. §2 asserts metric-learning is fragile at 1-shot, about a method never run
`OPEN` · raised by R1 W1 · L110–112. `A14_protonet_baseline.py` exists to fix this. **If
A14 drops for time, still soften L110–112** — the free half of the fix is the more
important half. (Note: the prototype head in A5's fix delivers a ProtoNet readout of our
own encoder at step 0 for free, which partially answers this at zero extra cost.)

---

## E. Provenance and integrity audits

### E1. A10's zero-shot path could not have run in the current tree
`AUDIT` — the one genuinely open integrity question · from rebuttal Appendix C.5

Commit `215cd94` renamed the `MetaGestureDataset` kwarg `target_trial_indices` →
`target_trial_reps` and missed six call sites, each raising `TypeError` on construction.
**One of them is A10's prototypical zero-shot path** (`A10_A11_A12_meta_pretrained.py:252`).
All six are now fixed, but: **if an A10 number appears anywhere in the paper, it cannot
have been produced by the current tree.** Trace its provenance before defending it. Worth
ten minutes of `git log`, and worth telling the PI either way — this is exactly the class
of thing that is survivable self-reported and not survivable reviewer-found.

### E2. A1/A2/A3_A4 L2SO fold counts need auditing
`AUDIT` · found in design review

`build_l2so_folds` is copy-pasted in five files and **only `M0_full_model.py` has a
`--fold-idx` CLI argument.** The others run all 32 folds *sequentially in one job*, which
is why `eval_launcher.sh` gives them `TIME="23:00:00"` — but at ~6 h/fold that is ~190 h,
so those jobs cannot have completed. **Any A1/A2/A4 L2SO number in the paper needs its
fold count checked**, including the param-matched A4 row that carries the "capacity does
not explain the MoE gain" claim.

### E3. Subject-specific numbers are final-epoch, not best-epoch
`MITIGATED`, disclose · raised while checking R3 Q2

`A7_A8_subject_specific.py:291` warns that `pretrain_trainer` saves **final-epoch**
weights. A7 (72.2%) and A8 (64.6%) are both final-epoch. This most likely understates
both and applies symmetrically, so the direction of the comparison is safe. State it in
the appendix; use best-epoch checkpointing for camera-ready.

### E4. `normalize_whole_dataset_features` computes one std over the whole dataset
`MITIGATED` · Off the main path — it feeds only the spectral-moment/feature-matrix
notebooks, so Tables 1/4 are unaffected. But any kNN/feature baseline fed by that path has
a global scalar estimated **including test subjects**, so the unqualified sentence "no
test-subject data informs normalization" cannot be written without a qualifier.
Evidence: `EMG_preprocessing/shared_processing.py:154-169`; flagged by `desk_checks.py`.

### E5. The 86.7% / 88.4% dual-protocol presentation caused a meta-review error
`OPEN`, no compute · The meta-review **transposed** the 3-way and 10-way numbers, and the
paper's own presentation is the likely cause: two different numbers are reported for the
same nominal 1-shot 3-way condition (86.7% L2SO Table 1, 88.4% fixed 24/4/4 Table 4) with
only one parenthetical in A.6 explaining the difference. **Name the evaluation protocol
explicitly in every table caption.** This is the single highest-priority correction in the
rebuttal: the meta-review drives the decision and currently contains an inverted premise.

### E6. Two divergent copies of a shared preprocessing library
`OPEN` · `IMU_preprocessing/shared_processing.py` is a near-exact copy of the EMG one; the
only difference is `num_imu_sensors=15` vs `12` in `load_segraw_data`, and the parameter is
unused either way (channels are hardcoded). A live footgun.

### E7. The IMU channel layout is not verifiable from this repo
`OPEN` — blocks rotation augmentation

The 72 IMU channels come from raw pickle column order.
`channel_visualization.ipynb` cell 4 labels IMU channels `[0,1,3]` as "Accel x/y/z", which
is either a plotting bug or evidence of a **non-contiguous** layout. Reported sensor counts
also disagree (12 vs 15) across preprocessing copies.

**Before running any IMU rotation augmentation, dump the 72 column names from the cluster
pickle and encode them as an explicit named layout with a runtime assertion.** A wrong
layout mixes axes across sensors; it will still train, still produce a number, and the
number will be quietly worse. Highest-risk item in the implementation plan.

### E8. Several code paths cannot be constructed from the shipped config
`OPEN`

- `system/nonparametric/metric_train.py` imports `MAML_MOE.maml_data_pipeline`, a package
  that does not exist in the tree (it is `system/MAML/`). Unrunnable as written.
- **`MetaCNNLSTM` is not buildable from `make_base_config()`**: it reads
  `config['cnn_filters']`, which the ablation config never defines (M0 is `DeepCNNLSTM`,
  which reads `cnn_base_filters`). It would `KeyError` at model construction. Found by
  `tests/test_models_functional.py`, which supplies the key so the model is still covered
  by the `functional_call` guards. Either alias the key or drop `MetaCNNLSTM` from the
  supported set — it is currently neither working nor removed.
- `pretrain_approach="frozen_enc_*"` raises `NotImplementedError`
  (`mpp_run.py:384-390`) — encoder freezing is not plumbed through `named_param_dict()`.
  Note Port B needs exactly this, via `include_substrings`.
- `MetaEMGWrapper` is duplicated four times with drift (one has a different `forward`
  signature).

### E9. Anonymity: A.3 names the compute cluster and institution
`OPEN`, no compute · raised by R4 (formatting) · Remove from the revised PDF.
**Do not volunteer** that references [14], [27] and [28] also narrow the author set
considerably — raising it draws attention to a broader problem than the one found. Flag it
for the resubmission checklist instead.

### E10. §4.6 / A.7 episode counts are inconsistent
`OPEN`, no compute · §4.6 says 100 query episodes per user, but A.7's 21,600 samples ÷ 32
users = 675 per user, and 100 × 27 = 2,700 ≠ 675 (675 = 25 × 27). Reconcile.

---

## F. Paper edits requiring no compute

Consolidated from rebuttal Appendix B. Tick as they land.

- [ ] Relabel the three Table 1 rows (D3)
- [ ] Name the evaluation protocol (L2SO vs fixed 24/4/4) in **every** table caption (E5)
- [ ] State realised query counts 9/7/5, measured not configured; check the paper does not assert fixed Q (B2)
- [ ] Note the N=10 single-class-set caveat (B3)
- [ ] Disclose final-epoch weights for the subject-specific rows (E3)
- [ ] Rewrite the Kaifosh preprocessing description around signal-vs-noise s.d. and the gain sweep — **not** a literal `2.46e-6` multiply (D4)
- [ ] Correct 60M → ~6.5M everywhere; present both readings of LP > FT (at K=1 with 6.5M params, full-FT overfitting is the simpler explanation) (D4)
- [ ] Add the band mismatch: our 20–450 Hz vs their 40 Hz HP, and 450 Hz against their 850 Hz analog LP (D4, D5)
- [ ] Add the head/metric conversion (9-way multilabel sigmoid detector → N-way episodic accuracy) to the C-row caption (D4)
- [ ] Move the [11] comparison into its own transfer-study subsection (D5)
- [ ] Add electrode topology to Limitations as a first-class confound, incl. the absent rotational-invariance module (D5)
- [ ] Replace "their device has no IMU" with "their models do not use IMU", incl. L229–230
- [ ] Cite "N=4,800 (largest model, Fig. 2f)" rather than a bare participant count; [11]'s own counts disagree (4,900 / 4,800 / 4,579)
- [ ] Quote [11]'s clinical-population caveat in §4.4
- [ ] Correct Table 2's expert search space; add the Fig. 3 grid note and the A.4 plateau reconciliation (C1)
- [ ] Soften L171–173 from an inductive-bias assertion to a labelled design hypothesis (D1)
- [ ] Soften L110–112 on metric-learning fragility, or support it with A14 (D9)
- [ ] Cut or soften the `$B` expressivity sentence in §2 — **PI sign-off** (D2)
- [ ] Add the MoEMeta citation to §2 plus a §1 sentence; rewrite Contribution 1 (D6)
- [ ] Add the expert-count note to A.4, ordinal or mechanistic framing only (C1)
- [ ] Fix the §4.6 / A.7 episode-count inconsistency (E10)
- [ ] Remove Rice/NOTS from A.3 (E9)
- [ ] Add a per-modality normalisation subsection: statistic, scope, and whether test-user data contributes (A2)
- [ ] Fig. 3 caption wording — **camera-ready only, no reviewer raised it, do not volunteer**

## G. Items requiring PI sign-off

From rebuttal Appendix C:

1. **`$B`** — report it or take only the floor (cut the §2 critique). Floor is
   non-negotiable: do not criticise an unmeasured method that runs on the identical
   dataset and per earlier runs outperforms EncoderMoE. If reported, frame as amortisation
   (O(users × classes) storage, no shared model, no cross-user generalisation) rather than
   accuracy. (D2)
2. **Both reframes** — [11] → transfer study; contribution → away from method novelty.
   (D5, D6)
3. **Statistics family** — adding conditions changes the correction family and shifts
   every §4.5 p-value. Two defensible options, pick one and declare it: refit the omnibus
   over all conditions with Holm–Bonferroni over the full family, or pre-specify the new
   comparisons as a second family. Keep Greenhouse–Geisser (Mauchly W=0.039, ε=0.46) and
   Cohen's d_z on per-subject difference scores. Any cell not evaluable per-participant
   across all 32 participants — including everything on the fixed split — sits outside the
   paired analysis, stated. (B1, B7)
4. **A10 provenance** (E1) and **A1/A2/A4 fold counts** (E2).
