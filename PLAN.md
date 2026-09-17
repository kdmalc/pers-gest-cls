# PLAN.md — phased resubmission plan

**What this is.** The phased implementation plan `LIMITATIONS.md` refers to by name
throughout ("Plan Phase 0.5–0.6", "plan A22/A23", "plan 0.6/0.7", "plan C3") but that
was never actually committed. This file is that plan: it doesn't repeat evidence
`LIMITATIONS.md` already has (linked by ID — A1, D9, etc.), it sequences the fixes into
phases and states what each phase is *for*.

**Relationship to other docs.** `LIMITATIONS.md` is the tracker (what's wrong, evidence,
status). `VENUE_STRATEGY.md` is the target (where to send it, and the 4 gates for a
main-track ML resubmission). `READING_LIST.md` is what to read before writing each
piece. This file is the *sequence* connecting them: what to do, in what order, and why
that order.

**Status, 2026-09-17.** Written after reading the full closed OpenReview thread and
syncing `LIMITATIONS.md` (see its 2026-09-17 note). Decision made with the PI-equivalent
(you): chase a top ML venue again — ICLR preferred if the gates in Phase 3 clear, else
ICML, deprioritizing a straight NeurIPS resubmission given at least one likely-recurring
reviewer. If Phase 3's gates are not clearing after Phases 0–2, `VENUE_STRATEGY.md`'s
IMWUT/TNSRE/TMLR path is the documented fallback, not abandoned.

---

## Phase 0 — trust and measurement infrastructure

Nothing built on top of this phase is credible without it; do it first regardless of
which fix looks more exciting.

- **B1**: fix the training-time RNG (`canonical_pid_order` + private episodic RNG), then
  3 seeds × 32 folds on every headline condition, with a published variance decomposition
  (`cross_subject_sd`, `seed_sd`, bootstrap CI). The current ~3pt same-config spread is
  larger than several claimed effects, including the paper's own headline M0-vs-A4 gap.
- **B7**: exploit the already-paired episode structure — `stats_utils.py` with paired
  bootstrap CIs, Wilcoxon, Holm correction, Cohen's d_z, and a stated minimum detectable
  effect. Zero GPU cost; do this even before B1's reruns land, on existing results.
- **B3/B4/B5**: macro-F1 + pooled confusion matrix (answers the "is 10-way collapse
  uniform or three-gestures-merging" question directly), consistent model-selection
  metric, exhaustive-rep eval episode design at N=10.
- **B12**: keep extending the test suite (`test_fusion.py`, `test_proto_head.py`,
  `test_modality_dropout.py`, `test_aug.py`, `test_hybrid_np.py`, `test_metrics.py`,
  `test_smoke_train.py`) alongside each phase below rather than after — the existing 93
  tests already caught B13/E8-class bugs (an undeclared config key, an unbuildable model)
  before they cost cluster time twice.

**Exit criterion:** a headline number can be reported with a seed/fold-based CI, not a
single run.

---

## Phase 0.5 — fusion (your priority #1)

Targets `LIMITATIONS.md` A1/A2/A3. A1's modality ablation (now landed, see
`LIMITATIONS.md`) makes this priority concrete instead of hypothetical: **the current
fused model does not beat its own IMU-only branch at any vocabulary size, and EMG-only
collapses hardest exactly where the paper's deployment target (10-way) lives.** That is
the fact this phase has to change.

1. **A3 first, cheapest lever**: sweep the EMG envelope rate {20, 50, 100, 200} Hz at
   fixed IMU rate. EMG is currently a 20 Hz envelope with a 10 Hz Nyquist — if EMG is
   dead because it's been filtered to near-nothing, every downstream fusion fix is
   fighting the wrong problem. Requires regenerating the tensor pickle; do this before
   investing in a new fusion architecture.
2. **A2**: per-channel learned-affine normalisation (per-channel z-score, then a
   learnable per-channel gain/bias inside the model/checkpoint). Note plain per-channel
   z-scoring is wrong here — relative cross-channel amplitude *is* the class signal for
   EMG (which muscle fired) — so equalise first, then let the model re-learn the
   weighting it needs.
3. **A1 architecture fix, in increasing order of commitment**:
   - Per-modality stems feeding a shared trunk (cheapest structural change).
   - Modality dropout during training (ModDrop-style — see `READING_LIST.md` §1) +
     per-modality auxiliary heads + a gradient-balancing scheme (OGM-GE) so EMG cannot
     be starved by the optimisation even once it has its own stem.
   - **High-leverage candidate, cross-referenced from the FirstPrinciples doc**: a
     **per-modality Covariance Embedding Network** — compute channel covariance
     separately for the EMG block and the IMU block, embed each with a small learned
     encoder, and fuse in covariance-embedding space rather than raw-signal space. This
     is the FirstPrinciples doc's "Approach 2," and its own stated rationale (`$B`
     succeeds because per-sample channel covariance is the natural discriminative
     feature) is exactly why `$B` already does PCA *per modality* rather than jointly
     (`LIMITATIONS.md` A2's evidence). Doing this per-modality is a structural fix for
     A1 (each modality gets its own covariance statistic, so IMU's 72-channel bulk
     cannot statistically dominate EMG's 16 by construction) **and** a direct attempt at
     D2 (beat `$B` on its own mechanism, made learnable). One architecture, two open
     items. Prototype this before the more generic gated/cross-attention fusion modes if
     time is scarce — it's the candidate most likely to earn its complexity.
4. Report the modality ablation (A1) under L2SO, not just the fixed 24/4/4 split, before
   any claim about fusion beating unimodal goes back in the paper — the current 4-subject
   fixed-split contrast is explicitly underpowered for it.

**Exit criterion:** a fused model that beats IMU-only at 10-way with a CI that excludes
zero, under L2SO.

---

## Phase 1 — the N-way collapse (your priority #2)

Targets `LIMITATIONS.md` A5, C3. D9's ProtoNet numbers (now landed) already show the
readout mechanism, not just capacity, matters at high N: a bare prototype head degrades
with N in the same direction as the MLP head, from a worse starting point — so fixing
*both* the head shape and giving MAML something better than a from-scratch head to adapt
is the right diagnosis, not an either/or.

1. **A5**: prototype-initialised cosine head (ProtoMAML-style, Triantafillou et al. 2020
   — see `READING_LIST.md` §2), replacing the randomly-initialised MLP head MAML
   currently has to fit from N examples in 10 inner steps. Sweep `maml_alpha_init_eval`
   alongside it, since cosine logits at τ≈10 produce ~10× larger head gradients than the
   MLP head's — reusing the old step size will misbehave.
2. **C3**: retrain the headline config at `n_way=10` (currently tuned only at 1-shot
   3-way) and evaluate the resulting 10-unit head at N ∈ {3, 5, 10} from one model, one
   selection rule — not a multi-objective Pareto front, which just relocates the
   arbitrariness reviewers already flagged.
3. Re-run the K-shot sweep (K ∈ {1,3,5}) at 10-way after both fixes land. The current
   flat K-response (67.7 → 64.4 → 68.5) is itself evidence the head, not the shot count,
   is the bottleneck — confirm it moves once the head does.

**Exit criterion:** 10-way 1-shot accuracy materially above 67.7% with a CI, and a
K-shot curve that actually responds to K.

---

## Phase 1.5 — MoE specialisation, and the best current candidate for a *new* method result

Targets `LIMITATIONS.md` A4, C1, C2, C5. Routing is currently 98.2% of maximum flatness
— the MoE's measured benefit (M0 86.78% vs A4 81.75%, single-encoder param-matched
control) is an **ensembling** gap, not a specialisation gap, and D6's Port A result
(support- vs query-conditioned routing: 91.24% vs 91.25%, a null) is consistent with
routing not really discriminating much of anything per-sample.

1. `routing_mutual_information_loss` (the sharpness term the current three aux losses
   lack — they constrain batch marginals only, proven by `tests/test_moe_gating.py`) +
   gate-temperature annealing to τ<1 (currently 1.529, which actively flattens) +
   re-enable `MOE_importance_coeff` (C5) with the caveat that CV² is a third balance
   term, not a sharpness fix, so don't expect it alone to change the picture.
2. **`expert_ablation_curve`** as the causal specialisation diagnostic — if sharpening
   routing raises measured mutual information but costs accuracy, that is a legitimate
   and publishable negative result ("this MoE's benefit in this regime is ensembling,
   not specialisation") that directly answers R2 Q4 / the meta-review Q10. Plan for
   either outcome; don't treat "routing sharpens and helps" as the only acceptable
   result.
3. **Task-conditional routing (`MOE_task_routing_mode="add"`)** is the best current
   candidate for a *new* method result — the thing venue condition 1 (below) actually
   needs. It's a third option distinct from both of MoEMeta's ports (support-derived,
   which Port A already showed is a null here, and frozen-mixture, which Port B already
   showed costs 28.6pts). If task-conditioned routing beats *both* fixed baselines, that
   is a result MoEMeta's precedent does not cover and directly answers ARaC's structural
   objection rather than re-litigating the MAML-vs-ANIL point Port B already settled.
4. **C1/C2**: re-base the expert-count "plateau" claim on the A5 post-hoc sweep (where
   E=8 is actually the maximum, above E=22/E=24) rather than on the HPO search space
   (which started at 20 and could not have found E=8), and fix the A5 HPO's
   expert-count/top-k confound. Needed regardless of the outcome above — it's a factual
   correction, not conditional on anything else landing.

**Exit criterion for the "new method result" claim specifically:** task-conditional
routing beats both the support-derived and frozen-mixture baselines with a CI, framed
explicitly against MoEMeta's two ports as the point of novelty.

---

## Phase 2 — beating the classical baselines (your priority #3)

Targets `LIMITATIONS.md` D2, D9. `$B` (per-sample PCA subspace matching, unsupervised,
near-zero parameters) is believed competitive with or better than EncoderMoE, and per
`LIMITATIONS.md` G.1 this is a **non-negotiable floor**: do not criticise a baseline in
the paper that was never actually measured and, per earlier runs, wins.

1. Implement `A6_pca_knn_baseline.py` for real (currently `raise NotImplementedError`)
   using the already-complete `system/nonparametric/eval_knn_proto.py` — this is wiring,
   not new research.
2. Add the two other strong, cheap, low-tuning-surface classical baselines the
   `READING_LIST.md` §0/§4 flags as the literature's actual answer to "does a simple
   embedding + readout beat meta-learning here": **Riemannian tangent-space + logistic
   regression** on channel covariance (Barachant et al. 2012 — genuinely likely to beat
   `$B` itself, so find that out before a reviewer does) and **ROCKET/MiniRocket**
   (random convolutional kernels + linear classifier, near-SOTA time-series
   classification, effectively hyperparameter-free so it can't be tuned in EncoderMoE's
   favor or against).
3. **Insurance, not a substitute for 1–2**: `HybridProtoEnsembleHead` with a meta-learned
   mixing weight between the neural branch and the nonparametric prototype method. It
   reduces to `$B` when the neural branch is useless and to the pure learned model when
   the mixing weight is zero, so the reported number cannot be worse than the better of
   the two — but report the honest breakdown (both endpoints and the learned mixture),
   not just the ensemble, or this reads as hiding a loss.
4. If Phase 0.5's per-modality Covariance Embedding Network (above) is built, it is the
   natural thing to compare against `$B` head-to-head, since both are covariance-based —
   the paper's cleanest possible story is "a *learned* metric on the same per-modality
   covariance statistic `$B` uses, unsupervised, outperforms `$B`'s fixed L1 metric."
5. **PI sign-off still required** (`LIMITATIONS.md` D2/G.1) on whether/how to report
   `$B` if it does end up winning outright — the non-negotiable floor is report-and-frame
   (e.g., amortisation cost: `$B` requires O(users × classes) storage and no shared
   model), not silence.

**Exit criterion for venue condition 3 (below):** EncoderMoE beats `$B`, Riemannian
tangent-space, and ROCKET with a CI — or the paper's claims are honestly restated to
what the data supports, which (per `VENUE_STRATEGY.md`'s Goal 2 framing) is itself a
publishable finding at the right venue.

---

## Phase 3 — the 4 gates for a main-track ML resubmission

From `VENUE_STRATEGY.md`, restated as a literal checklist against the phases above so
venue viability is legible at a glance rather than requiring a re-read of that document.
**All four are required, not a majority.**

- [ ] **Gate 1 — a method result that's new, not just better.** Phase 1.5's
      task-conditional routing beating both MoEMeta-style ports is the current best
      candidate. Without this, Contribution 1 stays an application contribution (as the
      authors already conceded to ARaC), which is a legitimate paper but not one this
      reviewer pool will originality-score above a 2.
- [ ] **Gate 2 — transformer baseline, fairly built.** `LIMITATIONS.md` D1: param-matched
      preset (`d_model=256, n_blocks=7, patch_len=4` ≈ 5.75M, matching M0's 5,538,216 —
      the shipped TST is ≈197k params, a 28× deficit, and not a fair comparison as-is),
      its own equal HPO budget, `patch_len ∈ {1,2}` in the search space (patch_len=8
      gives a 9-token "transformer," which is not a real test of self-attention), and the
      naive preset reported *alongside* the matched one so neither "we tuned it until it
      lost" nor "we used a straw man" is available as a criticism.
- [ ] **Gate 3 — beats the classical baselines.** Phase 2's exit criterion, above.
- [ ] **Gate 4 — effects exceed variance.** Phase 0's exit criterion, above, applied
      specifically to whichever comparisons carry the paper's headline claims (currently
      the M0-vs-A4 MoE gap and any Phase 1.5 task-routing result).

**Decision checkpoint.** Re-evaluate after Phases 0–2 land. If Gates 1–4 clear: draft for
ICLR first (open review lets the MoEMeta objection be pre-empted in the paper itself
rather than discovered by a reviewer; rebuttal-driven score movement is more common there
per `VENUE_STRATEGY.md`), ICML second, and deprioritize NeurIPS given at least one likely
recurring reviewer and the meta-review's existing (if now-corrected) error in
institutional memory. If they do not clear on the timeline you need: `VENUE_STRATEGY.md`
Goal 1 (IMWUT) is the primary fallback, with TNSRE and TMLR as documented alternates —
that decision table doesn't need to be re-derived, just re-opened.

---

## Explicitly out of scope for this plan

Per `LIMITATIONS.md` D7 (irreducible) and D5 (irreducible): no second dataset, no
cross-session/cross-device data collection, no repair of the Kaifosh electrode-topology
mismatch. These are disclosed limitations, not open work items — don't let them creep
back onto a checklist meant to be finished in 3–6 months.
