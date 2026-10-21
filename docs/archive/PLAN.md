# PLAN.md — phased resubmission plan

**What this is.** The phased implementation plan `LIMITATIONS.md` refers to by name
throughout ("Plan Phase 0.5–0.6", "plan A22/A23", "plan 0.6/0.7", "plan C3") but that
was never actually committed. This file is that plan: it doesn't repeat evidence
`LIMITATIONS.md` already has (linked by ID — A1, D9, etc.), it sequences the fixes into
phases, breaks each phase into commits, and states what each phase is *for*.

**Relationship to other docs.** `LIMITATIONS.md` is the tracker (what's wrong, evidence,
status). `VENUE_STRATEGY.md` is the target (where to send it, and the 4 gates for a
main-track ML resubmission). `READING_LIST.md` is what to read before writing each
piece. This file is the *sequence* connecting them: what to do, in what order, in what
commit, and why that order.

**Status, 2026-09-17 (written), 2026-09-17 (commit structure + MoE efficiency added).**
Decision made with you: chase a top ML venue again — ICLR preferred if the gates in
Phase 3 clear, else ICML or AAAI, deprioritizing a straight NeurIPS resubmission given
at least one likely-recurring reviewer. If Phase 3's gates are not clearing after
Phases 0–2, `VENUE_STRATEGY.md`'s IMWUT/TNSRE/TMLR path is the documented fallback, not
abandoned. You've also flagged that you're not fully committed to the MAML+MoE
architecture itself — the paper is written around it, and there's real organizational
cost to abandoning it (you talked your PI out of a clustering-based approach that
wasn't working), but real downside too if a simpler approach would win once your PI's
dataset goes public and someone else tries it. Phase 0.8 below is the hedge against
that, run in parallel with everything else rather than after it.

---

## How we work — the per-commit template

Every commit below follows the same handoff, referenced as **(→ template)**:

1. I make the code/config change in the repo.
2. I hand you a **debug script** — short, cheap, based on the existing launcher
   conventions (`a16_launcher.sh` already has a smoke-mode default of
   `PARTITION=debug`, 15-minute wall cap — that's the pattern). Where possible it runs
   **locally, off the repo's existing synthetic `tensor_dict` fixture** (`tests/synthetic.py`)
   so you don't need the cluster at all to sanity-check it; where it genuinely needs real
   data it's sized for the cluster's `debug` partition instead.
3. **You run it. Not me — I have no cluster access and won't pretend to.** You report
   pass/fail (and any traceback) back to me.
4. Only once it passes do I hand you the **full evaluation script**, modeled on the
   existing full launchers (`commons` partition, real wall-time budget).
5. **You submit it and report results back.** I update `LIMITATIONS.md`/`PLAN.md` with
   the real number, citing the results artifact — not before there is one, per
   `LIMITATIONS.md`'s own rule ("CLOSED only when there's a committed number").
6. Next commit.

Commits are written to be small and independently revertible — if a debug script fails,
that commit is wrong or incomplete, not the whole phase.

---

## Phase 0 — trust and measurement infrastructure

Nothing built on top of this phase is credible without it; do it first regardless of
which fix looks more exciting.

**Commit 1 — deterministic training RNG.** `canonical_pid_order=True` +
`episodic_train_rng="private"` (B1's root cause: the global `random` module drawing
training subjects over a config-order-dependent PID list). (→ template)

**Commit 2 — paired-stats utility.** `stats_utils.py`: paired bootstrap CIs over 32
subjects, Wilcoxon, Holm correction, Cohen's d_z, a stated minimum detectable effect
(B7). Zero GPU cost — runs on *existing* results JSONs, no cluster needed at all, so
this can land before Commit 1's reruns even finish. (→ template, debug-only, no full
eval script needed)

**Commit 3 — seed/fold rerun harness.** 3 seeds × 32 folds on every headline condition,
emitting a published variance decomposition (`cross_subject_sd`, `seed_sd`, bootstrap
CI) (B1). This is the first genuinely expensive commit in the whole plan — full-eval
script only, no meaningful debug-partition version beyond "does it launch and checkpoint
correctly for 1 fold."

**Commit 4 — macro-F1 + pooled confusion matrix + consistent model selection.** B3/B4/B5:
answers "is the 10-way collapse uniform or three-gestures-merging" directly, and fixes
the query-weighted-vs-unweighted selection/reporting mismatch. (→ template)

**Ongoing, not a numbered commit:** extend `tests/` (`test_fusion.py`,
`test_proto_head.py`, `test_modality_dropout.py`, `test_aug.py`, `test_hybrid_np.py`,
`test_metrics.py`, `test_smoke_train.py`) alongside whichever phase below is active,
same day as the code they cover — the existing 93 tests already caught B13/E8-class bugs
(an undeclared config key, an unbuildable model) before they cost cluster time twice.

**Exit criterion:** a headline number can be reported with a seed/fold-based CI, not a
single run.

---

## Phase 0.5 — fusion mechanism (your priority #1)

Targets `LIMITATIONS.md` A1/A2. A1's modality ablation (landed) makes this concrete: the
fused model doesn't beat its own IMU-only branch at any N, and EMG-only collapses hardest
at 10-way, the deployment target.

**Commit 1 — per-modality learned-affine normalisation.** A2:
`input_norm_mode="learned_affine"` (per-channel z-score, then a learnable per-channel
gain/bias inside the model/checkpoint). Note plain per-channel z-scoring is wrong for
EMG — relative cross-channel amplitude *is* the class signal (which muscle fired) — so
equalise first, then let the model re-learn the weighting. Cheapest lever, do first.
(→ template)

**Commit 2 — per-modality stems.** Split the shared `Conv1d(88→64)` front end into
separate EMG/IMU stems feeding a shared trunk. Structural minimum for A1. (→ template)

**Commit 3 — modality dropout + per-modality auxiliary heads + gradient balancing.**
ModDrop-style dropout (`READING_LIST.md` §1) during training, an auxiliary loss per
modality, and OGM-GE-style gradient balancing so EMG can't be starved by the
optimisation even with its own stem. (→ template)

**Commit 4 — per-modality Covariance Embedding Network (high-leverage candidate).**
Cross-referenced from the FirstPrinciples doc's "Approach 2": compute channel
covariance separately for the EMG block and the IMU block, embed each with a small
learned encoder, fuse in covariance-embedding space. `$B` already does PCA
*per-modality* for exactly this reason (`LIMITATIONS.md` A2 evidence) — this makes that
insight learnable. One architecture, two open items: it's a structural fix for A1 (each
modality gets its own covariance statistic, so IMU's 72 channels can't statistically
dominate EMG's 16 by construction) *and* a direct attempt at D2 (beat `$B` on its own
mechanism). Prototype this even if Commits 2–3 also land — they're not mutually
exclusive, and this is the candidate most likely to earn its complexity. **Double-dips
with Phase 2 Commit 4 below — don't build it twice.** (→ template)

**Commit 5 — report under L2SO.** Re-run the modality ablation under L2SO, not just the
fixed 24/4/4 split, before any fusion-beats-unimodal claim goes back in the paper — the
current 4-subject fixed-split contrast is explicitly underpowered for it. (→ template,
full-eval only)

**Exit criterion:** a fused model that beats IMU-only at 10-way with a CI that excludes
zero, under L2SO.

---

## Phase 0.6 — EMG resolution and MoE compute efficiency (new)

Combined into one phase because they're causally coupled: you cannot run the resolution
experiments without first fixing or at least understanding the memory ceiling. This
phase exists because of something you flagged and I then verified in the repo: **EMG is
natively 2 kHz; the 20 Hz envelope currently used everywhere is a legacy artifact, not a
scientific choice**, and switching toward it would also finally deliver the Kaifosh
matched-featurization comparison the rebuttal promised for camera-ready and never
produced (`LIMITATIONS.md` D4).

**What's already in the repo, confirmed this session — this phase is not starting from
scratch:**
- `system/universal_preprocessing/EMG_preprocessing/build_2khz_tensor_dict.ipynb`
  already builds `segfilt_2khz_emg_tensor_dict.pkl` (T=4,300 @ 2 kHz) for the A10–A12
  ablations.
- `ablation_config.py:255`: `front_end_stride` — a single shared strided `Conv1d`
  placed *before* the context projector and the per-expert CNNs, so the whole MoE bank
  sees an already-downsampled signal rather than raw-length input. Comment: *"Only A12
  sets this to 20 to handle 2kHz data on a 32GB GPU."* This is EMG-only and **not yet
  wired into the fused multimodal M0 path** — almost certainly why a naive extension of
  M0 to 2 kHz OOMs: it's very likely being run without this mechanism, not because the
  mechanism doesn't work.
- The deeper, separate issue confirmed in `system/MOE/MOE_encoder.py`: every expert is
  computed **densely for every sample** regardless of routing
  (`stacked = torch.stack(expert_feats, dim=1)` — the same pattern at lines 710, 949,
  1145, 1337, 1618, 1643 across the model variants), so peak activation memory is
  `O(batch × num_experts × C × T)` independent of `top_k`. `front_end_stride` works
  around this by shrinking `T` before the fan-out, not by fixing the fan-out itself.
  `LIMITATIONS.md` B14 already found why the "obvious" fix — true sparse dispatch,
  compute only the routed experts — is dangerous here: an expert receiving zero samples
  in a batch gets **no gradient at all**, silently (`allow_unused=True` → `None`). So
  this phase treats memory as a real design problem with a fast unblock and a slower,
  correct fix, not a single patch.

**Commit 1 — audit and wire `front_end_stride` for M0.** Confirm whether the fused
multimodal path has this wired at all; if not (expected), wire it, reusing the proven
A12 mechanism rather than inventing a new one. This alone may resolve the OOM you hit.
(→ template)

**Commit 2 — EMG envelope-rate sweep.** {20, 50, 100, 200} Hz at fixed IMU rate (the
original A3 plan), cheap relative to the rest of this phase, run before the 2 kHz
commit so there's a monotonic-or-not curve to compare the 2 kHz result against rather
than a single new data point. (→ template)

**Commit 3 — near-raw 2 kHz EMG, fused with IMU.** Reuse
`segfilt_2khz_emg_tensor_dict.pkl` + Commit 1's `front_end_stride` wiring, fused
multimodal. This commit is the one that also produces the outstanding Kaifosh
matched-featurization number from `LIMITATIONS.md` D4 — plan the eval script to emit
both numbers from one run rather than as separate commits. **Debug script here should
specifically stress-test memory** (smallest batch/model that still exercises the full
`(batch, num_experts, C, T=4300)` path) before you ever submit the full run — the whole
point of this commit is not repeating the OOM. (→ template, debug step is the important
one)

**Commit 4a — fast unblock: request more cluster memory.** For any run that still OOMs
after Commits 1–3, just ask for more memory on the job submission. No shame in it, and
it's the right move for anything that's a one-off rather than a pattern. (Not a code
commit — a note for you when you submit.)

**Commit 4b — real fix, feeds Phase 1.5: chunked/grouped expert evaluation.** Process
experts in groups (e.g. 4–6 at a time), accumulate, instead of stacking all `E` at once.
Bounds peak memory to `O(batch × group_size × C × T)` with **identical forward math** —
nothing about what gets learned changes, only how much is live in memory at once. This
is the safe middle ground between "always dense" (current) and "true sparse dispatch"
(blocked by B14 until that's solved on its own terms in Phase 1.5). (→ template)

**Exit criterion:** the fused M0 model trains at 2 kHz EMG without OOM on the cluster's
standard allocation, and the EMG envelope-rate curve + the 2 kHz point + the Kaifosh
matched-featurization number are all in `LIMITATIONS.md` A3/D4 with committed results.

---

## Phase 0.7 — augmentation (low priority)

Already scoped in `READING_LIST.md` §5. One commit, not expanded further — not one of
your three stated priorities, revisit only if time allows.

**Commit 1 — IMU rotation + jitter/scale augmentation.** Um et al. 2017-style. Blocked
on `LIMITATIONS.md` E7 (verify the 72-channel IMU layout before any rotation
augmentation — a wrong axis layout trains fine and is silently worse). (→ template)

---

## Phase 0.8 — parallel spike: is there a fundamentally better approach? (~10% of effort)

**Runs alongside every phase above, starting now — not after them.** The point is early
warning: you've said you're not fully committed to MAML+MoE, the paper is written
around it partly because it was "close enough" to beat a clustering approach that
wasn't working, and the real risk is sinking the full plan into an architecture that a
simpler method would have beaten once your PI's dataset is public. This phase is sized
to ~2 commits out of the ~20 in this plan — a real budget, not a token gesture, but
explicitly not a majority of the effort either.

**Commit 1 — FirstPrinciples Approach 1: Neural Subspace Classifier.** A learned,
support-conditioned projection matrix, the direct neural analog of `$B`. Cheapest and
smallest commit in the entire plan — a support encoder producing a projection, L1
distance in the projected space, trained on within-user contrastive pairs. (→ template)

**Commit 2 — FirstPrinciples Approach 2: standalone Covariance Embedding Network.**
Note the overlap: Phase 0.5 Commit 4 already builds a *per-modality* version of this
inside the fusion architecture. This commit is the *standalone* version — a single
covariance embedding over the full 88-channel window, evaluated on its own as a
from-scratch classifier rather than as a fusion mechanism inside EncoderMoE — so it's a
fair independent data point, not a rebadged version of the Phase 0.5 result.

**Go/no-go checkpoint (not a commit — a decision point).** Time this to land alongside
Phase 2's classical-baseline results (Phase 2 exit criterion). Compare four things on
equal footing: EncoderMoE-with-Phases-0–1.5-fixes, `$B`/Riemannian/ROCKET (Phase 2), and
these two spike prototypes.
- If EncoderMoE beats all of them: continue as an EncoderMoE paper, the spike was cheap
  insurance that paid off by *not* finding something better.
- If a spike prototype or a classical baseline wins outright: that is the single most
  important finding in the whole plan. The honest move is to say so plainly and re-plan
  around it — not to bury it to protect the sunk cost in the current architecture, and
  not to quietly keep it out of the paper. Tell your PI either way; a paper built on the
  method that actually wins is worth more than defending the one you happened to pick
  first, and this was exactly the scenario you asked this phase to catch.

---

## Phase 1 — the N-way collapse (your priority #2)

Targets `LIMITATIONS.md` A5, C3. D9's ProtoNet numbers (landed) already show the readout
mechanism, not just capacity, matters at high N.

**Commit 1 — prototype-initialised cosine head.** ProtoMAML-style (Triantafillou et al.
2020, `READING_LIST.md` §2), replacing the randomly-initialised MLP head. Sweep
`maml_alpha_init_eval` alongside it — cosine logits at τ≈10 produce ~10× larger head
gradients than the MLP head's, so reusing the old step size will misbehave. (→ template)

**Commit 2 — retrain at n_way=10.** C3: currently tuned only at 1-shot 3-way. One model,
one selection rule, evaluated at N ∈ {3, 5, 10} — not a multi-objective Pareto front,
which just relocates the arbitrariness reviewers already flagged. (→ template)

**Commit 3 — re-sweep K-shot at 10-way.** K ∈ {1, 3, 5}, after Commits 1–2 land. The
current flat K-response (67.7 → 64.4 → 68.5) is itself evidence the head, not the shot
count, was the bottleneck — confirm it actually moves once the head does. (→ template,
full-eval only, cheap relative to Commit 2)

**Exit criterion:** 10-way 1-shot accuracy materially above 67.7% with a CI, and a
K-shot curve that actually responds to K.

---

## Phase 1.5 — MoE, done properly

Targets `LIMITATIONS.md` A4, C1, C2, C5, plus your broader point that MoE here hasn't
been well tuned or ablated at all — routing is currently 98.2% of maximum flatness, and
the measured MoE benefit (M0 86.78% vs the parameter-matched single-encoder A4, 81.75%)
is an ensembling gap, not a specialisation gap.

**Commit 1 — routing sharpness.** `routing_mutual_information_loss` (the sharpness term
the current three aux losses lack — proven by `tests/test_moe_gating.py` to constrain
batch marginals only) + gate-temperature annealing to τ<1 (currently 1.529, which
actively flattens) + re-enable `MOE_importance_coeff` (C5), with the caveat noted in
`LIMITATIONS.md` that CV² is a third balance term, not a sharpness fix. (→ template)

**Commit 2 — `expert_ablation_curve` as a causal diagnostic.** If Commit 1 raises
measured routing MI but costs accuracy, that's a legitimate, publishable negative result
("this MoE's benefit in this regime is ensembling, not specialisation") that directly
answers R2 Q4 / the meta-review Q10 — plan for that outcome explicitly, don't treat
"routing sharpens and helps" as the only acceptable one. (→ template)

**Commit 3 — gating alternatives, actually compared.** Current fixed softmax/temperature
vs. Soft MoE-style soft slot assignment (`READING_LIST.md` §3 — structurally sidesteps
B14's dead-gradient problem rather than working around it) vs. load-balanced top-k with
a proper router z-loss (ST-MoE). Select by measured routing MI *and* accuracy on a held
val set, not by implementation convenience. This is the direct answer to "different ways
of gating weren't tried." (→ template)

**Commit 4 — expert retention/lifecycle, actually tested.** Shared-expert isolation
(DeepSeekMoE-style — the codebase already has an unused `SharedExpert` class,
`READING_LIST.md` §3, currently off) compared against the current all-routed bank. This
is the direct answer to "different ways of handling how experts are retained weren't
tried." Depends on Commit 3's gating choice being settled first (shared-expert isolation
interacts with how the rest of the bank is gated). (→ template)

**Commit 5 — re-base the expert-count claim.** C1/C2: fix the A5 HPO's confound between
expert count and routing density, and rest the "plateau" claim on the corrected A5
post-hoc sweep (where E=8 is the actual maximum) rather than on the HPO search space
(which started at 20 and could not have found E=8). Factual correction — do regardless
of what Commits 1–4 find. (→ template)

**Commit 6 — task-conditional routing.** `MOE_task_routing_mode="add"`, informed by
whichever gating (Commit 3) and retention (Commit 4) choices won rather than bolted onto
the current under-examined default. This is still the best candidate for venue Gate 1's
"new, not just better" result — a third option distinct from both of MoEMeta's ports
(support-derived: Port A already null; frozen-mixture: Port B already costs 28.6pts). If
it beats both fixed baselines, that's a result MoEMeta's precedent doesn't cover.
(→ template)

**Exit criterion for the "new method result" claim specifically:** task-conditional
routing beats both the support-derived and frozen-mixture baselines with a CI, framed
explicitly against MoEMeta's two ports as the point of novelty.

---

## Phase 2 — beating the classical baselines (your priority #3)

Targets `LIMITATIONS.md` D2, D9. `$B` is believed competitive with or better than
EncoderMoE, and per `LIMITATIONS.md` G.1 this is a non-negotiable floor: do not
criticise a baseline in the paper that was never actually measured and, per earlier
runs, wins.

**Commit 1 — implement `A6_pca_knn_baseline.py` for real.** Currently
`raise NotImplementedError`; wire it to the already-complete
`system/nonparametric/eval_knn_proto.py` — this is plumbing, not new research.
(→ template)

**Commit 2 — Riemannian tangent-space + logistic regression.** Barachant et al. 2012 —
genuinely likely to beat `$B` itself on channel covariance, so find that out before a
reviewer does. (→ template)

**Commit 3 — ROCKET/MiniRocket.** Random convolutional kernels + linear classifier,
near-SOTA time-series classification, effectively hyperparameter-free so it can't be
tuned in EncoderMoE's favor or against. (→ template)

**Commit 4 — `HybridProtoEnsembleHead` insurance.** A meta-learned mixing weight between
the neural branch and the nonparametric prototype method, reducing to `$B` when the
neural branch is useless and to the pure learned model at the other end — so the
reported number can't be worse than the better of the two. Report the honest breakdown
(both endpoints and the learned mixture), not just the ensemble, or this reads as hiding
a loss. Not a substitute for Commits 1–3. (→ template)

**Note:** if Phase 0.5 Commit 4 (per-modality Covariance Embedding Network) landed, it's
the natural thing to compare against `$B` head-to-head here — both are covariance-based,
and "a learned metric on the same per-modality covariance statistic `$B` uses,
unsupervised, outperforms `$B`'s fixed L1 metric" is the paper's cleanest possible
story.

**PI sign-off still required** (`LIMITATIONS.md` D2/G.1) on whether/how to report `$B`
if it wins outright — the non-negotiable floor is report-and-frame (amortisation cost:
`$B` needs O(users × classes) storage, no shared model), not silence.

**Exit criterion for venue Gate 3 (below):** EncoderMoE beats `$B`, Riemannian
tangent-space, and ROCKET with a CI — or the paper's claims are honestly restated to
what the data supports (itself a publishable finding at the right venue, per
`VENUE_STRATEGY.md` Goal 2).

---

## Phase 3 — the 4 gates for a main-track ML resubmission

From `VENUE_STRATEGY.md`, restated as a literal checklist against the phases above.
**All four are required, not a majority.**

- [ ] **Gate 1 — a method result that's new, not just better.** Phase 1.5 Commit 6
      (task-conditional routing beating both MoEMeta-style ports) is the current best
      candidate. Without this, Contribution 1 stays an application contribution, which
      is a legitimate paper but not one this reviewer pool will originality-score above
      a 2.
- [ ] **Gate 2 — transformer baseline, fairly built.** `LIMITATIONS.md` D1: param-matched
      preset (`d_model=256, n_blocks=7, patch_len=4` ≈ 5.75M, matching M0's 5,538,216 —
      the shipped TST is ≈197k params, a 28× deficit), its own equal HPO budget,
      `patch_len ∈ {1,2}` in the search space, and the naive preset reported *alongside*
      the matched one.
- [ ] **Gate 3 — beats the classical baselines.** Phase 2's exit criterion.
- [ ] **Gate 4 — effects exceed variance.** Phase 0's exit criterion, applied to whichever
      comparisons carry the paper's headline claims.

**Venue order if all four clear:** ICLR first (open review lets the MoEMeta objection be
pre-empted in the paper itself; rebuttal-driven score movement is more common there per
`VENUE_STRATEGY.md`). **ICML and AAAI both worth a look second** — AAAI added at your
request; note its review culture runs closer to ICML/NeurIPS's closed process than to
ICLR's open review, so the same pre-emption advantage doesn't apply there, but its
scope (broad AI, not ML-only) and deadline cadence may fit the timeline better than
ICML's. NeurIPS stays deprioritized given at least one likely-recurring reviewer and the
meta-review's existing (if now-corrected) error in institutional memory. Pick between
ICML and AAAI once Phase 3 is actually reached and real deadlines are in hand — no need
to decide that now.

**If Gates 1–4 do not clear on the timeline you need:** `VENUE_STRATEGY.md` Goal 1
(IMWUT) is the primary fallback, with TNSRE and TMLR as documented alternates — that
decision table doesn't need to be re-derived, just re-opened.

---

## Explicitly out of scope for this plan

Per `LIMITATIONS.md` D7 (irreducible) and D5 (irreducible): no second dataset, no
cross-session/cross-device data collection, no repair of the Kaifosh electrode-topology
mismatch. These are disclosed limitations, not open work items — don't let them creep
back onto a checklist meant to be finished in 3–6 months.
