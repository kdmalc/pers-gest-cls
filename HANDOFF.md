# HANDOFF.md — pick-up notes for the EncoderMoE resubmission effort

**Written 2026-09-23, end of session, because Kai needs to restart his laptop.**
Read this first when you come back — it's the fast way back to full context
without re-reading the whole chat history.

## Executive summary

`EncoderMoE` (NeurIPS 2026 submission 31103) is effectively rejected — 4
reviewers, originality scored 3/2/2/1, structural objection (MoEMeta
precedent). Goal: turn this into a resubmission to a top ML venue
(ICLR preferred, ICML/AAAI acceptable, NeurIPS deprioritized) within
3–6 months. Two sessions so far have been **pure planning/docs work — no
code has been changed yet.** Everything lives on a local branch,
`docs/resubmission-plan-and-tests`, checked out in the repo right now.
That branch has 2 local commits **not pushed to origin** (origin only has
the 1 commit a prior session made before this chat started). `main` has not
been touched at all.

**Where to look, in order:** `PLAN.md` (the actual phased, commit-by-commit
execution plan — start here), `LIMITATIONS.md` (why each fix is needed, with
file:line evidence and rebuttal numbers), `VENUE_STRATEGY.md` (where to
submit and the 4 gates for a main-track ML venue), `READING_LIST.md` (papers
to read per problem). All four are at the repo root on this branch.

## State right now

- **Branch:** `docs/resubmission-plan-and-tests`, currently checked out.
  Local commits on top of `main`'s `8d20376`: `d3158d0` (synced
  `LIMITATIONS.md` with final rebuttal numbers, added `PLAN.md`), then
  `380f692` (restructured `PLAN.md` into numbered commits + added the
  EMG-2kHz/MoE-efficiency phase + the 10% hedge spike + AAAI). **Neither is
  pushed.** `origin/docs/resubmission-plan-and-tests` is one commit behind
  (only has the prior session's original `d9de5af`).
- **`main` is untouched** the whole time — verified after every commit
  (`git rev-parse main origin/main` both still `8d20376`).
- **No code has been written or changed.** Both sessions were entirely
  documentation/planning. The very next thing to happen, whenever you
  resume, is the *first real commit* — see "Next step" below.
- One pre-existing untracked directory, `system/stats/` (a notebook, not
  mine, was untracked before this chat started) — left alone throughout,
  not part of any commit.

## What happened, in order (for context, not action)

1. Read both PDFs off Kai's OneDrive Desktop (`Claude_FirstPrinciples_Aim2_Approach.pdf`,
   a pre-rebuttal strategy doc with 3 proposed neural approaches built on the
   `$B` PCA baseline; `OpenReview_downloaded.pdf`, the full closed rebuttal
   thread for the actual submission) and the entire pre-existing
   `docs/resubmission-plan-and-tests` branch (a prior session's `LIMITATIONS.md`/`READING_LIST.md`/`VENUE_STRATEGY.md`/`tests/`).
2. Synced `LIMITATIONS.md` with the rebuttal's final posted numbers it
   predated (modality ablation — IMU dominates, EMG collapses at 10-way;
   ProtoNet baseline; MoEMeta Port A/B; corrected Kaifosh numbers; new items
   for the meta-training user sweep and the still-missing A16 upper bound),
   fixed a reviewer-score transcription error, added a §H on what actually
   moved each reviewer at close. Wrote `PLAN.md` from scratch (referenced
   everywhere in `LIMITATIONS.md`, never previously committed). Committed as
   `d3158d0`.
3. Kai added: consider AAAI too; EMG is native 2 kHz and the 20 Hz envelope
   used everywhere is a legacy artifact; he hit an OOM trying to extend the
   model to 2 kHz. Investigation found **`front_end_stride`**
   (`ablation_config.py:255`) already exists and already solves this for the
   EMG-only A12 ablation ("sets this to 20 to handle 2kHz data on a 32GB
   GPU") but isn't wired into the fused multimodal M0 path — and separately
   found the real mechanism behind the OOM: `MOE_encoder.py` computes every
   expert densely for every sample regardless of routing (`torch.stack` over
   all `E` experts, 6 call sites), so memory scales with
   `num_experts × T` regardless of `top_k`. Kai also said MoE here has never
   been properly ablated (expert count, gating variants, expert retention),
   and that he's not fully committed to the MAML+MoE architecture itself —
   he talked his PI out of a clustering approach that wasn't working by
   arguing MoE was "close enough," so there's real cost to abandoning it, but
   real risk too if something simpler would win once the dataset goes public.
   He asked for ~10% of effort as a parallel hedge, and for `PLAN.md` to be
   broken into explicit commits with a stated workflow: Claude writes code +
   a debug script (local synthetic data, or the cluster's `debug`
   partition), Kai runs it and reports pass/fail, only then does Claude hand
   over a full evaluation script (cluster `commons` partition) for Kai to
   submit and report results from — **Kai is the only one who ever touches
   the cluster.** Rewrote `PLAN.md` completely to reflect all of this
   (new "How we work" section, new Phase 0.6 for EMG resolution/MoE memory,
   expanded Phase 1.5 for MoE gating/retention, new Phase 0.8 for the hedge
   spike, AAAI in Phase 3). Committed as `380f692`.
4. Session paused here (Kai needs to restart his laptop) — this file is that
   pause point.

## Next step, whenever you resume

Read `PLAN.md`'s "How we work" section and Phase 0 / Phase 0.6, then ask Kai
which he wants to start with:
- **Phase 0 Commit 1** (deterministic training RNG) — cleanest starting
  point since every later number's credibility depends on it, but not urgent
  on its own.
- **Phase 0.6 Commit 1** (audit + wire `front_end_stride` for the fused M0
  path) — unblocks the 2 kHz EMG work Kai was already trying to do when this
  came up, likely the thing he's most eager to see land.

Either way, the first deliverable is a **debug script**, not a full
implementation — follow the template at the top of `PLAN.md` exactly.

## Don't re-derive these — they took real effort to establish

- `front_end_stride` (`ablation_config.py:255`) is the existing fix for
  2 kHz EMG memory, proven on A12, not yet on M0.
- The OOM's actual cause is dense per-expert compute in `MOE_encoder.py`
  (`torch.stack(expert_feats, dim=1)`), not just "T got bigger" — this
  matters for Phase 1.5 too, not just Phase 0.6.
- `segfilt_2khz_emg_tensor_dict.pkl` (built by
  `build_2khz_tensor_dict.ipynb`) already exists for A10–A12; no need to
  regenerate it for EMG-only 2 kHz work.
- Reviewer originality scores were **3/2/2/1** (Lpuc/xh89/ARaC/fkxS), not the
  "2/2/1" the prior session's `VENUE_STRATEGY.md` originally had — already
  corrected in both `LIMITATIONS.md` and `VENUE_STRATEGY.md`, but worth
  knowing so it doesn't get miscited in the paper itself.
- Data paths: the two source PDFs are under Kai's **OneDrive-redirected**
  Desktop (`C:\Users\kdmen\OneDrive\Desktop\...`), not the bare
  `C:\Users\kdmen\Desktop` (which only has a Box shortcut) — if anything
  else needs to be found on Desktop later, check the OneDrive path first.
- `origin/docs/resubmission-plan-and-tests` is stale relative to local — if
  you ever want this pushed, it needs an explicit `git push`, and Kai should
  decide when (he said not to push yet, as of this session).
