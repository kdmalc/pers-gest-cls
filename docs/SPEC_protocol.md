# SPEC — evaluation protocol and honest scope (Opus P2)

*Written 2026-10-21 by Opus. Inputs: Kai's collection facts (one session per participant, sensors never removed, later rep numbers recorded later, reps probably back-to-back within about 1–5 minutes) and Kai's decision against an ordered split. Section 1 needs Kai's confirmation; the rest is ready to use.*

## 1. Headline task — RECOMMENDATION, needs Kai's OK
Make **10-way** the primary task, with K in {1, 3, 5}. **10-way 1-shot** is the single primary cell for the pre-registered comparison family (`SPEC_stats.md`).

Why:
- Every user defined exactly 10 gestures, so a deployed system must choose among all 10. 3-way and 5-way are random subsets of a user's own vocabulary and overstate deployment accuracy.
- Reviewers already flagged the 10-way collapse (67.7% vs 88.4% at 3-way). Leading with 3-way invites the same objection.
- 10-way has only one class set per user (B3). Since labels are now permuted per episode, episodes differ in the support reps and the label assignment. That is enough variety, and the user remains the unit of analysis.

3-way and 5-way stay in the paper as the secondary grid, for continuity with the few-shot literature (5-way is that literature's convention).

Consequence for training: the n-way used during meta-training becomes a hyperparameter selected on dev users at the 10-way 1-shot cell (`SPEC_hpo.md`). It is no longer fixed at 3.

## 2. Protocol statement (paper, Methods)
> We evaluate on 28 of the 32 participants with 7-fold participant-level cross-validation (20 training, 4 validation and 4 test participants per fold; each participant is a test participant exactly once). The remaining 4 participants are used only for hyperparameter selection and are excluded from every reported result. Folds are stratified by participant group (non-disabled Kapanji, disabled Kapanji, disabled non-Kapanji). An N-way K-shot episode uses N of a test participant's gestures in a random label order, K randomly chosen repetitions per gesture as support, and all remaining 10−K repetitions as query, so query size varies with K (9/7/5 per gesture). All methods are evaluated on the identical, pre-generated set of episodes (200 per participant per setting). Accuracy is computed per participant and summarised over the 28 participants with participant-level bootstrap confidence intervals and paired Wilcoxon tests.

## 3. Scope: what the data supports and what it does not
**Supported claims.** Within-session, few-shot calibration of user-defined gestures from pre-segmented repetitions, recorded with a fixed sensor placement.

**Not supported. State these as limitations; do not imply them anywhere:**
1. *Cross-session or cross-day robustness.* There is one session per participant and sensors were never removed, so electrode shift, re-donning, sweat and fatigue across sessions are untested.
2. *Temporal generalisation.* The repetitions of a gesture were recorded within a few minutes, so support and query come from essentially the same conditions. Reported accuracies are within-session calibration accuracies, and should be read as an upper bound on what a user would see after recalibrating on a later day.
3. *Online or streaming recognition.* Every sample is an experimenter-segmented, whole repetition. Gesture onset detection and continuous rejection of non-gestures are not addressed.
4. *Other devices or montages.* There is one sensor layout. The Kaifosh topology mismatch (archived LIMITATIONS D5) shows this is a first-order factor.

Suggested limitations paragraph:
> All data come from a single session per participant with fixed sensor placement, and each gesture's repetitions were recorded within minutes. Our accuracies therefore measure within-session calibration from pre-segmented repetitions; they do not speak to robustness across sessions, re-donning, or days, nor to online segmentation. Few-shot recalibration is precisely the mechanism one would use at the start of each session, but quantifying cross-session degradation requires repeat-session data, which we leave to future work.

The last sentence turns the limitation into motivation without overclaiming. Keep it.

## 4. The "user-defined gestures" claim
Wording depends on the zero-shot result. `SPEC_zeroshot.md` section 4 fixes the rule before any number exists. Until that result exists:
- Use only "participants chose their own movement for each of 10 gesture labels".
- Do **not** use "effectively ~320 distinct gestures".

## 5. Other protocol commitments (already implemented or specified)
- Labels are permuted per episode at train, val and test. There is no cross-user label prior.
- Baselines with a flat global head (A1, A2) re-initialise the head for each episode at evaluation, so they cannot use a label prior either. Paper sentence: "Because label order is randomised per episode, no method can exploit a cross-participant correspondence between label index and gesture."
- Every hyperparameter is selected on the 4 dev participants only. This includes inner steps, learning rates, epochs, training n-way, architecture and expert count.
- Checkpoints are selected on the fold's validation participants using the per-participant mean accuracy.
- Seeds and statistics follow `SPEC_stats.md`.
- The realised query count is reported in every few-shot caption.
- Disclosure for the appendix: earlier development, including the original architecture choice, used participants who are now test participants (audit 1.3). The dev-only architecture check in `SPEC_hpo.md` is the mitigation. Report its result whichever way it goes.

## 6. Still open
- Kai: whether reps were strictly back-to-back. `inspect_rep_structure.py` output is pending. This affects only the wording of limitation 2 ("within minutes" vs "within the session"), not the protocol.
- Kai: confirm the 10-way primary task in section 1.
