# DECISION — is EncoderMoE still the method? (Opus P9)

*Criteria written 2026-10-21, **before** any matched number exists, so the decision cannot be fitted to the results. The verdict section stays empty until the Tier 1 runs (`SOTA_COMPARISONS.md` section 2) are on the shared episode cache for all 7 folds.*

## Inputs required
- The primary cell (`SPEC_protocol.md`), 28-user per-user accuracies (`SPEC_stats.md`) for:
  - M0, tuned, 3 seeds
  - A4
  - `$B`
  - ProtoNet (best variant)
  - Riemannian (best variant)
  - MiniRocket
  - fine-tune transfer, dev-tuned budget
  - T0-matched
  - zero-shot Z\*
- The X0–X4 MoE matrix on dev users (`SPEC_moe_compute.md`).

## Decision rules (apply in order; the first matching row wins)
| # | Condition (28 users, Holm-corrected paired Wilcoxon, plus mean difference against the MDE) | Decision | Paper consequence |
|---|---|---|---|
| R1 | M0 beats **every** baseline significantly, **and** X1 (gated) beats X2 (uniform mixture) by at least the MDE on dev | Keep EncoderMoE as the method | Current paper, rewritten results. MoE claims are supported by the gate control. |
| R2 | M0 beats every baseline, but X2 ≈ X1 | Keep the architecture, change the claim | Report it as "a meta-learned expert ensemble". Remove specialisation claims and keep the routing analysis as a negative finding. Originality must come from elsewhere (task-conditioned routing X4, if it wins, or the problem/benchmark framing). |
| R3 | A non-MoE learned method (A4, ProtoNet, fine-tune, T0) is statistically indistinguishable from M0, and both beat the classical baselines | Simplify | Lead with the simpler model. MoE becomes an ablation. Framing moves towards the problem, the dataset and the protocol (IMWUT/ASSETS track strengthens). |
| R4 | A classical method (`$B`, Riemannian MDM or tangent space, MiniRocket) is indistinguishable from, or better than, every neural method | Re-plan | Say so plainly. The paper becomes an evaluation and benchmark contribution: "strong classical baselines match deep meta-learning for user-defined gestures". Pursue the learned-`$B` / covariance-embedding line (`system/nonparametric/metric_models.py`) as the method only if it beats the classical winner on dev. |
| R5 | Zero-shot Z\* > 40% and nothing beats it by more than the MDE at 10-way 1-shot | Stop the few-shot claim | Follow `SPEC_zeroshot.md` section 4, row 3. |

## Also decided now
- **Venue mapping.**
  - R1, and R2 only if X4 provides a new-method result: ICML submission (January) is viable.
  - R3/R4: IMWUT or ASSETS first. The benchmark-and-framing paper fits there; ICML reviewers scored originality low last time.
  - R5: IMWUT/ASSETS with an HCI framing ("which gestures transfer, and when personalisation pays").
- **Kai has said he is not wedded to EncoderMoE.** If R3 or R4 fires, this document is the evidence for the PI conversation and must be shown as is.

## Verdict
*(empty until the inputs above exist; fill with the results-JSON paths and the date)*
