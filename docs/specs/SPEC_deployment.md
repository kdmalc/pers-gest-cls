# SPEC — offline vs real-time (deployment) discussion

*Written 2026-10-23. Origin: peer feedback that reviewers at the target journals will ask for classical baselines and an offline-vs-real-time discussion. Facts below are stated from the current pipeline; measurements are TODO on the final model.*

## What the current system is (state this plainly in the paper)
- **Offline, segment-level classification.** Each input is a pre-segmented gesture trial (median 3.2 s, tightly cut, 0.8-12.2 s overall). There is no gesture-onset detection and no continuous stream in the evaluation.
- **Non-causal preprocessing.** EMG-FS64 uses a zero-phase band-pass (`sosfiltfilt`), window lengths up to 2 x trial_len / 64, and trial-level `$B` block normalisation (a statistic of the whole trial). The 20 Hz envelope path is likewise computed offline. A deployed system needs causal filtering and running normalisation.
- **Test-time adaptation.** MAML-style methods take gradient steps on the K support samples of each new user, so deployment needs on-device (or server-side) fine-tuning, plus a short enrolment session per gesture.
- **Recording hardware.** Research-grade 2048 Hz, 16-channel sEMG plus 12 IMUs. This is not a consumer-device configuration (see `SPEC_2khz_emg.md` §D3 for the rate policy).

## Measurements to collect on the final model (TODO after reruns)
| Quantity | How | Existing tool |
|---|---|---|
| Parameters (total / active per sample for the MoE) | count; top-k experts active | `V7_checkpoint_param_count.py` |
| Adaptation time per episode (inner steps), CPU and GPU | wall clock over cached episodes | new, small |
| Inference time per sample | wall clock, batch 1 | new, small |
| Peak memory | | peak-memory script (S4) |
| Window needed before a prediction | trial length distribution vs a causal-window ablation | optional |
| Classical-baseline latency for contrast (`$B`, LDA) | wall clock | `A6_classical_baselines.py` |

## Optional sensitivity check
Rebuild EMG-FS64 on the dev users with a causal filter (`sosfilt`) and a running (not whole-trial) normalisation, then compare M0/A4 accuracy. This quantifies how much of the result depends on offline processing. Run only if a reviewer-facing claim about real-time use is made.

## What the paper may and may not claim
- May: accuracy of few-shot personalisation on segmented trials with research-grade sensing.
- May not: real-time or streaming performance, or consumer-device performance, unless the sensitivity check and latency table above support it.
- Put a short "offline vs real-time" paragraph in the Discussion and the limitations section; include the latency table in the appendix.
