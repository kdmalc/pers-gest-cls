# SPEC — Gate 0: the laptop pre-screen for a personalisation idea

*Written 2026-10-30 (Fable). Code: `system/nonparametric/gate0.py`, tests `tests/test_gate0.py`. Generalises the body-region Gate 0 (`SPEC_body_regions.md` §3-4, `nonparametric/region_headroom.py`), which killed a bad idea in a day instead of a month of GPU time.*

## 1. What it is for
Any originality candidate that amounts to "personalisation = choosing or weighting *which signals* a person's classifier uses" can be tested without training. The body-region line was one such candidate (fixed anatomical groups). Others: per-user channel selection, modality weighting, data-driven sensor groups, calibration-time channel pruning for a wearable rig.

The harness runs on the 4 dev users only (`P005, P131, P116, P109`), on the cached 10-way 1-shot episodes, with balanced 1-NN L1. The 28 test users stay clean.

## 2. What a candidate is
A function `candidate(view, others, **kw) -> weights` giving a non-negative weight per channel (16 EMG, 12 IMU sensors), either one vector or one per episode. The distance is additive over channels, so weighting is exact and instantaneous.

- `view` holds **support-only** data for the evaluated user (support-support distances and support labels). The query side is not reachable: the body-region "oracle" looked good because it picked regions on query data, and the honest support-only selector then failed. The harness makes that mistake impossible rather than discouraged.
- `others` holds the other dev users' full bundles, as training data for reference statistics.

## 3. The rules (pre-stated; the same as the body-region gate)
| Rule | Meaning |
|---|---|
| **H1 gain** | mean gain over the uniform baseline ≥ 2 points **and** ≥ 3 of 4 dev users improve |
| **H2 beats null** | the candidate beats the 95th percentile of its own permutation null (200 draws) |

The null permutes the candidate's weights across channels per episode. It keeps the sparsity and scale exactly, so it asks only whether *which* channels were chosen matters. **Pass = H1 and H2.** A pass earns a deep screen (P1 / M0 variant through the normal HPO screen). A fail closes the idea at the data level, as it did for body regions.

Baseline reference: uniform 1-NN L1 on fs64 is about 78% on the dev users (notebook 01).

## 4. What it cannot test
Anything that needs a trained encoder: new routing schemes, new losses, adaptation rules, fusion. For those the first gate is the architecture screen itself (`SPEC_architecture_exploration.md` §3), at ≥ 24 trials with frozen shared HPs.

## 5. Running it
```bash
# built-in examples (uniform must reproduce the baseline exactly; it is the sanity check)
python system/nonparametric/gate0.py --candidate uniform --out /tmp/g0_uniform.json
python system/nonparametric/gate0.py --candidate support_spread:m=14 --out /tmp/g0_spread.json
# your own idea: a file with  def weights(view, others): ...
python system/nonparametric/gate0.py --candidate-path my_idea.py:weights --out /tmp/g0_mine.json
```
Needs `dataset/meta-learning-sup-que-ds/segfilt_fs64_tensor_dict.pkl` (laptop with the data, or the cluster login node).

## 6. Log of candidates
| Date | Candidate | Gain | vs null | Verdict |
|---|---|---|---|---|
| 2026-10-30 | Body-region anatomical groups, CV oracle (`region_headroom.py`) | +1 to +3 (oracle) | 94th pct (fail at 95) | closed |
| 2026-10-30 | Support-spread selector (regions) | negative | — | closed |
