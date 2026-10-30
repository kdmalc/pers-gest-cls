# SPEC — per-ability-group analysis of the MoE gain

*Written 2026-10-23, before any group-level number has been seen. Status: module and tests done (`system/analysis/group_analysis.py`, `tests/test_group_analysis.py`); real run waits for the kfold7 reruns. Origin: `10_23_26_ExtraIdeas.txt`, idea 1.*

## Claim being tested
The paper's mechanism: users differ in how they perform gestures (ability level), a mixture of experts can specialise to those differences, so the MoE-vs-no-MoE gain should be larger where users are more heterogeneous. This is a within-dataset test, no extra data needed.

## Pre-registered design
- **Contrast:** per-user accuracy of M0 (EncoderMoE) minus A4 (same MAML network without MoE), on identical cached episodes. Same-hyperparameter and re-tuned A4 are both analysed when both exist (tuned-vs-frozen table).
- **Primary cell:** 10-way 1-shot (`SPEC_protocol.md`). The other 8 cells are consistency checks, labelled exploratory.
- **Groups** (from `kfold28_hpo4_split.json`, tested users only): Non-disabled Kapanji (n=5), Disabled Kapanji (n=9), Disabled Non-Kapanji (n=14). HPO-only users never enter.
- **Representations:** run for `env20` and for the adopted EMG representation.
- **Seeds:** 3 train seeds, averaged per user first (`aggregate.py`); the seed SD of each group mean is reported separately.

## What is reported (no reliance on p-values; n is 5/9/14)
1. Per group: mean gain in points, 95% user-bootstrap CI, Hedges-corrected `d_z`, W/T/L, the no-MoE baseline mean and its SD across users (heterogeneity proxy), and the minimum detectable effect for that group size.
2. Group-vs-group differences in gain with a stratified-bootstrap CI.
3. Group-agnostic check over all 28 users: Spearman between A4 accuracy ("difficulty") and gain, with a bootstrap CI.
4. Fold-cluster bootstrap CI of the overall gain (users in a fold share one model).
5. Optional: expert-routing dispersion per group from `system/MOE/MOE_analysis.py`, as a mechanism-side measure.

## Pre-stated readings (both are reported)
- **Gain concentrates where heterogeneity is highest** (largest SD of A4 accuracy / hardest group): supports the mechanism.
- **Gain is flat across groups** (contrast CIs include 0 with the MDE stated): the motivation story is weakened and the text says so. "No detectable difference (MDE = x points)", never "equal".
- A group whose CI includes 0 is not reported as "MoE does not help that group".

## Not allowed
Choosing the group split, cell, or contrast after seeing results; adding groups; dropping a group for being small.

## Run
`python system/analysis/run_group_analysis.py <results.json ...> --cell 10 1` (after `aggregate`'s checks). A prelim on legacy L2SO per-user results (32 users, groups 6/11/15) is possible only through an adapter, and must be labelled "exploratory: legacy protocol, legacy EMG" (see the open EMG anomaly in `FULL_RUNNING_LOG.md`).
