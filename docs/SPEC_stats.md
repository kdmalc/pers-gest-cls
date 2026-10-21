# SPEC — statistics, seeds and reporting (Opus P8)

*Written 2026-10-21 by Opus. Status: ready for Sonnet. Applies to every reported few-shot number. Inputs are the per-episode results defined in `SPEC_episode_cache.md`.*

## Unit of analysis: the user
- **Per-user accuracy.** For each (method, cell, train seed), a user's accuracy is `sum(n_correct) / sum(n_query)` over that user's episodes in the cell. All episodes in a cell have the same Q, so this equals the mean of the episode accuracies.
- **Several train seeds.** Average the per-user accuracy over seeds first. That gives one number per user, 28 per method per cell.
- **Never treat episodes as independent.** No standard error, CI or test is computed over episodes. With 10 reps, the 200 episodes of one user reuse the same 100 samples.
- **Selection metric.** Checkpoint and hyperparameter selection also use the per-user mean (on val or dev users), not accuracy pooled over queries.

## What every table reports (per method, per cell)
- The mean over the 28 users.
- A 95% bootstrap CI: resample users with replacement, 10,000 resamples, percentile interval.
- The SD across users.
- The chance level (1/N), in the caption or a column.
- In the appendix: the 7 fold means. When train seeds exist, the SD of the 28-user mean across seeds, reported separately from the user CI.

## Comparisons
- **Method A vs method B:**
  - Paired per-user differences `d_u = acc_A(u) - acc_B(u)`.
  - Two-sided Wilcoxon signed-rank test, using `scipy.stats.wilcoxon(..., zero_method="pratt")`.
  - Report the mean difference with a 95% user-bootstrap CI, `d_z = mean(d)/sd(d)`, and wins/ties/losses out of 28.
- **Primary family:** pre-registered before any number is seen.
  - The primary cell is the one fixed in `SPEC_protocol.md`. In it, the main method is compared against each of: `$B`, ProtoNet, Riemannian tangent-space, MiniRocket, A4 (MAML without MoE), the best transformer, and the zero-shot cross-user model.
  - Holm correction is applied over this family only.
  - The family list is written into `system/analysis/primary_family.json` and committed before the full runs.
- **Secondary:** the other grid cells and the ablations. Report Holm-corrected p-values within each table and label them exploratory in the text.

## Dependence caveat (state it in the paper)
Users tested in the same fold share one trained model, so the 28 users are not fully independent.
- **Primary analysis:** user-level, as above.
- **Sensitivity check:** a cluster bootstrap that resamples the 7 folds with replacement and keeps their users. Report whether any primary conclusion changes. If one does, the paper reports the fold-level result as primary for that comparison.

## Seeds
| Methods | Train seeds |
|---|---|
| Main method (M0 or its successor per `DECISION_method.md`), A4, best transformer, best learned baseline | 3 (0, 1, 2) on all 7 folds |
| `$B`, kNN, Riemannian, MiniRocket | 1. These are deterministic given the episodes; MiniRocket's kernel seed is fixed. |
| Remaining ablations (A1, A2, A3, A7, A8, A13, A15, portB) | 1. Stated in every caption. |

- The episode cache is the same for all seeds, which keeps comparisons paired.
- Sensitivity check for the primary cell only: one extra regeneration of the test episodes with `BASE_SEED + 1`, with the main method and the best baseline re-evaluated on it.

## Power, so claims are sized correctly
- With n = 28 paired users and alpha = 0.05 two-sided, 80% power needs `d_z` of about 0.55 (paired t; Wilcoxon has about 95% efficiency). The first Holm step in a 7-comparison family uses alpha = 0.0071 and needs `d_z` of about 0.75.
- `stats_utils.mde(sd_diff, n=28, alpha, power)` converts this to points once the SD of the per-user differences is known. The paper states the minimum detectable effect next to every null result.
- A non-significant difference is reported as "no detectable difference (MDE = x points)", never as "equal".

## Tooling (Sonnet)
- **`system/analysis/stats_utils.py`** (numpy and scipy only):
  - `per_user_acc(per_episode_records)`
  - `bootstrap_ci(values, n=10000, seed=0)`
  - `paired_compare(a, b)`, returning mean diff, CI, Wilcoxon p, d_z and W/T/L
  - `holm(pvals)`
  - `cluster_bootstrap(per_user, fold_of_user)`
  - `mde(...)`
- **`system/analysis/aggregate.py`:**
  - Reads results JSONs.
  - Enforces the episode-hash and id-set equality from the episode-cache spec.
  - Builds the per-user table (CSV) and the paper tables.
  - Raises on any mismatch, missing fold or missing user.
- **Tests:** `tests/test_stats_utils.py`, on synthetic data with known answers. Cases: a Holm ordering example, bootstrap CI coverage on a normal sample, Wilcoxon equal to scipy, and the aggregator raising on a hash mismatch.
- Do not put this in `system/stats/`. That folder holds Kai's untracked notebook.
