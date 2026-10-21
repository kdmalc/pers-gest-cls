# SPEC — hyperparameter optimisation (Opus P7)

*Written 2026-10-21 by Opus. Status: ready for Sonnet, but requires (a) Kai's OK on the primary task in `SPEC_protocol.md` section 1, because the HPO objective uses it, and (b) the dev episode file from `SPEC_episode_cache.md`. Observed costs from the old launcher: a MAML+MoE trial takes about 3–4 h; supervised trials about 15 min.*

## 1. Fixed rules
- **Users.** Train on the 20 fold-0 train users; select on the **4 dev users** (P005, P131, P116, P109). Already wired: `ablation_config.get_hpo_split()`.
- **No test user, and no fold val user, is ever seen by HPO.** Assert this in `ablation_hpo.py`: the HPO config's train and val users must be disjoint from every fold's test set, except fold 0's train set, which they are by construction.
- **Objective.** Mean over the 4 dev users of the per-user accuracy on the cached dev episodes at the **primary cell** (proposed: 10-way 1-shot). Report 3-way 1-shot as a secondary user-attr, never as the objective.
  - With 4 users the objective is noisy. That is why the top-3 re-evaluation below exists.
- **Burn-in, early stopping and pruning.**
  - Keep the existing burn-in rule (best val after `HPO_BURNIN_EPOCHS`).
  - Add an Optuna `MedianPruner(n_startup_trials=10, n_warmup_steps=5)` on the per-epoch dev score, to cut cost.
- **Winner's-curse control.** After the trials finish, re-train the **top 3** configurations with 3 seeds each on the same split. Pick the config with the best mean over seeds. Record all three in the study's user-attrs and in `models/hpo_best_models/best_hpo_configs.json`, which is machine-readable and replaces the hand-written `.txt`.
- **Equal budgets** within a family, so no method is tuned harder than its competitors. The old M0 study had 500 trials against 100 for the others; that is not repeated.

| Family | Methods | Trials | Approx. GPU-h |
|---|---|---|---|
| Meta-learned, MoE | M0 (or its successor) | 60 | about 200 |
| Meta-learned, no MoE | A4, T0-MAML, best of T1/T2 (from `SPEC_moe_compute.md`) | 60 each | about 150 each |
| Supervised transfer | A1, A2, T0-supervised, ADANN | 60 each | about 15 each |
| Metric-learned | ProtoNet (episodic), Proto-MAML head | 60 each | about 30–150 |
| Cheap, grid instead of TPE | fine-tune budget (steps x lr) for A1/A2 eval; Riemannian (shrinkage, LR C); MiniRocket (ridge alpha); kNN/`$B` (PCA dims, metric) | full grid | under 5 each |

- Warm start: every TPE study enqueues the method's current default config as trial 0 (Trial 89 for M0). The search starts from what we have, but is not anchored to it.

## 2. Search spaces (additions and changes relative to `ablation_hpo.py`)
- **All episodic neural methods:** add `n_way_train ∈ {3, 5, 10}` (training task shape; the objective is still 10-way 1-shot) and `k_shot_train ∈ {1, 3}`.
- **MAML methods:**
  - Keep the existing MAML space.
  - Add `maml_inner_steps_eval ∈ {5, 10, 25, 50}` as a tuned value. This replaces the old sweeps on test users (audit 1.4).
  - Add `maml_inner_scope ∈ {head, head+last_block, all}` for the transformer variants.
- **MoE:**
  - Widen `num_experts ∈ {4, 8, 16, 22}` (archived C1/C2: the old space started at 20 and could not find small E).
  - Use `MOE_top_k ∈ {1, 2, 4, 8, all}`, filtered to `top_k <= E`.
  - Keep the other MoE knobs.
- **Supervised (A1/A2/T0-sup/ADANN):**
  - Pretraining `lr`, `wd`, `label_smooth`, `dropout`, epochs.
  - The **eval-time fine-tune budget** (`ft_steps ∈ {10, 25, 50, 100, 200}`, `ft_lr` log-uniform 1e-4..1e-1, `ft_mode ∈ {head_only, full}`) is tuned on dev episodes.
  - Report alongside it the MAML-mirrored budget variant (`SOTA_COMPARISONS.md` section 0.3).
- **Transformers (T0 matched):**
  - `d_model ∈ {128, 192, 256}`, `n_blocks ∈ {4..8}`, `n_heads ∈ {4, 8}`, `patch_len ∈ {1, 2, 4}`, `dropout`, `lr`, `wd`, warmup epochs.
  - Trials outside M0's params ±10% raise `optuna.TrialPruned`.
  - The naive preset is not tuned.
- **Not tuned, but fixed and recorded:** seeds, episode files, data representation (`emg_repr`, chosen per `SPEC_2khz_emg.md` E2 using these tuned configs, with a small re-tune of `lr`/`dropout` only).

## 3. Tuned vs frozen hyperparameters for ablations (Kai: "both")
| Ablation | Tuned (own HPO) | Frozen (M0's tuned values, only the ablated part changed) |
|---|---|---|
| A1 No-MAML + MoE | yes | yes |
| A2 No-MAML, no MoE | yes | yes |
| A3 MAML, no MoE (one-expert size) | — | yes |
| A4 MAML, no MoE (param-matched) | yes | yes |
| A5 expert count | — | covered by the widened `num_experts` axis plus the X0–X4 matrix |
| A7/A8 subject-specific | — | yes (training length chosen on dev users) |
| A13 modality | — | yes |
| A15 user count | — | yes |
| portB | — | yes |

**Reporting:**
- The main ablation table shows the frozen version: the classic "remove one component" reading.
- A second column shows the tuned version, so the table answers whether a component's apparent contribution survives giving its absence a fair tuning budget.
- Wording when the two disagree: "the contribution of X is not robust to re-tuning".

## 4. Architecture sanity check (audit finding 1.3)
- **Why.** The DeepCNNLSTM backbone and the MoE encoder placement were chosen in early HPO, which used users who are now test users.
- **Check, on dev users only, with the M0 tuned values:**
  - Backbone ∈ {DeepCNNLSTM (current), MetaCNNLSTM, TST-matched}.
  - MoE placement ∈ {encoder (current), middle}.
  - 1 seed x 3 repeats each.
- **Rule:**
  - If the current choice is within the noise band of the best (overlapping seed ranges), keep it and report the table in the appendix.
  - If another choice is clearly better, switch to it and disclose why.
- Either outcome goes in the appendix.

## 5. Order (cluster, Kai)
1. Generate the dev episode file (`SPEC_episode_cache.md`).
2. Run the cheap families: grids and supervised, about 1 day of wall-clock.
3. Run the meta-learned families in parallel, about 2–3 days of wall-clock at `%10` array concurrency.
4. Do the top-3 x 3-seed re-evaluation.
5. Run the architecture check (section 4) and the X0–X4 MoE matrix (`SPEC_moe_compute.md`) on dev users with the tuned values.
6. Only then run the 7-fold evaluations (`SPEC_stats.md` seeds).

The zero-shot experiment (`SPEC_zeroshot.md`) and the Tier 1 classical baselines need no deep HPO and can run during step 3.

## 6. Implementation notes for Sonnet
- `ablation_hpo.py`:
  - Switch the objective to cached dev episodes and the per-user mean.
  - Add the `n_way_train`/`k_shot_train`/`maml_inner_steps_eval`/`num_experts` changes.
  - Add the transformer and ProtoNet objectives.
  - Add a `--family` flag for the equal-budget launcher.
- Write the export `hpo_export_best.py`: Optuna journal → `best_hpo_configs.json`, with the exact config-key mapping. Some trial parameter names differ from config keys, e.g. `outer_lr` → `learning_rate`, `wd` → `weight_decay` (`ablation_hpo.py:317-318`).
- `make_base_config(ablation_id, hpo_json=PATH)`:
  - Applies the exported config strictly: every key must already exist in the base config, otherwise raise.
  - Records `hpo_source` in the config.
  - Eval scripts get a required `--hpo-config {tuned,frozen}` flag. There is no default, so a run cannot silently use stale Trial-89 values.
