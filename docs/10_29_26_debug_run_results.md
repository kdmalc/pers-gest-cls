# 10/29/26 — Cluster debug run (kfold7 smoke evals + 1-trial HPO debug)

New standalone file on purpose (another Claude instance may be editing CURRENT_TODOs / FULL_RUNNING_LOG / notebooks).
Fold these findings into those files later.

Code: `bd559d9-dirty` on NOTS (cluster working copy is dirty — find out why before real runs).
Hardware: L40S. Logs: `/scratch/my13/kai/runs/paper/sub2/{eval,hpo}/logs`.

## 1. What ran

| Job | What | rc | Wall |
|---|---|---|---|
| 2465696 | M0 kfold7 fold0 `--smoke` | 0 | ~1 min |
| 2465697 | A1 kfold7 fold0 | 0 | ~4 min |
| 2465698 | A2 kfold7 fold0 | 0 | ~1.3 min |
| 2465699 | A4 kfold7 fold0 `--smoke` | 0 | ~0.5 min |
| 2465700 | A1 HPO, 1 trial, debug-single, InMemory Optuna | ok | 206 s trial |

Smoke accuracies (M0 11.4%, A4 10.0%, A1 head 16.7% / full-ft 29.5%, A2 13.3% / 26.5%) are chance-level or 1-epoch
numbers — **not meaningful**. All hyperparameters were the LEGACY Trial-89 set. The only thing this validates is plumbing.

## 2. Plumbing: what the run confirms works
- Episode cache loads (`system/episodes/kfold7/val_fold0.json.gz`, sha256 logged), 4 test users / 40 test episodes, test source hash `9e436bd07c17`.
- Smoke flag is written to results JSON (`"smoke": "True"`) so the aggregator can reject it.
- Checkpoints + summary JSONs land in the expected `kfold7/<abl>/fold0/` dirs; HPO study name `ablation_A1_10w1s_kfold7_v1`.
- Whole HPO trial (pretrain 16 epochs + head-only FT eval) is ~3.5 min on one L40S. 100-trial studies are feasible.

## 3. Issues found (ranked)

**Needs a decision / check before real runs**
1. **Unclear whether A1/A2 evals were smoke.** Their grepped logs have no `SCRIPT:`/`SMOKE RUN` line (M0/A4 do). Check the full logs; if they were *not* smoke, 1 fold at 26–29% with legacy HPs is just a weak/untuned model, not a bug.
2. **HPO val split looks like the legacy one.** Config shows `user_split_json_filepath=hpo_strat_kapanji_split.json`, `hpo_val_PIDs=[P005,P109,P116,P131]` (not in `all_PIDs`, which has 28), and the log prints "legacy kapanji fold-0 constants loaded". Pretrain loader reports 20 train PIDs / 4 val PIDs. Need to confirm HPO is really using kfold7 **dev** users/episodes (`val_episode_source: dev`) and that HPO-val users are disjoint from every kfold7 test fold. (SPEC_hpo.md / SPEC_protocol.md are the reference.)
3. **Config/log mismatches for the same run**: earlystopping_patience=8 in config but log says `patience=12`; `gradient_clip_max_norm=10.0` in config but `GradClip=1.0` printed. One of the two is not being honoured — find which value is actually applied.
4. **Heavy overfit in the supervised pretrain stage**: train acc → 97% by ep16 while val acc stays ~20% (chance 10%); best epoch = 4 (selected on val *loss*; val acc peaked later ~21.8% at ep5). Head-only val after FT = 46.8% vs pretrain-only 21.8%, so the HPO objective is dominated by the FT hyperparameters (`ft_lr=0.0226`, 50 steps), not the encoder. Consider whether the search ranges for `ft_lr` / `label_smooth` are sensible. Sampled `label_smooth=0.42` is very large — probably should cap the range (~0.0–0.2).

**Diagnostics are misleading**
5. **MoE entropy diagnostic is saturated.** Mean entropy 1.384 nats ≈ ln(4)=1.386 → with top_k=4 the nonzero gate weights are ~uniform, so normalised entropy 0.386 ≈ ln4/ln36 regardless of what the gate learns. Std ~0.002–0.02. Report entropy over the *pre-top-k* softmax or the selection-frequency entropy instead, otherwise it can't distinguish collapse from specialisation.
6. **Imbalance ratios are div-by-~0 artifacts** (e.g. 632500032x, 232500016x) from dead experts. Should print `inf`/`n/a` or use eps-free metrics (e.g. fraction of experts with selection_freq < 1%, CV of load). Also `np.corrcoef` divide warnings = zero-variance (dead) experts.
7. **"Routing by Participant" prints `?`** — PIDs aren't passed to the routing analysis in the supervised path, so the per-user routing claim (central to the ability-cluster story) can't be read from these logs yet.
8. Qualitatively, routing is still concentrated: experts 10/5/24 dominate in every checkpoint, many experts have ~0 dominant freq, and gesture→expert mapping is unstable across epochs. Only one trial with legacy HPs, so not a conclusion — but worth checking aux/importance coeffs (this trial sampled `MOE_importance_coeff=0.0`).

**Cosmetic / infra**
9. Mojibake in FINAL lines (`Â±`) — console encoding vs UTF-8 `±`; set `PYTHONIOENCODING=utf-8` or use `+/-`.
10. M0 log has no `FINAL` line (A1/A2/A4 do) — M0 script missing the summary print.
11. HPO sbatch header prints literal `$(hostname)`, `$(date)`, `${SLURM_JOB_ID}` (and `JOB_END date=$(date)`) — single-quoted echo/heredoc in the debug-single launcher.
12. Each HPO trial writes a ~35 MB `trial_N_foldK_best.pt`. At 100+ trials × folds that's a lot of scratch; keep only the best trial's checkpoint (or none; the final eval retrains).
13. Cluster tree reports `-dirty`: make sure the launchers' "refuse stale copy" check also flags uncommitted edits for *real* runs.

## 3b. Resolutions (same day)
- **Issue 2 (split): not a bug.** `ablation_hpo.apply_fold_to_config` uses `get_hpo_split()` = kfold7 fold-0's 20 train users, validated on the 4 HPO-val users, test=[]. The Kapanji JSON is only loaded (unused `ALL_SPLITS`), and the "legacy constants loaded" line is just a print. HPO-val users are outside `all_PIDs`, so they are disjoint from every test fold by construction. Config is all in-repo.
- **Issue 3 (patience/clip): fixed.** `pretrain_trainer.pretrain` only read `es_patience`/`grad_clip`/`use_early_stopping`/`es_min_delta`; the ablation config sets `earlystopping_*`/`gradient_clip_max_norm`, so defaults (12, 1.0, 1e-4) silently applied. Trainer now falls back to the config-style keys (explicit `es_*`/`grad_clip` still win, so other callers are unchanged). Effect for A1/A2/HPO: patience 12->8, clip 1.0->10.0, min_delta 1e-4->5e-3. **Any earlier A1/A2 HPO results ran with the old values.**
- **Issue 13 (dirty tree): benign.** Deleted OLD_* launchers, a tracked `system/__pycache__/__init__.cpython-310.pyc`, untracked `maml_res/` and a slurm .out. Should `git rm --cached` the tracked .pyc and gitignore `__pycache__`.

## 4. Plan

**Tonight (cluster only, no laptop compute)** — optional, only if trivial:
- `cat` the full A1/A2 logs to settle issue 1; `git -C /projects/my13/kai/meta-pers-gest/pers-gest-cls status --short` to see what's dirty (issue 13).
- Smoke the remaining ablations (those not yet smoked) with the same `--smoke --fold-idx 0` command if the launcher makes that a one-liner.

**Tomorrow, in order**
1. Resolve issues 2 and 3 (HPO split provenance; patience/clip actually applied). These can invalidate HPO results, so do them before launching anything expensive.
2. Fix cheap logging bugs 5–7, 9–11 (one small commit; no effect on results). Keep routing-by-participant fix (7) since it feeds the paper analysis.
3. Tighten HPO search space (label_smooth cap; sanity range for ft_lr/ft_steps; decide on importance-coeff floor), update SPEC_hpo.md.
4. Add checkpoint pruning (12), commit, sync to cluster so `GIT_REV` is clean.
5. Launch real HPO (per SPEC_hpo.md), then kfold7 evals with tuned HPs for M0 + ablations; classical baselines remain on the laptop.

## 5. Open questions for Kai
- Were A1/A2 run without `--smoke` on purpose?
- Is the HPO val set intentionally 4 separate users outside the 28 (P005, P109, P116, P131)? If so, they must never appear in a kfold7 test fold.
