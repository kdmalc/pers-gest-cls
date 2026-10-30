# CURRENT TODOs
*Short list of open threads only; finished work goes to `docs/LOG.md`. Run tracker: `docs/REBUTTAL_CHECKLIST.md`. Top-level plan: `docs/PLAN_resubmission.md`. Docs index: `docs/README.md`. Architecture work: `docs/specs/SPEC_architecture_exploration.md`. Updated 2026-10-30, evening.*

**Goal:** resubmit EncoderMoE (rejected NeurIPS'26) on one clean protocol: 28 users, 7 folds, cached episodes, 10-way 1-shot primary. Add the reviewer-requested baselines and ablations. **Venue: ICML (Jan) or TMLR, decided after the main runs finish.**

> ## ⚠️ KAI: FIND THE SENSOR-INDEX-TO-BODY-LOCATION MAP
> Find the map from sensor index to body location (all 16 EMG and 12 IMU sensors) and share it with Claude. It is not in the repo. **The body-region expert work cannot start without it** (`docs/analysis/ANALYSIS_moemeta.md` §6).

## 0. What Claude is doing now
- **Done this session (2026-10-30), all tested locally (356 tests pass), NOT yet run on the cluster:** see §3 for the full list.
- **Next code, waiting for Kai's go:**
  1. kfold eval wiring for the screen finalists;
  2. the raw-EMG stem (`raw3s`);
  3. zero-shot Z1-Z4;
  4. upper bound;
  5. `--seed` flag;
  6. A13 at 10-way.

## 1. Decisions for Kai (proposals in `SPEC_architecture_exploration.md` §3)
1. **Tuning order.** Kai's objection is right: M0 and the param-matched controls change if the MoE scheme changes. **Proposal:**
   - tune **A4 and A2 now**: their architecture does not depend on the MoE result;
   - freeze A4's best training HPs and screen every variant with only its own HPs searched;
   - full HPO only for the finalists and their re-matched controls (M0, A1, A4).
2. **Budgets.**
   - Screen: **24 trials** per variant with frozen shared HPs (2-6 searched dims).
   - Finalists: **100 trials**, minimum 60, depending on the measured trial cost.
   - Pick after the debug runs (§2B) give the per-epoch time.
3. **E and top-k:** a **grid** (E ∈ {4, 8, 16, 22, 36} × k ∈ {1, 2, 4, 9}, k ≤ E, 2 seeds) around the best MoE config, not TPE. This gives the E=8-vs-22 figure reviewers asked for.
4. **PI sign-offs:** fs64 for the main table; building td64 (the build command is in `LOG.md`, 2026-10-25; then cluster tokens `A6_td_knn A6_td_lda A6_td_lda_pool`); reporting `$B` if it wins.

## 2. Cluster commands (Kai runs these)
**Every session starts with:**
```bash
cd /projects/my13/kai/meta-pers-gest/pers-gest-cls && git pull && git status --short
L=/projects/my13/kai/meta-pers-gest/pers-gest-cls/system/NOTS/paper/ablations/launchers
cp $L/kfold_launcher.sh $L/hpo_ablation_launcher.sh /scratch/my13/kai/runs/paper/sub2/
cd /scratch/my13/kai/runs/paper/sub2
module purge && module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh && source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba activate /projects/my13/kai/meta-pers-gest/envs/fl-torch
```

**A. Classical baselines (launched 2026-10-30, CPU).** When `squeue --me` shows no `kfold7_A6_*` job:
```bash
for f in /scratch/my13/kai/runs/paper/sub2/eval/logs/kfold7_A6_*.out; do rc=$(grep -o "PYTHON_EXIT_CODE=[0-9]*" "$f"); [[ "$rc" == "PYTHON_EXIT_CODE=0" ]] || echo "${rc:-NO_EXIT_LINE(timeout/killed)}  $f"; done   # empty = all OK
sacct -u km82 -S 2026-10-30 -X --format=JobID,JobName%34,State,Elapsed | grep -E "TIMEOUT|FAILED|OUT_OF_ME"                                          # empty = none timed out
```
Then copy the results to the laptop:
```bash
# on the laptop (Git Bash)
mkdir -p ~/gest_results/A6_cluster && scp -r "km82@<NOTS login host>:/scratch/my13/kai/runs/paper/sub2/eval/kfold7/A6_*" ~/gest_results/A6_cluster/
```
Claude then runs notebook 02 with `A6_RESULTS=~/gest_results/A6_cluster`.

**B. NEEDS CLUSTER TEST: debug trials of the new code, fs64.** These are 15-minute single trials on the `debug` partition, with nothing written to the journals.
```bash
bash hpo_ablation_launcher.sh A4 M0 X2 X3 X4 X6 T2 F1 F2 F3 T0 T4 P1 P1_moe P1_rel P1_feat P3 A4_ANIL --emg-repr fs64 --debug
```
When they finish, paste Claude the output of:
```bash
LOGS=/scratch/my13/kai/runs/paper/sub2/hpo/logs
for f in $(ls -t $LOGS/hpo_*.out | head -18); do echo "== $(basename $f)"; grep -m1 "build_model\]" $f; grep -m2 -E "Epoch completed in|\[metric:.*epoch 1:" $f; grep -m1 -E "Traceback|Error" $f; done
```
- **Pass:** every log shows `[build_model] ... parameters` and at least one epoch line, with no Traceback.
- The epoch times give the trial cost, which sets the budgets in §1.

**C. After Kai decides §1:** stage 0, the full HPO of A4 and A2 (default 60 trials; add `--n-trials 100` if chosen).
```bash
bash hpo_ablation_launcher.sh A4 A2 --emg-repr fs64 --dry-run
bash hpo_ablation_launcher.sh A4 A2 --emg-repr fs64
```
When A4 is done, freeze its training HPs and launch the screen. The exact command list will be given then. Its shape:
```bash
P=/projects/my13/kai/meta-pers-gest/pers-gest-cls/system/NOTS/paper/ablations
python $P/hpo_tools.py freeze --db-dir hpo/optuna_dbs --study ablation_A4_10w1s_kfold7_fs64_v1 --trial best \
  --keys outer_lr,wd,maml_inner_steps,maml_alpha_init,maml_alpha_init_eval,maml_use_lslr,use_lslr_at_eval,use_maml_msl,episodes_per_epoch_train \
  --out /scratch/my13/kai/runs/paper/sub2/hpo/shared_maml_fs64.json
bash hpo_ablation_launcher.sh X2 X3 X4 X5 X6 T2 F1 F2 F3 --emg-repr fs64 --n-trials 24 --study-suffix screen \
  --fixed-params /scratch/my13/kai/runs/paper/sub2/hpo/shared_maml_fs64.json
```

**D. Status, timing and GPU-hours of any HPO, at any time:**
```bash
python /projects/my13/kai/meta-pers-gest/pers-gest-cls/system/NOTS/paper/ablations/hpo_tools.py report --db-dir /scratch/my13/kai/runs/paper/sub2/hpo/optuna_dbs
# per-trial detail for one study:  ... report --db-dir ... --trials ablation_A4_10w1s_kfold7_fs64_v1
# from the logs instead:           grep -h HPO_TRIAL_TIMING /scratch/my13/kai/runs/paper/sub2/hpo/logs/*.out
sacct -u km82 -S 2026-10-29 -X -n -P --format=JobID,ElapsedRaw,AllocTRES | \
  awk -F'|' '{ if (match($3,/gres\/gpu=[0-9]+/)) { n=substr($3,RSTART+9,RLENGTH-9); s+=$2*n } } END {printf "%.1f GPU-hours\n", s/3600}'
```

**E. Housekeeping still open from the 10/29 debug run:**
- run one `--smoke` M0 on fs64: `bash kfold_launcher.sh M0 --folds 0 --debug --hpo-config legacy --emg-repr fs64`;
- delete the non-smoke A1/A2 fold-0 JSONs under `.../eval/kfold7/{A1,A2}/fold0/`.

## 3. Code status
**Tokens** in `ablation_hpo.ARCH_VARIANTS`; each is its own Optuna study.

| Item | Token / file | Local tests | Cluster |
|---|---|---|---|
| X2 uniform gate | `X2` (`MOE_gate_mode`) | ✓ | ☐ §2B |
| X3 sparse top-1/2 + z-loss | `X3` | ✓ | ☐ |
| X4 support-set routing | `X4` (`mamlpp.task_support`) | ✓ | ☐ |
| X5 shared expert | `X5` | ✓ | ☐ |
| X6 cross-attention router | `X6` | ✓ | ☐ |
| F1 mid / F2 FiLM / F3 cross-attention fusion | `F1 F2 F3` (`arch_variants/flexnet.py`) | ✓ | ☐ |
| T0 transformer / T2 CNN experts + transformer / T4 MoE-FFN transformer | `T0 T2 T4` | ✓ | ☐ |
| ProtoNet / Matching / Relation / R2-D2 / FEAT, episodic | `P1 P1_match P1_rel P1_ridge P1_feat`, `P1_moe` (`arch_variants/metric_learners.py`) | ✓ | ☐ |
| ADANN-style (gradient-reversal user head) | `P3` (`arch_variants/adann.py`) | ✓ | ☐ |
| ANIL | `A4_ANIL M0_ANIL` | ✓ | ☐ |
| Routing diagnostics (fixes debug issues 5-7) | `MOE/routing_metrics.py`; stored per MoE trial as user attr `routing` | ✓ | ☐ |
| Per-trial timing + `hpo_tools.py` (report / freeze / grid) | every trial | ✓ | ☐ |
| Frozen shared HPs, study suffix, E×k enqueue | `--fixed-params`, `--study-suffix`, `--enqueue-json` | ✓ | ☐ |
| **Bug fix:** HPO collapse penalty zeroed balanced trials with top_k/E ≳ 0.7 | `_objective_episodic` | ✓ | ☐ |

**Not built yet:**
- 7-fold eval scripts for the new tokens (finalists only);
- EMG-RAW stem;
- F4 (modality-specific expert pools);
- T1/T3;
- CNP (low priority);
- learned `$B` (only if R4 triggers);
- zero-shot;
- upper bound;
- `--seed`;
- A13 at 10-way;
- ports of A7/A8/portA/A16 and Kaifosh to the cache.

## 4. Results so far
**Dev EMG decision (notebook 01):** fs64. At 10-way 1-shot, EMG-only gains +28.9 (proto) / +17.2 (1-NN) points over env20, on 4/4 users.

**28 users, 10-way 1-shot, fused input (notebook 02):**

| method | env20 | fs64 |
|---|---|---|
| kNN-L1 | 58.0 [54.0, 62.1] | **72.9** [69.4, 76.3] |
| ProtoNet (raw features) | 51.1 [46.8, 55.7] | 65.6 [61.5, 69.7] |

- fs64 is better for 28/28 users.
- **Bar for learned models:** about 78% on the dev users (1-NN L1).
- **R4 risk:** old M0 scored 67.7% (old protocol), which is below the untrained kNN.

**Uniqueness (notebook 03):**
- Cross-user zero-shot gets 13-16%, against 62-78% within-user.
- Executions recur across users under different labels.
- The ability-group effect is weak (Mantel r 0.05-0.09).

## 5. Open issues
- **IMU time axis:** fs64 still uses the dict's IMU, which carries the env20 anomaly. Rebuilding the IMU from the raw CSVs costs about a day; decide before the finalist HPO.
- **fs64 standardisation** uses the 4 dev users only. That is fine, but state it in the paper.
- **Rebuttal commitments:**
  - EMGBench EMG-only with gate statistics;
  - the user sweep (A15);
  - the modality ablation;
  - E=8 vs 22 (the §1 grid);
  - citations: Rahimian 2021, Antoniou 2019, Proroković 2020, Raghu (ANIL);
  - remove the NOTS mention;
  - relabel the Côté-Allard / Kaifosh rows.
- **Ability story** is weak at the data level; plan the framing for a null.
- **Venue / second dataset:** decide after the `DECISION_method.md` verdict. ICML needs HPO started by about mid-November.
- **Rules:**
  - no silent fallbacks;
  - never commit `.pkl` files or reviewer/paper PDFs (`docs/papers/NeurIPS-2025-moemeta.pdf` is untracked; leave it out);
  - only Kai launches jobs;
  - refresh the scratch launchers after every `git pull`.
