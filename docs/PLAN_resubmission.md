# PLAN — EncoderMoE resubmission (reviews → experiments → venue)

*Written 2026-10-29 (Opus). This is the top-level plan. It links to the detailed docs and does not repeat them. Status board: `HIGH_LEVEL_STATUS.md`. Kai's immediate to-dos: `../CURRENT_TODOs.md` / `KAI_TODO.md`. History and decisions: `FULL_RUNNING_LOG.md`. Go/no-go rules: `DECISION_method.md`. `archive/PLAN.md` is the older September plan and is superseded by this file.*

---

## 1. What the reviewers said (NeurIPS 2026, submission 31103, rejected)

All four reviewers gave **Significance 2**. Originality was 3 / 2 / 2 / 1. Quality and clarity were fine. The core complaint was not that the experiments were wrong. It was that the method is MoE + MAML++ (no novel component), the baselines were all our own architecture, and the evaluation was narrow.

| # | Concern | Who | What we said or promised in rebuttal | Plan item (§3) | Status |
|---|---|---|---|---|---|
| R1 | No novelty: MoE + MAML++ are known; MoEMeta (Wu, NeurIPS'25) is close; Port B is just MAML vs ANIL | ARaC, fkxS, xh89 | Relabelled Contribution 1 as an *application* contribution; Ports A/B; cite Raghu (ANIL) | E-ALT, E-MOE, framing (§5) | framing open |
| R2 | No SOTA EMG-IMU comparisons; all baselines are our own architecture | Lpuc, fkxS | Promised clearer Kaifosh / Côté-Allard rows and **EMGBench (EMG-only) with gate statistics** | E-CLS, E-SOTA | classical built; deep SOTA not started |
| R3 | Architecture not validated: why LSTM / CNN experts and not transformers? | Lpuc, fkxS | Promised a param-matched transformer (CNN experts and LSTM replaced), own HPO, same folds; removed the L171-173 claim | E-TRF | spec done (`SPEC_moe_compute.md` §3), **not implemented** |
| R4 | Modality contribution (EMG vs IMU) | Lpuc, xh89, fkxS | Ran it: IMU dominant, EMG collapses at 10-way; withdrew the fusion claim; promised a rerun under the paired design | E-MOD, E-REP | A13 exists (3-way default, needs 10-way) |
| R5 | Primary task should be 10-way (full vocabulary), not 3-way | xh89 | Corrected the meta-review's transposed numbers | protocol | **done**: 10-way 1-shot primary |
| R6 | Single dataset, single session, 32 users | Lpuc, xh89 | Stated as a limitation | E-DATA2 (optional), limitations text | open decision (§6) |
| R7 | Is the cross-subject gain just data volume? Vary the number of meta-training users | xh89 | User sweep N=4/8/16 done; promised 24/30 | E-SWEEP (A15) | token exists |
| R8 | Upper bound: fine-tune our pretrained model on all of a test user's data | ARaC | Promised for camera-ready | E-UB | **not built** |
| R9 | Kaifosh comparison unfair (EMG-only, different featurisation); run EncoderMoE on the same 2 kHz EMG-only input | ARaC | Corrected Kaifosh preprocessing; promised matched-featurisation EncoderMoE | E-REP, E-KAI | FS64 built; A10-A12 not rewired |
| R10 | Why E=22 experts when E=8 is as good? Is any subject served by a single expert? | Lpuc | Answered (no private experts; E=8 ≈ E=22) | E-MOE (X-matrix) | spec done |
| R11 | MAML for EMG already exists (Proroković 2020); cite | xh89 | — | related work | text only |
| R12 | Citations: Rahimian 2021 (FS-HGR) not cited; Antoniou entry wrong; anonymity leak (NOTS mention, A.3) | fkxS | Promised fixes | writing | text only |
| R13 | Trust: unprompted retraction + several "will be done" replies hurt credibility | fkxS, Lpuc | — | pre-registration, shared episode cache, every number on one protocol | protocol done |

**Lesson for the resubmission.** Every number must come from one protocol and be finished before submission (no "TODO" results). Results that go against us are reported, with the prediction written down first (`DECISION_method.md`, `SPEC_group_analysis.md`, `SPEC_zeroshot.md` already do this).

---

## 2. One unified protocol (decided 2026-10-21; details `SPEC_protocol.md`, `SPEC_episode_cache.md`, `SPEC_stats.md`)
- **Users.** 4 dev users (P005, P131, P116, P109) are used only for HPO and design choices and are never tested. The other 28 are split into **7 folds of 20 train / 4 val / 4 test**, stratified by ability group (`system/fixed_user_splits/kfold28_hpo4_split.json`). Every user is tested exactly once.
- **Episodes.** Committed, hashed episode files (`system/episodes/kfold7/`). Every method, deep or classical, reads the same episodes, and the aggregator rejects mismatches. Labels are permuted per episode. Queries are all non-support reps.
- **Task.** **10-way 1-shot is primary.** One 10-way checkpoint is evaluated on the full 3/5/10-way x 1/3/5-shot grid.
- **Statistics.** The user is the unit. We report bootstrap CIs over users, paired Wilcoxon + Holm over a pre-registered primary family, MDE, and 3 seeds for headline methods.
- **Scope.** Single session, reps within minutes, so there is no temporal or cross-session split (Kai's decision). This is stated as a limitation.

Nothing on this protocol has produced a real number yet. **Every table must be rerun.**

---

## 3. Experiment matrix

"Where" means **L** = laptop (CPU, minutes to hours) or **C** = cluster (GPU). Tokens refer to `kfold_launcher.sh`.

### Tier 0 — cheap, no training (run first; laptop is enough)
| ID | What | Answers | Where | Status |
|---|---|---|---|---|
| E-CLS | `$B`, `$B`-user, kNN-L1, ProtoNet-Euclid, Riemannian MDM / TS, MiniRocket on 28 users, env20 **and** fs64 | R2; the "$B may beat us" risk; R4 (by representation) | **L** (or C via `A6_*`) | built, **verified to run on the laptop 2026-10-29** (kNN-L1 fold 0 10x1 = 58.0%, 24 s) |
| E-TD | Hudgins TD features + kNN / LDA / pooled-LDA | journal reviewers' standard baselines | L | built; needs `segfilt_td64_tensor_dict.pkl` (raw CSVs from Box) |
| E-UNIQ | Gesture uniqueness: cross-user nearest-neighbour label agreement, motif matches vs a null, Mantel test vs ability group (study plan H1/H2) | the "user-defined" premise; ability-cluster story | **L** (notebook) | **not built** |
| E-ZS0 | Zero-shot Z4 (Riemannian + LR across users) and a nearest-centroid-over-train-users variant | R1/framing: is few-shot needed at all? | L | spec `SPEC_zeroshot.md`; script **not built** |

### Tier 1 — headline table (cluster, after HPO)
| ID | What | Token(s) | Status |
|---|---|---|---|
| M0 | EncoderMoE | `M0` | ready; smoke pending |
| A1/A2 | Supervised pretrain + head-only / full fine-tune (Côté-Allard-style transfer, R2) at the mirrored **and** a dev-tuned adaptation budget | `A1 A2` | ready; dev-tuned budget is in HPO |
| A3/A4 | MAML++ without MoE (param-matched = A4) | `A3 A4` | ready |
| A14 | ProtoNet readout on the frozen M0 / A4 backbone | `A14_M0 A14_A4` | ready (runs after M0/A4 per fold) |
| E-TRF | Transformers T0 (single TST, matched), T1 (transformer experts), T2 (transformer temporal encoder), T3 (both) | — | **not implemented** (`SPEC_moe_compute.md` §3) |
| E-ZS | Zero-shot Z1-Z3 (needs training) | — | **not built** |
| ANIL | Head-only inner loop (= Port B relabelled per ARaC) | `portB_head` | exists; check it is on the cache |
| ProtoNet-trained | Episodically trained ProtoNet (Sarwat et al. 2024 is the closest prior work) | — | **not built** |

### Tier 2 — ablations and mechanism
| ID | What | Answers | Status |
|---|---|---|---|
| E-MOE | X0-X4 MoE matrix on dev users: **X2 = uniform mixture (is the gate doing anything?)**, X3 sparse top-2, X4 task-conditioned routing | R1, R10; is MoE ensembling or routing? | spec only |
| E-MOD | Modality ablation, 10-way, 7 folds | R4 | `A13_*` exists, defaults to 3-way: **switch to 10-way** |
| E-REP | env20 vs FS64 (vs raw-stem) for M0/A4 | R4, R9 | FS64 built, needs a cluster copy; raw stem not built |
| E-SWEEP | Meta-training users 4/8/12/16/20 | R7 | `A15_*` exists |
| E-UB | Upper bound: pretrained model fine-tuned on all 9 non-test reps of the test user | R8 | **not built** |
| E-GRP | Per-ability-group M0-A4 gain | ability story (ExtraIdeas idea 1) | built + tested, runs after M0/A4 |
| E-SEED | 3 seeds for M0, best baseline, best ablation | stats | **seed flag not plumbed** in the eval scripts (only seed 42) |

### Tier 3 — SOTA ports and extras (only if time allows)
| ID | What | Status |
|---|---|---|
| E-SOTA | FS-HGR (Rahimian 2021, also fixes R12), EMG-CrossFormer (Atzori group, MIT code), TinyMyo frozen encoder + ProtoNet | not started (`SOTA_COMPARISONS.md`) |
| E-KAI | Kaifosh re-evaluation on the new protocol (A10-A12) | needs rewiring to the cache |
| E-PORTS | A7/A8 (subject-specific), portA, A16 ported to the cache | not ported |
| E-DATA2 | Second dataset (§6) | on hold |

---

## 4. Sequence and gates

Critical path: **smoke → representation decision → HPO → 7-fold runs → analysis → verdict → venue.**

| Stage | What | Gate to move on | Est. time |
|---|---|---|---|
| S0 | Commit and push the local work. Cluster smoke: M0/A1/A2/A4 fold 0 `--debug`, plus an HPO `--debug` trial. Install pyriemann/aeon. Check the dict hash. Copy fs64 (and later td64) to the cluster. | Smoke reaches `[Test]` + `Saved to` for all four; hash matches | 1-2 days |
| S0' (parallel, laptop) | E-CLS on all 7 folds for env20 + fs64 (primary cell first, then all cells); E-UNIQ notebook; build td64 | — | ~1 week part-time |
| S1 | **Representation decision before HPO.** Inputs: E-CLS env20 vs fs64 on 28 users; one-fold M0/A4 on fs64 with legacy hyperparameters; the PI's call on the EMG rate policy (`SPEC_2khz_emg.md` §8); the open 20 Hz anomaly | Kai + PI pick **one** `emg_repr` for the main table | ~1 week |
| S2 | HPO on dev users at 10-way 1-shot: M0, A4, A2, A1 (60 trials each, ~600 GPU-h), then the transformers once implemented | `best_hpo_configs.json` exported | 1-2 weeks wall clock |
| S2' (parallel, code) | Implement E-TRF, E-MOE X2/X4, E-ZS, E-UB, the seed flag, A13 at 10-way | unit tests pass | concurrent with S2 |
| S3 | 7-fold runs: Tier 1, then Tier 2 | aggregator accepts all JSONs | ~1-2 weeks |
| S4 | Stats, group analysis, fill the `DECISION_method.md` verdict | verdict written | days |
| S5 | Venue decision (§6) and writing | — | — |

**Why S1 comes before S2.** HPO is the most expensive step (~600 GPU-h). If HPO runs on env20 and we then switch the main table to FS64, all of it must be redone. The dev-user look (FS64 77.8% vs env20 48.9% nearest-prototype, 10-way 1-shot, 4 users) and the unresolved env20 time-axis anomaly (`FULL_RUNNING_LOG.md`, 2026-10-21 late) both point towards FS64. The 28-user classical comparison (laptop, S0') is the cheap way to confirm it. Note that the FS64 dict reuses the **old dict's IMU**, which shows the same time-axis mismatch against the raw CSVs. Rebuilding IMU from the raw CSVs is worth considering in the same step.

**Timeline check against ICML (deadline late January 2027; verify the exact date).** About 12 weeks remain. S0-S1 must finish by mid-November for HPO to finish by early December and the 7-fold runs by mid-December, leaving January for writing. Any slip in S0/S1 pushes ICML out of reach.

---

## 5. Parallel track: alternatives to EncoderMoE
Kai is not wedded to EncoderMoE. Pre-registered rules decide this (`DECISION_method.md` R1-R5), so the choice is not fitted to results. Candidates, in order of expected value and cost:
1. **The simplest learned model that wins.** If A4, a fine-tune baseline or the trained ProtoNet ties M0 (R3), lead with it and make MoE an ablation.
2. **Learned `$B` / covariance embedding** (`system/nonparametric/metric_models.py`) if a classical method wins (R4). It is cheap to iterate on, using dev users only.
3. **Task-conditioned routing (X4).** This is the only MoE variant with an originality argument against MoEMeta: the router sees the episode's support set.
4. **Better input representation (FS64 / raw stem) with a small backbone.** If FS64 is the main jump, the paper's contribution may be "representation + protocol + benchmark", not the architecture.

All exploration happens on the **dev users and dev episodes only**, so the 28-user test folds stay clean for the final comparison.

---

## 6. Open decisions

### Venue and second dataset (decide at S4, not now)
| Outcome at S4 | Venue | Second dataset? |
|---|---|---|
| R1 (MoE wins and the gate matters), or R2 + X4 wins | ICML (Jan) | Helpful for the single-dataset critique (R6) but not required. Only if it fits by mid-December. |
| R3 / R4 (simpler or classical method ties) | TMLR (rolling, values correctness over novelty), IMWUT/ASSETS (benchmark + population framing), or IEEE TNSRE / JBHI / Sensors | Not needed for TMLR or IEEE. For IMWUT, a heterogeneity comparison on public data (NinaPro amputee subsets; ExtraIdeas idea 2) strengthens the paper. |
| R5 (zero-shot already works) | IMWUT/ASSETS, HCI framing | No |

**Recommendation.** Finish the reruns first, as Kai suggested. A second dataset only helps ICML if the method survives S4, and no public dataset has user-defined gestures from disabled users, so it cannot test the main claim directly. It can only test the heterogeneity prediction (ExtraIdeas idea 2), which needs explicit approval.

### Other open decisions
- "hpo ok" (equal 60-trial budgets; frozen + tuned A1-A4) and "start hpo" (`KAI_TODO.md` §5).
- EMG rate policy (PI sign-off) and the main-table `emg_repr` (S1).
- Whether to report `$B` if it wins (yes; `DECISION_method.md` already requires it).
- Whether ICML is still the target after the S4 verdict.

## 7. Known risks
- **The env20 anomaly.** The 20 Hz EMG (and the dict IMU) time axis does not reproduce from the raw CSVs. Every old number used it. Settle it, or move to raw-derived inputs, before claiming anything about EMG.
- **Zero-shot preview.** A flat 10-way classifier scored about 15-25% on val users (chance 10%). That is near the "soften framing" band, but not a result yet.
- **Untested on the cluster.** HPO pruning, replay and export have only been tested locally.
- **Compute.** HPO plus 7 folds x 3 seeds x Tier 1-2 is large. If time runs short, cut Tier 3 and seeds for non-headline methods. Never cut the X2 uniform-mixture control, the classical baselines or the transformer comparison (all three were reviewer-requested or decisive).
