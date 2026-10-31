# REBUTTAL CHECKLIST — every run that ends up in the paper (or decides what does)

**Venue:** ICML (Jan) or TMLR. We decide after the main runs below are finished.

**Protocol for every number:** fs64, 10-way 1-shot, 28 test users (7 folds), mean ± std over users.

**Screens:** the screen and HPO rows use the 4 dev users and report the best dev accuracy.

**Status key:**
- ⬜ not started (no code)
- 🔧 implemented, not run
- 🏃 running
- ✅ done
- ❌ failed or dropped
- ⏸ on hold

## HPO
| Run | Status | Best dev acc |
|---|---|---|
| A4 (MAML, no MoE) | ⏸ until EncoderMoE is final | |
| A2 (supervised + fine-tune) | ⏸ until EncoderMoE is final | |
| Architecture screen: one study per variant below, plus M0 and A4 as references, legacy HPs frozen, ≥24 trials each | 🔧 | |
| E × top-k grid on the best MoE config | 🔧 | |
| Finalists + re-matched controls (M0 or successor, A1, A4), full space | ⬜ | |
| Top-3 × 3-seed replay and export of the finalists | 🔧 | |

## Main paper conditions (7 folds)
| Run | Status | Acc |
|---|---|---|
| M0: EncoderMoE (MAML++ + MoE) | 🔧 | |
| A1: supervised + MoE, fine-tune | 🔧 | |
| A2: supervised, no MoE, fine-tune | 🔧 | |
| A3: MAML++, no MoE (original width) | 🔧 | |
| A4: MAML++, no MoE (param-matched) | 🔧 | |
| A14: ProtoNet / kNN readout on frozen M0 and A4 encoders | 🔧 | |
| ANIL (head-only inner loop; M0 and A4) | 🔧 | |
| 3 seeds for the headline rows | 🔧 | |

## MoE structure (screen; the best goes in the paper, the controls go in the appendix)
| Run | Status | Dev acc |
|---|---|---|
| X1: M0 gate (reference) | 🔧 | |
| X2: uniform gate (routing vs ensembling) | 🔧 | |
| X3: sparse top-1/2 + z-loss | 🔧 | |
| X4: support-set-conditioned routing | 🔧 | |
| X5: shared expert + routed experts | 🔧 | |
| X6: cross-attention router | 🔧 | |
| Routing diagnostics X1 vs X2 (entropy, I(expert; user)) | 🔧 | |

## Fusion (screen on A4)
| Run | Status | Dev acc |
|---|---|---|
| F0: early concat (= A4, reference) | 🔧 | |
| F1: mid fusion | 🔧 | |
| F2: FiLM | 🔧 | |
| F3: cross-attention fusion | 🔧 | |
| F4: modality-specific expert pools | ⬜ | |
| Best fusion carried into the MoE | 🔧 | |

## Transformer ablations
| Run | Status | Acc |
|---|---|---|
| T0: single transformer (no MoE) | 🔧 | |
| T1: transformer experts + LSTM | ⬜ | |
| T2: CNN experts + transformer instead of the LSTM | 🔧 | |
| T3: transformer experts + transformer temporal model | ⬜ | |
| T4: transformer with MoE feed-forward layers | 🔧 | |

## Baselines: classical / non-parametric (all 9 k×n cells, env20 and fs64)
| Run | Status | 10w1s acc, env20 / fs64 (mean ± SD over 28 users) |
|---|---|---|
| kNN-L1 | ✅ | 58.0 ± 11.2 / 72.9 ± 9.5 |
| ProtoNet on raw features | ✅ | 51.1 ± 12.4 / 65.6 ± 11.4 |
| `$B` (PCA + L1 1-NN) | ✅ | 78.1 ± 11.2 / 80.4 ± 8.9 |
| `$B`-user | ✅ | 58.6 ± 10.8 / 74.0 ± 9.4 |
| MiniRocket + ridge | ✅ | 78.8 ± 9.4 / 79.5 ± 9.6 |
| Riemannian MDM | ✅ | 90.4 ± 10.3 / **91.4 ± 8.6** |
| Riemannian tangent space + LR | ✅ | 86.8 ± 12.5 / 88.8 ± 9.4 |
| Time-domain features: kNN / LDA / pooled LDA | 🔧 (needs the td64 dict) | |

## Riemannian approaches (`specs/SPEC_riemannian.md`)
| Run | Status | Acc / result |
|---|---|---|
| MDM sanity: shuffled labels / EMG-only / IMU-only (env20; fs64) | ✅ | 9.5 / 73.8 / 91.0; 9.6 / 76.6 / 91.0 |
| Cluster gate G1-G4 (laptop, 32 users) | 🔧 | |
| Failure analysis P1-P4 + gesture descriptors (laptop) | 🔧 | |
| High- vs low-movement analysis (P2) | 🔧 | |
| MDM on 3 time segments (`seg_mdm`) | 🔧 | |
| Re-centring + pooled tangent metric (`rc_pool`; `cluster_pool`, 3 clusters) | 🔧 | |
| X7 / X7T: covariance-vector router (sample; + support set) | 🔧 | |
| X8 / X8W: Riemannian experts (centroid gate; + re-centred experts) | 🔧 | |
| X9 / A4_cov: covariance pooling (MoE; no-MoE control) | 🔧 | |
| R1 / R1_A4: Euclidean alignment by the support set (MoE; control) | 🔧 | |
| P1_cov: ProtoNet + covariance pooling on the expert bank | 🔧 | |
| P1_spd: ProtoNet on SPDNet | 🔧 | |
| H1: hybrid model + MDM (leave-one-user-out weight) | 🔧 (needs a model's logits) | |
| SPDNet experts inside MAML | ⬜ (after P1_spd) | |

## Baselines: few-shot / meta-learning
| Run | Status | Acc |
|---|---|---|
| ProtoNet, trained episodically (A4 encoder; also on M0's expert bank) | 🔧 | |
| Matching Network | 🔧 | |
| Relation Network | 🔧 | |
| R2-D2 (closed-form ridge head) | 🔧 | |
| FEAT | 🔧 | |
| ADANN-style (gradient-reversal user head) | 🔧 | |
| CNP | ⬜ | |
| MoEMeta-TS (time-series port of Wu & Yin; `Q1`, controls `Q1_nola Q1_mlp Q1_uni Q1_ce`) | 🔧 (go from Kai 2026-10-30 evening) | |

## Baselines: EMG-specific / SOTA
| Run | Status | Acc |
|---|---|---|
| Meta GNI (Kaifosh): zero-shot and fine-tuned, on raw 2 kHz EMG | ⬜ (needs rewiring to the episode cache) | |
| EncoderMoE EMG-only on raw 2 kHz vs GNI, same episodes | 🔧 our side (A13 emg_only on raw3s); ⬜ GNI side | |
| EMGBench EMG-only, with gate statistics (promised to fkxS) | ⬜ | |
| FS-HGR | ⬜ | |
| EMG-CrossFormer | ⬜ | |
| TinyMyo frozen encoder + ProtoNet | ⬜ | |

## Reviewer-requested ablations
| Run | Status | Acc |
|---|---|---|
| Modality: EMG-only / IMU-only / both (A13, 10-way, all cells) | 🔧 | |
| EMG input: env20 vs fs64 (dev decision) | ✅ fs64 (+17 to +29 pts EMG-only) | |
| EMG input: fs64 vs a learned raw-EMG stem | 🔧 (needs the raw3s dict built) | |
| Meta-training user sweep: 4 / 8 / 12 / 16 / 20 users (A15) | 🔧 | |
| Upper bound: fine-tune on all of the test user's other reps (A16; A2 and M0) | 🔧 | |
| Zero-shot Z1-Z4 | 🔧 | |
| Shots × ways grid: k ∈ {1, 3, 5} × n ∈ {3, 5, 10} (M0, A2, A4) | 🔧 | |
| Subject-specific models (A7 / A8) | ⏸ (not ported) | |

## Analyses
| Run | Status | Result |
|---|---|---|
| Gesture uniqueness across users | ✅ | cross-user zero-shot 13-16% vs within-user 62-78% |
| Per-ability-group gain (M0 − A4) | 🔧 (after M0/A4) | |
| Routing follows motif vs label | ⬜ (needs non-uniform routing) | |
| Body-region Gate 0 (laptop, dev users, 1-NN) | ✅ | anatomy 81.3 vs random 80.4 (94th pct, H2 fail) |
| Body-region experts (deep) | ❌ closed by Gate 0, low priority (`specs/SPEC_body_regions.md` §4) | |
| `DECISION_method.md` verdict | ⬜ (after the main runs) | |
| Second dataset | ⏸ (decided with the venue) | |
