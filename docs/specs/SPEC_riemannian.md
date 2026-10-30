# SPEC — Riemannian / covariance track

*Written 2026-10-30 (Opus), at Kai's request ("implement all of those"). Background: `docs/analysis/COVARIANCE_TUTORIAL.md`. The rules in §2 and §3 were committed **before** the gate and the failure analysis were first run on real data.*

**Why.** A training-free Riemannian MDM scores 91.4% (fs64, 10-way 1-shot, 28 users). That beats every neural number we have, and it is carried by the IMU covariance. This track asks two questions:
- Can EncoderMoE build on the covariance, rather than lose to it?
- Does the clustering / MoE story survive in covariance space?

## 1. What is built
| Token / script | What | Keeps MoE / clustering? | Run where |
|---|---|---|---|
| **X7** | M0, router input = log-Euclidean tangent vector of each modality's input covariance | MoE | cluster GPU (screen) |
| **X7T** | X7 + the support set's mean covariance vector (task-level routing, like X4) | MoE | cluster GPU |
| **X8** | Riemannian experts: gate = −distance to E learnable centroids in covariance space. The centroids are initialised by k-means over the fold's TRAIN users | MoE + clustering | cluster GPU |
| **X8W** | X8 + each expert re-centres its input at its centroid (cluster-wise re-centring) | MoE + clustering | cluster GPU |
| **X9** | M0 with covariance (second-order) pooling of the LSTM states | MoE | cluster GPU |
| **A4_cov** | X9's control without MoE | no | cluster GPU |
| **R1** | M0 on inputs whitened by the episode support set's mean covariance (Euclidean alignment) | MoE | cluster GPU |
| **R1_A4** | R1's control without MoE | no | cluster GPU |
| **P1_cov** | ProtoNet on M0's expert bank with covariance pooling (CovaMNet-style) | MoE | cluster GPU |
| **P1_spd** | ProtoNet on an SPDNet per modality (BiMap / ReEig / LogEig) | no | cluster GPU |
| **H1 hybrid** | `system/analysis/hybrid_fusion.py`: model logits + MDM, weight chosen leave-one-user-out | whatever the model is | laptop, after the model's eval |
| `A6_riemann_seg_mdm` | MDM on 3 time segments: keeps coarse temporal order | no | laptop for 10x1; cluster CPU for all cells |
| `A6_riemann_rc_pool` | Per-user re-centring → tangent space → a within-class metric pooled over the TRAIN users | no | cluster CPU |
| `A6_riemann_cluster_pool` | The same with 3 user clusters (k-means on the users' mean covariances) | clustering, no MoE | cluster CPU |
| Cluster gate G1-G4 | `system/analysis/cov_cluster_gate.py` | — | laptop |
| Failure analysis | `trial_distances.py`, `gesture_descriptors.py`, `failure_analysis.py` | — | laptop |

**Not built:**
- SPDNet experts *inside* MAML. MAML would backpropagate through `eigh` in every inner step. P1_spd tests SPDNet without the inner loop first.
- Re-centring before plain MDM. It is a no-op: the affine-invariant distance does not change under re-centring.

**Implementation notes.**
- Raw-input covariances (X7, X8, R1) are computed under `no_grad`, in float64, with fixed shrinkage `cov_shrinkage` = 0.1. No gradient flows through `eigh` there.
- The X9 / P1_cov pooling defaults to Newton-Schulz `sqrt` normalisation (matmuls only); `log` (`eigh`) is an HPO choice.
- Every test-query logit of `M0` and of each `V_*` run is saved (`--dump-logits`). These feed H1 and the failure analysis.

## 2. Gate before any covariance-plus-clustering claim (pre-stated)
`cov_cluster_gate.py`, all 32 users, on env20 and on fs64. Each user's signature is the Riemannian mean of their trial covariances.

| Rule | Test | Pass |
|---|---|---|
| G1 identifiability | signature from reps 0-4 vs reps 5-9; nearest other-half signature is the user's own | rank-1 ≥ 0.5 (chance 0.03) |
| G2 cluster structure | best silhouette over K = 2..6 vs a Gaussian null with the same mean and covariance | observed > null 95th percentile |
| G3 stability | adjusted Rand index between the two halves' clusterings at the best K | ARI ≥ 0.5 |
| G4 shared motifs | after per-user re-centring, a user's gesture is nearest to another user's SAME referent | rate ≥ 0.2 (chance 0.1) |

| Outcome | Decision |
|---|---|
| G2 + G3 | Clusters exist in covariance space. X8 / X8W and `riemann_cluster_pool` carry the clustering story. |
| G1 only | Users have stable but continuous signatures. Prefer continuous routing (X7 / X7T) and frame it as "routing by user signature", not clusters. |
| G4 | Executions are shared across users: the covariance version of the ability-cluster study's H1. Report it, and test whether X7 / X8 routing follows it (the analysis's H3). |
| None | Covariance does not encode user identity well enough for routing. The clustering story must rest on something else. |

With 32 users these are go/no-go signals, not results.

## 3. Failure analysis (pre-stated)
`failure_analysis.py` on every 10-way 1-shot cached test episode (28 users, 7 folds). Classical nearest-neighbour methods come from the trial distance bank; neural models come from their `--dump-logits` files. Gesture descriptors are computed from the raw CSVs (`gesture_descriptors.py`; the labels are defined in its docstring before any accuracy is looked at).

| ID | Prediction | Supported if |
|---|---|---|
| P1 complementarity | some method gets right what MDM gets wrong | oracle-or(MDM, X) ≥ MDM + 3 points for some X, **and** a z-scored distance fusion ≥ MDM + 1 point |
| P2 movement | IMU-only MDM gains with movement; EMG carries the low-movement gestures | (EMG-only − IMU-only) is larger in the low-movement tertile than in the high one |
| P3 predictability | where MDM wins or loses is predictable from gesture properties | leave-one-user-out Spearman of the predicted Δ(MDM − X) > the within-user permutation 95th percentile, for some X |
| P4 temporal order | covariance's blind spot matters | `seg_mdm` beats `mdm` on ≥ 20% of (user, gesture) cells |

P1 decides whether H1 / fusion is worth pursuing. P2 is the requested high- vs low-movement analysis. P3 decides whether a gesture-type-aware selector (or a router trained on these descriptors) is worth building.

## 4. Promotion rules for the deep variants
These are the architecture-screen rules (`SPEC_architecture_exploration.md`):
- the dev accuracy beats M0 by ≥ 2 points, on ≥ 3 of the 4 dev users;
- every token is screened with ≥ 24 trials and the legacy shared HPs frozen.

**R4 gate:** a finalist must also beat MDM on the dev users (`DECISION_method.md` R4). If none does, the paper reports MDM as the method to beat, and the track's contribution is the analyses.

## 5. Tests to run, and where
| # | What | Where | Command / status |
|---|---|---|---|
| 1 | Unit tests | laptop | `pytest tests` (439 pass, 2026-10-30) |
| 2 | Distance bank, descriptors, failure analysis, cluster gate | laptop (the raw CSVs are on Box) | Claude runs these |
| 3 | Debug trials of the 10 new tokens: each must build, train and log epoch times | cluster GPU, debug QOS | `CURRENT_TODOs.md` §2 |
| 4 | Screen: ≥ 24 trials each, legacy HPs frozen (M0-based and A4-based tokens) | cluster GPU | `CURRENT_TODOs.md` §2 |
| 5 | Classical extensions on all 9 cells, 7 folds | cluster CPU | `CURRENT_TODOs.md` §2 |
| 6 | 7-fold eval of the finalists with `--dump-logits`; then H1 and the failure analysis with the neural rows | cluster GPU, then laptop | after the screen |
