# Covariance and Riemannian classifiers: a tutorial

Why a training-free classifier (Riemannian MDM, 91.4% at 10-way 1-shot on fs64, 28 users) beats every neural model we have, where the method comes from, what it means for EncoderMoE, and what to read.

Code: `system/nonparametric/classical.py` (`riemann_mdm`, `riemann_ts`, `dollar_b`); sanity checks: `system/analysis/riemann_sanity.py`.

---

## 1. The object: one covariance matrix per trial

A trial is a matrix X of shape (C, T): C channels (16 or 112 EMG, 72 IMU) by T = 64 time steps. Its channel covariance is

    S = (1 / (T-1)) · (X - mean_t X)(X - mean_t X)^T        shape (C, C)

- **Diagonal entries** S_ii: the variance (power) of channel i over the trial, i.e. how much that muscle or that IMU axis is active.
- **Off-diagonal entries** S_ij: how channels i and j move together. Same sign means they rise and fall together; opposite sign means one rises as the other falls; near zero means they are unrelated.
- **What it throws away:**
  - *Temporal order.* Shuffle the 64 time steps and S does not change.
  - *The per-channel mean.* `pyriemann` centres each channel first, so a static posture offset is lost.

So S is a summary of the **spatial co-activation pattern**: which muscles fire together and which body segments move together, and how strongly. For a gesture that is most of its identity.

**Why this works as a statistic.** If X were zero-mean Gaussian, S would be its sufficient statistic: two trials with the same S are indistinguishable to any likelihood-based classifier. EMG envelopes and IMU signals are not Gaussian, but this is still the right intuition. S is the second-order "shape" of the multichannel signal.

## 2. Why covariance beats raw-tensor methods here

| Property | What it buys at 1-shot |
|---|---|
| **Invariant to timing** (onset, speed, pauses within the 64 steps) | kNN-L1 on the raw flattened tensor scores 58.0% (env20) because two reps of one gesture are misaligned in time. S does not care about alignment. |
| **No parameters to learn** | With 1 example per class, any learned model must get its prior from other users. MDM learns nothing and needs no prior. |
| **Affine invariance** (§3) | Per-channel gain changes (electrode impedance, sensor placement, a stronger or weaker user) leave the distance unchanged. |
| **Well-defined comparison** | Distances respect the geometry of covariance matrices (§3), so a large variance is not automatically "far" from a small one. |

**The price** (things S cannot tell apart):
- **A gesture and its time-reversal.** "Raise then lower" and "lower then raise" give the same S.
- **Two gestures with the same co-activation but a different sequence.**
- **Two static postures that differ only in mean orientation**, since S is centred.

The neural models can, in principle, use all of these. So far they lose more by being timing-sensitive and data-hungry than they gain from this extra information.

## 3. Where the Riemannian part comes from

### 3.1 Covariances live on a curved space
A covariance matrix is **symmetric positive definite (SPD)**. The set of SPD matrices is an open convex cone, not a flat vector space:
- The difference of two SPD matrices need not be SPD.
- The straight-line (Euclidean) average shows the **swelling effect**: the determinant of the arithmetic mean can exceed the determinant of every input.
- Euclidean distance is dominated by the largest-variance channels.

The fix is to give the cone a Riemannian metric. The standard choice is the **affine-invariant metric** (AIRM; Pennec et al. 2006; Förstner and Moonen 1999 for the distance):

    d(A, B) = || log( A^{-1/2} B A^{-1/2} ) ||_F  =  sqrt( Σ_i log² λ_i(A^{-1} B) )

Here λ_i are the generalised eigenvalues of the pair (A, B). The distance compares the two matrices by **ratios** of variance along matched directions, not differences. Doubling every variance moves you by a fixed log 2 per direction, wherever you start.

**Affine invariance:** d(W A W^T, W B W^T) = d(A, B) for any invertible W.
- In practice, any fixed linear mixing or rescaling of channels that hits support and query alike cancels out. Examples: per-channel gains (diagonal W), cross-talk between electrodes, or a rotated IMU frame.
- Within one user and one session this holds closely. That is a big part of why the method is so robust at 1-shot.
- The AIRM is also the **Fisher-Rao metric** of zero-mean Gaussians (up to a constant), so it is the "natural" distance between the distributions the covariances describe.

### 3.2 Means and the tangent space
- **Riemannian (Fréchet or Karcher) mean:** the matrix minimising the sum of squared AIRM distances to a set (Moakher 2005). There is no closed form; it is found by a fixed-point iteration.
- **Tangent space:** take a reference point C_ref and map each S to the vector
  `upper( log( C_ref^{-1/2} S C_ref^{-1/2} ) )`, with off-diagonal entries weighted by √2.
  - Near C_ref, Euclidean distances between these vectors approximate AIRM distances.
  - So any ordinary classifier (LDA, logistic regression, SVM) can be used on them.
  - Dimension: C(C+1)/2. That is 136 for 16-channel EMG, 6,328 for 112-channel fs64 EMG and 2,628 for the IMU.
- **Log-Euclidean metric** (Arsigny et al. 2007): a cheaper approximation that uses log(S) directly, at the identity.

### 3.3 The two classifiers we run
- **MDM (minimum distance to mean; Barachant et al. 2012).**
  - Fitting: one Riemannian mean per class from the support covariances.
  - Prediction: the class whose mean is nearest in AIRM distance.
  - **At 1-shot the class mean is the single support covariance, so MDM is just 1-nearest-neighbour under the AIRM.**
  - We run one MDM per modality and sum the EMG and IMU distances.
- **Tangent space + LR (Barachant et al. 2013):**
  - Covariances are mapped to the tangent space at the mean of all support covariances, then classified with logistic regression.
  - With 10 support points in a space of thousands of dimensions it overfits a little, which is why it trails MDM (88.8 vs 91.4 on fs64).

### 3.4 Estimating S when C ≥ T
- With T = 64 steps, the sample covariance has rank ≤ 63. That leaves it **singular** for the 72 IMU channels and the 112 fs64 EMG channels.
- A singular matrix is not SPD: log and inverse are undefined, and the AIRM is infinite.
- We therefore use **shrinkage**: Ŝ = (1-ρ)S + ρ·(tr S / C)·I.
  - The weight ρ is chosen analytically, by Ledoit-Wolf (2004) or OAS (Chen et al. 2010).
  - We use OAS.
- This is not cosmetic: without shrinkage the method cannot run on fs64 at all.

### 3.5 History in one paragraph
- **Earlier uses of the geometry:**
  - Diffusion-tensor MRI, where every voxel is a 3×3 SPD tensor (Pennec et al. 2006; Arsigny et al. 2007).
  - Computer vision "region covariance" descriptors (Tuzel et al. 2006).
  - Radar signal processing.
- **Common Spatial Patterns (CSP; Ramoser et al. 2000; Blankertz et al. 2008):** for decades the standard EEG pipeline for motor imagery.
  - It finds spatial filters from the two class covariances.
  - The log-variance of the filtered signals then feeds an LDA.
- **Barachant's insight:** CSP + LDA is an indirect way of classifying covariance matrices, so classify them directly with the right geometry.
  - MDM and the tangent-space pipelines (2010-2013) matched or beat CSP with fewer steps.
  - They placed first in several Kaggle EEG competitions (around 2014-2015).
  - They became standard BCI baselines (reviews: Congedo et al. 2017; Yger et al. 2017), implemented in `pyRiemann`.
- **Cross-subject transfer:** "re-centering" (Zanini et al. 2018) whitens each subject's covariances by that subject's own mean, so every subject's data sits at the identity. Riemannian Procrustes (Rodrigues et al. 2019) also matches dispersion and rotation. Euclidean Alignment (He and Wu 2020) is the cheap Euclidean version.
- **Deep SPD networks:** SPDNet (Huang and Van Gool 2017), Riemannian batch norm (Brooks et al. 2019) and domain-specific SPD batch norm (TSMNet; Kobler et al. 2022).
- **EMG:** Riemannian methods have come later to EMG than to EEG. Gowda and Miller (2024) is a recent EMG application.

## 4. Our sanity checks (`riemann_sanity.py`)

Same cached test episodes as the cluster, all 7 folds, 10-way 1-shot, 28 users, 50 episodes per user.

| Condition | env20 | fs64 |
|---|---|---|
| Fused MDM (as on the cluster) | 90.4 ± 10.1 | 91.5 ± 8.3 |
| Support labels shuffled | **9.5 ± 1.0** (chance 10) | **9.6 ± 1.1** |
| EMG only | 73.8 ± 14.7 | 76.6 ± 10.6 |
| IMU only | **91.0 ± 9.6** | **91.0 ± 9.6** (same IMU) |

Accuracy by |rep(query) − rep(true-class support)| (fs64 is within 1-2 points: 96.1 at gap 1, 76.4 at gap 9). env20:

| Gap | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Accuracy (%) | 95.2 | 93.5 | 92.5 | 90.2 | 88.3 | 86.6 | 83.3 | 80.1 | 76.7 |

What the checks show:
- **No leakage.** With the support labels shuffled, accuracy falls to chance.
- **The IMU covariance does the work.** IMU-only matches the fused score, so EMG adds nothing on top at 10-way.
- **Some within-block drift.** The 10 reps of a gesture were recorded back to back, and neighbouring reps are easier to match (95%) than reps far apart (77%). Even the farthest pairs stay 7× above chance, so most of the accuracy is genuine. Every method sees the same episodes, so comparisons stay fair, but the paper should report this curve.

## 5. Does `$B` use covariance? Yes, in a different way

`$B` (`classical.dollar_b`, the faithful port of `run_$B.ipynb::exact_dollar_B_classify`) works like this:
1. For **each support template**, compute its own (88 × 88) channel covariance.
2. Take the top 50 eigenvectors. These are PCA **spatial filters**: the template's main co-activation directions.
3. Project the template and every query onto those 50 directions, giving (50, T) trajectories.
4. Compare them with **L1 over the full time course**, and classify by 1-NN.

So `$B` uses the covariance to choose *a coordinate system per template*. It then compares **time-aligned trajectories**, which keeps the timing sensitivity that MDM avoids. That plausibly explains the ordering:

| Method | env20 |
|---|---|
| kNN-L1, raw tensor | 58.0 |
| `$B` | 78.1 |
| MDM | 90.4 |

`$B` gains from the spatial filtering but still pays for misalignment. **Not tested:** why exactly the per-template basis helps this much over kNN-L1. With 88 channels and rank ≤ 63, 50 PCs keep most of each template's energy, so the gain is not simple denoising.

`$B`-user fits **one** PCA on the pooled support and is much weaker (58.6 on env20). The template-specific basis is what matters.

**Other places covariance appears in the repo** (probably what you remember from "feature engineering"):
- `system/nonparametric/eval_knn_proto.py`: channel-wise PCA spatial filters fit on the support covariance (averaged per trial for K > 1), plus the cumulative-explained-variance plot for EMG / IMU / both.
- `system/nonparametric/metric_models.CovarianceEmbeddingNet`: the upper triangle of the raw covariance → MLP → L2 distance. This is the "learned covariance metric" line in `DECISION_method.md` R4. **Caveat:** it uses the raw covariance, not the log-map, so it ignores the geometry of §3. Any revival should feed it `upper(log(Ŝ))` with shrinkage.
- `classical.td_lda_pool`: LDA whose pooled within-class covariance (Ledoit-Wolf) comes from the TRAIN users. This is covariance used for whitening, not as the feature.

## 6. Implications for EncoderMoE

1. **The bar is a training-free 91%, carried by the IMU.** A neural model must at least recover what IMU channel co-activation already gives. The R4 risk in `DECISION_method.md` is live.
2. **Our encoders lack the right inductive biases.**
   - The CNN-LSTM summarises a trial by first-order, time-ordered features. It has no explicit second-order pooling and no gain invariance.
   - It must learn timing invariance from other users' data, which is the hardest thing to transfer across users with user-defined gestures.
3. **Build on covariance rather than compete with it.** Candidate directions (from the 2026-10-30 discussion, **none built**):
   1. **Hybrid late fusion.** Learned logits + λ·(−MDM distance), with λ tuned on dev (`hybrid_np_enable`). Cheapest, and it can only add to MDM if λ is tuned.
   2. **Second-order pooling head.** Encoder features → shrunk covariance → log-map → ProtoNet or MDM head (CovaMNet, Li et al. 2019; MPN-COV, Li et al. 2017). The network then learns *which features to take the covariance of*.
   3. **X7 router:** tangent-space vector of the sample's or support set's covariance as the MoE routing signal. This is a natural "user / gesture style" descriptor.
   4. **Per-user re-centering** (Zanini et al. 2018) or Euclidean alignment (He and Wu 2020) of the inputs using the support covariance, before any model. This targets cross-user shift directly.
   5. **SPDNet experts.** Expensive: MAML would backpropagate through `eigh`, which is unstable when eigenvalues are close (Ionescu et al. 2015).
4. **Covariance's blind spots are where a learned model can win:** temporal order, time-reversed pairs, mean posture. If a dev-set confusion analysis shows MDM failing on such pairs, that is a concrete story for why a learned temporal model is needed.
5. **Report in the paper:** MDM and TS as strong baselines, the shuffled-label check, the modality split and the rep-gap curve.

## 7. Reading list

**Start here**
- Congedo, Barachant, Bhatia (2017). *Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review.* Brain-Computer Interfaces 4(3). Best single introduction.
- Barachant, Bonnet, Congedo, Jutten (2012). *Multiclass brain-computer interface classification by Riemannian geometry.* IEEE TBME 59(4). MDM and the tangent-space idea.
- Yger, Berar, Lotte (2017). *Riemannian approaches in brain-computer interfaces: a review.* IEEE TNSRE 25(10).

**Geometry**
- Förstner, Moonen (1999). *A metric for covariance matrices.* (Tech. report; reprinted 2003.) The AIRM distance.
- Pennec, Fillard, Ayache (2006). *A Riemannian framework for tensor computing.* IJCV 66(1).
- Moakher (2005). *A differential geometric approach to the geometric mean of symmetric positive-definite matrices.* SIAM J. Matrix Anal. Appl. 26(3).
- Arsigny, Fillard, Pennec, Ayache (2007). *Geometric means in a novel vector space structure on symmetric positive-definite matrices.* SIAM J. Matrix Anal. Appl. 29(1). Log-Euclidean.
- Bhatia (2007). *Positive Definite Matrices.* Princeton University Press. Reference text.

**Estimation**
- Ledoit, Wolf (2004). *A well-conditioned estimator for large-dimensional covariance matrices.* J. Multivariate Analysis 88(2).
- Chen, Wiesel, Eldar, Hero (2010). *Shrinkage algorithms for MMSE covariance estimation.* IEEE TSP 58(10). OAS, the estimator we use.

**Classifiers and precursors**
- Barachant, Bonnet, Congedo, Jutten (2013). *Classification of covariance matrices using a Riemannian-based kernel for BCI applications.* Neurocomputing 112. Tangent space.
- Ramoser, Müller-Gerking, Pfurtscheller (2000). *Optimal spatial filtering of single trial EEG during imagined hand movement.* IEEE TRE 8(4). CSP.
- Blankertz, Tomioka, Lemm, Kawanabe, Müller (2008). *Optimizing spatial filters for robust EEG single-trial analysis.* IEEE Signal Processing Magazine 25(1).
- Tuzel, Porikli, Meer (2006). *Region covariance: a fast descriptor for detection and classification.* ECCV.

**Transfer across subjects (relevant to cross-user)**
- Zanini, Congedo, Jutten, Said, Berthoumieu (2018). *Transfer learning: a Riemannian geometry framework with applications to brain-computer interfaces.* IEEE TBME 65(5). Re-centering.
- Rodrigues, Jutten, Congedo (2019). *Riemannian Procrustes analysis: transfer learning for brain-computer interfaces.* IEEE TBME 66(8).
- He, Wu (2020). *Transfer learning for brain-computer interfaces: a Euclidean space data alignment approach.* IEEE TBME 67(2). arXiv 2018.

**Deep learning with covariances**
- Ionescu, Vantzos, Sminchisescu (2015). *Matrix backpropagation for deep networks with structured layers.* ICCV.
- Huang, Van Gool (2017). *A Riemannian network for SPD matrix learning.* AAAI. SPDNet.
- Li, Xie, Wang, Zuo (2017). *Is second-order information helpful for large-scale visual recognition?* ICCV. MPN-COV.
- Brooks, Schwander, Barbaresco, Schneider, Cord (2019). *Riemannian batch normalization for SPD neural networks.* NeurIPS.
- Li, Xu, Wang, Huo, Gao, Luo (2019). *Distribution consistency based covariance metric networks for few-shot learning.* AAAI. CovaMNet: covariance as the few-shot class representation.
- Kobler, Hirayama, Zhao, Kawanabe (2022). *SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG.* NeurIPS. TSMNet.

**EMG**
- Gowda, Miller (2024). Riemannian methods for EMG gesture decoding. J. Neural Eng. ([arXiv 2311.08548](https://arxiv.org/abs/2311.08548)). Already cited in `specs/SOTA_COMPARISONS.md`.

**Software**
- `pyRiemann` (Barachant et al.), the library we use: `Covariances(estimator="oas")`, `MDM(metric="riemann")`, `TangentSpace`.

*Check before citing:* the author lists and venue details above are from memory. Verify each against the publisher page before it goes into the paper. Verify Li et al. 2017 (MPN-COV) and Li et al. 2019 (CovaMNet) most carefully.

## 8. Fairness, feature combinations, and which options keep MoE / clustering (2026-10-30)

**Fairness.** The covariance is a fixed function of the same windows every method sees. It uses no extra data, no labels and no other users. Covariance of the *input* is feature engineering. Covariance of *learned features* is a pooling layer (second-order pooling), so it is architecture. Either is fair against a raw-signal model such as Meta GNI, as long as:
- every method uses the same episodes;
- MDM / TS are reported as baselines;
- there is an ablation of our model with and without the covariance part.

The real reviewer risk is "MDM alone gets 91%", not unfairness. The model has to beat MDM.

**Does anything carry information that covariance misses?** Test this offline before building any model:
- **Per-query complementarity:** the oracle "MDM or X correct" rate, and error overlap, for X ∈ {MiniRocket, td features, `$B`}.
- **Distance-level fusion:** a z-scored sum of distances (not feature concatenation, which dilutes at 1-shot).
- **Segment covariances:** split each trial into 2-4 windows and use one covariance per window. This keeps coarse temporal order, which fixes the time-reversal blind spot.

fs64 EMG is already a spectral time series: 16 channels × 7 spectral features = 112 channels. So its covariance already mixes in the spectral information.

**Options by how much MoE / clustering they keep**

| Keeps | Option | Reuses from the paper | Cost |
|---|---|---|---|
| MoE is the method | **X7:** router on the tangent-space covariance | MoE, MAML, routing diagnostics (X1 vs X2, I(expert; user)) | low |
| MoE is the method | **Riemannian experts:** each expert owns a reference point (cluster-wise re-centering), routed by distance on the SPD manifold | MoE framing, routing analyses, ability-cluster hypothesis | medium |
| MoE is the method | **Second-order pooling experts:** each expert = encoder → covariance → log-map; ProtoNet / MDM head | MoE, MAML, episodes | medium |
| MoE kept, covariance on the side | **Hybrid late fusion:** MoE logits + λ·(−MDM distance) | everything | lowest |
| Clustering without MoE | Riemannian k-means / Gaussian mixture over per-user mean covariances, then per-cluster re-centering + MDM | ability-cluster analyses | low |
| No clustering | Plain MDM / TS, per-user re-centering, CovaMNet, SPDNet | eval pipeline only | low to high |

**Gate before any clustering claim:** do users' mean covariances cluster at all, and do the clusters match ability groups or the MoE routing? If not, the covariance-plus-clustering story is weak.
