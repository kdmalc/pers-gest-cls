# Covariance and Riemannian classifiers: a tutorial

Why a training-free classifier (Riemannian MDM, 91.4% at 10-way 1-shot on fs64, 28 users) beats every neural model we have, where the method comes from, what it means for EncoderMoE, and what to read.

Code: `system/nonparametric/classical.py` (`riemann_mdm`, `riemann_ts`, `dollar_b`); sanity checks: `system/analysis/riemann_sanity.py`.

> **Sections 1-8** were written by the first Claude Code instance (Opus, 2026-10-30) with the classical results. **Sections 0 and 9-13** were added by a second instance (Fable, 2026-10-30 evening, the one that built MoEMeta-TS and Gate 0) after discussing the results with Kai. Sections 9-13 are analysis and plans. **Correction after `git pull` (same evening):** the first instance's `docs/specs/SPEC_riemannian.md` (commit fc3130a) already builds much of what §12 proposes; §12 now marks each item as *built by the first instance* or *new*, so the provenance is exact.

---

## 0. Executive summary (read this first on Monday)

**What we found.**
- A training-free Riemannian 1-NN (MDM on shrunk channel covariances, AIRM distance) scores **91.4 ± 8.6%** at 10-way 1-shot over the 28 test users on fs64 (90.4 on env20). Tangent space + LR: 88.8. Every other classical method is 10-20+ points lower: `$B` 80.4, MiniRocket 79.5, kNN-L1 72.9, ProtoNet on raw features 65.6. The old EncoderMoE number was 67.7 (old protocol).
- **The IMU covariance does the work:** IMU-only MDM = 91.0 vs fused 91.4 (env20). EMG-only MDM = 73.8.
- The result is genuine: shuffled support labels give 9.5% (chance 10). There is within-block drift: adjacent reps 95%, farthest reps 77%, still 7x chance. Every method sees the same episodes, so the comparison is fair, but the curve goes in the paper.

**Why it wins (§9).** A gesture is a movement; the IMU covariance is "which body segments co-move and how much", which is most of a gesture's identity, and it is invariant to timing and per-user gain by construction. The neural models had to *learn* those invariances from 20 other users' *different* gestures, and cross-user zero-shot is 13-16%, so there was almost nothing to transfer. **For user-defined gestures the useful prior is geometric, not semantic.** That sentence is the paper's finding; EncoderMoE at 67% is evidence for it, not an embarrassment.

**What it means for the method (§10, §11).**
- Stop treating EncoderMoE as *the* method. It stays as the strongest deep meta-learning row.
- Do not aim at "a neural net beats 91". The drift curve puts the realistic ceiling near 94-95; a significant 28-user win in that window is unlikely.
- Build on MDM, not against it. Of the ideas in §8, the ones that *augment EncoderMoE* (X7 router, hybrid late fusion) start from 67 and cannot close 24 points; the ones that *build on MDM* (Riemannian experts / cluster-wise reference points, SPD clustering of user signatures, second-order pooling) can keep the MoE / clustering / meta-learning vocabulary on a 91 base.

**The decisive gate (§12, new).** Does *any* cross-user learning help the geometric classifier? Fit one pooled linear metric on the train users' tangent vectors and apply it within-user on dev. If it beats plain MDM, a **mixture of metrics routed by the user's mean covariance** is a legitimate "meta-learned MoE on the manifold", i.e. the original story on a strong base. If it does not help, no mixture will either, and the paper is the finding paper.

**Two honest paper shapes (§11).** (a) *Finding paper*: geometry beats deep meta-learning for user-defined gestures, and why; EncoderMoE is a row; TMLR or TNSRE. (b) *Method paper*: meta-learning on the manifold; only if §12 passes; ICML possible. The intro premise (user-defined gestures, heterogeneous disabled users, one-shot personalisation) is unchanged in both; the rewrite is results + discussion.

**The clustering / ability story survives (§11.3)** as clusters of per-user mean covariances ("movement signatures"), testable with no training. A null is reportable and still motivates per-user personalisation.

**Monday plan.** Finish the fs64 sanity rows (§13), run the §12.1 gate and the §12.3 cheap tests on the dev users (laptop with the data, or the cluster login node), then decide (a) vs (b). Nothing deep gets built before the gate.

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

---

## 9. Why covariance wins here: the finding, stated as a claim

*(Fable, 2026-10-30 evening.)*

Three facts, one explanation.

1. **IMU-only MDM equals fused MDM** (91.0 vs 91.4). The gesture's identity, for this dataset and task, is the spatial co-movement pattern of body segments. EMG adds nothing on top at 10-way 1-shot (EMG-only MDM 73.8; the fs64 row is pending, §13).
2. **MDM learns nothing, yet beats every learned model by 20+ points.** Its two invariances (timing, affine/gain) are built in. Within one session and one user they hold almost exactly, so 1-NN under the AIRM is close to the right classifier.
3. **Cross-user zero-shot is 13-16%** (notebook 03), and executions recur across users under *different* labels. The semantic prior ("what gesture X looks like") barely transfers for user-defined gestures.

Put together: the neural models were trying to learn timing and gain invariance *from 20 other users' different gestures*, which is the one thing this data cannot teach. MDM gets those invariances for free. **For user-defined gestures, the useful prior is geometric, not semantic.** This is the paper's finding. It is a claim about the problem, not about one architecture, which is why it is worth a paper even though MDM itself (Barachant 2012) and its EMG use (Gowda & Miller 2024) are not new.

What this does *not* say: that learning is useless. It says learning must act *inside* the geometry (a learned metric on the tangent space, learned features to take the covariance of) rather than replace it. §12 tests whether that is true.

## 10. Where the proposed ideas sit: augmenting EncoderMoE vs building on MDM

Kai's question (2026-10-30): are the §8 ideas improving the MDM version, or bolting covariance onto EncoderMoE? Ranked:

| Idea (§8) | Spectrum | Judgment |
|---|---|---|
| **X7 / X7T** (built): tangent-space covariance as the MoE routing signal | augments EncoderMoE | A better router on a 67% model does not close 24 points. Keep only as a routing-diagnostics curiosity. |
| **H1 hybrid** (built, `hybrid_fusion.py`): model logits + MDM, LOUO weight | augments; capped by MDM | Pays only if the neural errors are *not* a subset of MDM's. The per-query complementarity test (§8) answers that first and is the right order. |
| **X9 / A4_cov / P1_cov / P1_spd** (built): second-order pooling, CovaMNet, SPDNet | middle | The one deep option that could beat MDM in principle (it learns *which features* to take the covariance of). Also the riskiest: it re-introduces learning from other users. Gate it on §12.1. |
| **X8 / X8W** (built): Riemannian experts inside M0; **`cluster_pool`** (built): mixture of pooled metrics on MDM | X8 augments EncoderMoE (67 base); `cluster_pool` builds on MDM | Most promising for the original story. Note (§12.2) that plain re-centering is a no-op for within-user MDM because the AIRM is affine invariant; the experts must own *metrics*, not just reference points. |
| **Cluster gate G1-G4** (built, `cov_cluster_gate.py`) | builds on MDM; keeps the ability story | Cheap, no training. Do first (§12.3). |
| Plain MDM / TS, CovaMNet, SPDNet | no clustering | Baselines and controls. |

**Recommendation:** go with the build-on-MDM side, and let the gates in §12 decide between "method" and "finding". "Whichever does best" is right, with one caveat: the hybrid and X7 can never do better than MDM by more than the complementarity test allows, so measure that before building either.

## 11. The paper: finding vs method, the story, the venues

### 11.1 Two honest shapes
| | Finding paper | Method paper |
|---|---|---|
| Claim | Geometry beats deep meta-learning for user-defined gestures, and here is why (§9) | Meta-learning *on the manifold*: a learned (mixture of) metric(s) on tangent vectors, routed by the user's signature, beats MDM |
| Needs | the §13 rows, the complementarity and failure-anatomy tests, the clustering gate | the §12.1 gate to pass, then Gate 1 (§12.2) to pass on 28 users with the stats of `SPEC_stats.md` |
| EncoderMoE's role | the strongest deep meta-learning row, with the X1-vs-X2 routing result as a negative finding | an ablation row |
| Venue | **TMLR** (correctness over novelty; "does X really beat Y and why" is a TMLR staple) or **TNSRE / a domain journal** (systematic comparison against standard baselines on a real population is expected there) | **ICML** (needs the method win and an insight), else TMLR |
| Rewrite | results + discussion; the premise and the dataset / protocol sections stand | same, plus a method section |

Kai (2026-10-30): venues under consideration are **ICML, TMLR, and possibly TNSRE or a generic domain journal**; nothing else.

### 11.2 Why the premise does not change
The introduction already argues: user-defined gestures, a heterogeneous disabled population, one-shot personalisation, no cross-user label alignment. Every one of those statements is *strengthened* by the covariance result (the zero-shot 13% and the 91% are the two ends of the same argument). What changes is the answer to "what should the personalised model be", not the question. The PI's mental model of the paper's question survives.

### 11.3 The clustering / ability story
Per-user mean covariance (the Fréchet mean of a user's support covariances) is a **movement signature** that costs no training. Test (§12.3 item 1): do signatures cluster on the SPD manifold, and do clusters align with the ability groups or with per-user MDM accuracy (SD 8.6 says some users sit near 75%)? Earlier Mantel r on raw features was 0.05-0.09; the geometry may show more, or a clean null. Both outcomes are reportable: a null becomes "heterogeneity is individual, not group-level", which still motivates per-user personalisation. If the §12.1 gate passes, the clusters are also the natural experts of the mixture.

### 11.4 Many comparisons: how the three venues read them
- **ICML:** a wide table without a mechanism reads as "benchmark paper" and scores low on significance. Each comparison must answer a question.
- **TMLR:** thorough comparisons and negative results are welcome; correctness is what is reviewed.
- **TNSRE / domain journal:** MDM, LDA, kNN are the baselines they would ask for anyway; a systematic table is expected, not penalised.

Kai's PI is right for a method paper: a main table with **one representative per family** (raw kNN-L1, MDM, pretrained + fine-tune, ProtoNet, MAML / EncoderMoE, ours), the rest in the appendix. The comparisons become interesting only when they expose **failure modes** that the proposed method fixes (Kai's own criterion). That analysis has not been done; §12.3 items 3-5 are it.

## 12. Planned analyses: reconciliation with `SPEC_riemannian.md`

**Provenance.** This section was drafted by the second instance (Fable) on 2026-10-30 evening *before* pulling the first instance's `SPEC_riemannian.md` (fc3130a). After the pull it turned out the first instance had already built the core of 12.1 (`A6_riemann_rc_pool`), a K=3 version of 12.2 (`A6_riemann_cluster_pool`), and several 12.3 items. Each item below is now tagged **[built: token]** (first instance; code exists, not run) or **[NEW]** (second instance; no code). The rules and gates in `SPEC_riemannian.md` §2-4 are the ones to follow; what this section adds is the *reading* of the results (what each outcome means for the paper) and the items tagged NEW.

All of it runs on CPU with no training beyond a linear fit, on the cached episodes. Dev users (4) for the gates; 28 users only for whatever passes.

### 12.1 The decisive gate: a pooled learned tangent metric  **[built: `A6_riemann_rc_pool`]**
**Question.** Does *any* cross-user learning help the geometric classifier? The first instance built exactly this (per-user re-centring → tangent space → within-class metric pooled over the TRAIN users). What is NEW here is (i) the decision reading below, and (ii) the second variant in step 2 (an episodically learned low-rank W), which is not built.

**Procedure.**
1. For every train user, map each trial's shrunk covariance (per modality, as in `riemann_mdm`) to the tangent space **at that user's own mean** (per-user re-centering, Zanini 2018; this is what makes tangent vectors from different users comparable, and is where re-centering actually matters, see 12.2).
2. Fit one linear map W on the pooled train-user tangent vectors so that within-user distances separate gesture classes. Start with the cheapest: LDA-style whitening by the pooled within-class covariance (Ledoit-Wolf), as `classical.td_lda_pool` already does for time-domain features. Second variant: a low-rank W learned episodically (a linear ProtoNet on tangent vectors, trained on train-user episodes).
3. On each dev user: re-centre at the user's support mean, apply W, 1-NN (nearest class mean at K > 1). Compare with plain MDM on the same episodes.

**Readout.** Gain over MDM per dev user at 10-way 1-shot and 5-shot. A gain of >= 2 points on >= 3/4 dev users is the same bar as Gate 0.

**What each outcome means.**
- *Passes:* cross-user learning helps inside the geometry. Proceed to 12.2 (the mixture). The method paper is alive.
- *Fails:* the geometric prior is already saturated for this data; no mixture of metrics will help either. Finding paper. EncoderMoE and MoEMeta-TS become rows.

### 12.2 Gate 1: a mixture of metrics routed by the user's signature ("MoE on the manifold")  **[built: `A6_riemann_cluster_pool` (K=3, k-means); controls NEW]**
Only if 12.1 passes. `cluster_pool` is the K=3 hard-routed version. NEW: the controls (K=1 = `rc_pool`, random cluster assignment of the same sizes, uniform routing) and soft routing; without the random-assignment control a `cluster_pool` win cannot be attributed to the clusters (the body-region lesson). Note also that **X8 / X8W** (`SPEC_riemannian.md`) are Riemannian experts *inside M0*, so they inherit the 67% base; `cluster_pool` is the version that builds on MDM.
- K metrics W_1..W_K instead of one. Each train user is assigned to a cluster (Riemannian k-means on user mean covariances, or a learned soft router on the tangent vector of the user mean); W_k is fitted on its cluster's users.
- A new user is routed from the support set alone (its mean covariance), then classified with W_k. Soft routing = weighted sum of distances.
- Controls, in the spirit of X1 vs X2: K = 1 (= 12.1), random cluster assignment, uniform routing. If ability groups are used as clusters, compare with random groups of the same sizes (the body-region lesson).
- **Why re-centering alone is not a method.** The AIRM is affine invariant, so whitening a user's covariances by that user's mean leaves all *within-user* MDM distances unchanged. Per-user or per-cluster reference points therefore change nothing for 1-shot MDM; they only matter as the preprocessing that makes *pooling across users* (12.1) meaningful. Experts must own metrics W_k, not just reference points.

### 12.3 Cheap tests that shape the paper either way
1. **[built: `cov_cluster_gate.py` G1-G4]** **SPD clustering of user signatures.** Riemannian k-means / Fréchet-mean clustering of the 32 per-user mean covariances (per modality). Silhouette vs a permutation null; Mantel / adjusted Rand against ability groups; correlation of cluster with per-user MDM accuracy. This is the clustering gate from §8, made concrete.
2. **[NEW]** **k-shot curve.** MDM (Riemannian class means) vs the best learned model at K = 1, 3, 5. If learning only pays at K = 5, say so; if MDM's lead shrinks with K, the method paper has a niche at higher shots.
3. **[built: `failure_analysis.py` P1-P4, `gesture_descriptors.py`]** **MDM failure anatomy.** On dev, list the confused class pairs. Are they time-reversed or same-co-activation-different-order pairs (the §2 blind spots)? Posture gestures that differ only in mean orientation? This decides whether a temporal model has any job left.
4. **[built: `A6_riemann_seg_mdm` (3 segments); augmented covariance NEW]** **Segment / augmented covariances.** 2-4 windows per trial, one covariance each, summed AIRM distances; or the augmented (time-lagged, Hankel-style) covariance, which encodes dynamics while staying SPD (Carrara & Papadopoulo 2023, "augmented covariance matrices"; verify the citation). If this recovers the item-3 failures cheaply, the temporal-model niche closes too.
5. **[partly built: `riemann_sanity.py` has the MDM curve; other methods NEW]** **Rep-gap stratified comparison for every method.** Accuracy vs |rep(query) - rep(support)| for MDM, kNN, `$B`, MiniRocket and the deep rows. If learned models degrade less with gap, that is the one robustness story they can still own (drift tolerance). If they degrade more, the finding is complete.
6. **[close to built: cluster gate G4 "shared motifs"; the Z-style accuracy readout is NEW]** **Cross-user zero-shot in the geometry.** Cross-user MDM on label-aligned executions (notebook 03's motif matches) as the geometric analogue of Z1-Z4. If it is also near chance, "geometric prior transfers, semantic does not" is nailed in the same space.
7. **[NEW]** **Per-user difficulty predictor.** Can the dispersion of a user's support covariances (mean pairwise AIRM) predict that user's MDM accuracy? A practical, HCI-flavoured result: "who needs more shots", from the calibration data alone.
8. **[NEW]** **Sensor-block Gate 0.** Per-IMU-sensor and per-EMG-channel block covariances scored with the Gate 0 logic (`SPEC_gate0.md`), to find the smallest sensor set that keeps 91. Wearability story; also tells whether the 72-channel IMU covariance is over-provisioned.

### 12.4 What stays off the table until the gates report
The first instance has already *built* X7/X7T, X8/X8W, X9/A4_cov, R1/R1_A4, P1_cov, P1_spd and the H1 hybrid (`SPEC_riemannian.md` §1), so "off the table" here means *do not spend GPU screen budget on them before `rc_pool`, `cluster_pool`, the cluster gate and the failure analysis (all CPU) have reported*. X9 / P1_cov / P1_spd are the second-order-pooling options and are the deep rows most worth screening if 12.1 passes. MoEMeta-TS on covariance features: not built, not planned. The architecture screen (`SPEC_architecture_exploration.md` §3) still runs as planned because its rows are needed for the finding paper too.

## 13. The pending fs64 sanity rows and how they will be used

The other instance is producing the fs64 versions of §4 (fused, shuffled labels, EMG-only, IMU-only, rep-gap). Planned use:
- **Shuffled labels at chance** confirms no leakage on the representation that goes in the paper.
- **IMU-only vs fused on fs64** decides the modality statement for the paper (the reviewer-promised modality ablation, R4): if IMU-only still matches fused, EMG contributes nothing to the headline classifier at 10-way and the paper says so for the classical row *and* for the deep rows (A13 at 10-way).
- **EMG-only fs64 vs 73.8 (env20)** tells whether the spectral EMG covariance (112 channels) carries more than the envelope covariance (16); this feeds the EMG-representation paragraph and the Meta-GNI fairness argument (`SPEC_2khz_emg.md`).
- **Rep-gap curve on fs64** goes in the paper as the drift caveat and is the reference for §12.3 item 5.
- If the fs64 numbers differ materially from env20 (they should not: MDM's invariances make it insensitive to the EMG representation), that itself is a finding about the representation.
