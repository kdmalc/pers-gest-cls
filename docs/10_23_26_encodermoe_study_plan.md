# EncoderMoE Study Plan: Testing the Sample-Level Structure Hypothesis

## 0. Purpose

Two goals, run as one study:

1. **Test the mechanism EncoderMoE claims** (sample-level structure exploited by routing) directly, instead of asserting it.
2. **Run a fair, complete baseline comparison**, including simple per-user methods ($B, PCA-KNN, DTW) that may be competitive or better, on both 20 Hz envelope EMG and 2 kHz EMG.

Success criterion: whatever the results, every claim in the resubmission matches the evidence.

---

## 1. Restating the hypothesis

**Original (user-level):** ability level -> users form clusters -> users in a cluster choose similar gesture vocabularies.
Status: not supported. Unsupervised user-level clustering found nothing, and the NeurIPS model's routing was near uniform.

**Revised (sample-level):** users do not share whole vocabularies, but individual gesture *executions* recur across subsets of users (shared movement "motifs"), and similar ability makes sharing more likely. A model that routes per sample can exploit partial, scattered overlap that user-level clustering misses.

**The revised hypothesis does not rest on clustering.** It rests on pairwise overlap between executions, which is a weaker and more testable claim. It decomposes into four parts, and only H3 and H4 involve EncoderMoE at all:

| ID | Claim | Test | Phase |
|---|---|---|---|
| H1 | Motifs exist: executions have cross-user near neighbors more often than a null predicts | Model-free cross-user neighbor analysis | 1 |
| H2 | Ability predicts sharing: user pairs with similar ability share more motifs | Mantel test on user x user matrices | 1 |
| H3 | Routing uses motifs: gate vectors follow execution similarity across users (more than user identity), and routing is causally necessary | Routing decomposition, gate ablations | 3 |
| H4 | MoE benefit tracks sharing: per-user gain from routing is larger for users whose executions have cross-user neighbors | Per-user correlation, pre-registered direction | 3 |

Predicted directions are written down before running (see Section 7).

---

## 2. Phase 0: $B reproduction (do this first, 1 to 2 days)

If $B (per-user PCA-KNN) reaches about 80% one-shot on the 20 Hz data under the same protocol EncoderMoE is evaluated on, that is the most important result in this study. It would mean per-user template matching beats cross-user meta-learning, which changes what the paper is.

Steps:
- Find the exact protocol behind the 80% figure: number of classes (N-way), templates per class, which repetitions were train vs. test, segmentation, same-session or not.
- Re-run $B and generic PCA + 1-NN on the **exact episodes** EncoderMoE uses (same users, N, K, query sets).
- Compare at matched N-way and K-shot.

Do not proceed to Phase 2 framing decisions until this is settled.

---

## 3. Phase 1: Model-free data analysis (H1, H2), about 1 week

Cheap, needs no training, and decides whether the sample-level premise holds.

### 3.1 Representation and units
- Primary: one fixed, non-learned representation per modality (handcrafted features, see 4.1 RF), time-normalized to a fixed length or compared with DTW.
- Normalization: decide per-user vs. global z-scoring up front and state it. Per-user normalization removes amplitude differences between users, which changes what "similar" means.
- Unit: execution prototype p(u, g) = mean over repetitions of gesture g by user u (32 x 10 = 320 prototypes). Repeat with individual repetitions as a robustness check.

### 3.2 H1: Do motifs exist?
- For each p(u, g), find its nearest neighbors among all other users' prototypes (any label).
- **Match definition (self-calibrating):** another user's execution is a "motif match" if it is as close to p(u, g) as user u's own repetitions are (e.g., within the 90th percentile of u's own repetition-to-prototype distances for g).
- **Null:** recompute matches after channel-permuting the other users' data (destroys spatial and movement structure, keeps amplitude statistics).
- **Label agreement:** among motif matches, fraction that share the same prompt label. Chance is about 1/10. This directly measures how much prompt semantics are shared across users (the question in dispute with Momona).
- Outputs:
  - Histogram of matches per execution (real vs. null).
  - Label agreement rate among matches.
  - User x user matrix M, where M[u, v] = number of motif matches between users u and v.

**Known confound:** low-motion executions may look alike simply because little is happening. Report results per modality, and flag low-energy executions.

### 3.3 H2: Does ability predict sharing?
- Mantel test (10k permutations) between M and an ability-similarity matrix.
  - Primary: clinical ability groups (same group = 1, else 0). Not derived from the signals, so not circular.
  - Secondary: range-of-motion proxy from IMU (e.g., per-user gyro energy or angular excursion averaged over gestures), using |proxy difference|.
- **Circularity rule:** when the ability proxy is IMU-derived, compute M on EMG only.
- Descriptive only: check whether the "finger-strategy" users form a visible block in M.
- Power: with N = 32, report the Mantel r with its permutation p-value and do not overinterpret marginal results.

### 3.4 Decision point
- H1 fails (match counts at null level, label agreement at chance): the sample-level premise is unsupported. MoE claims get limited to capacity or ensembling.
- H1 holds, H2 fails: motifs exist but are not ability-linked. The routing story can survive without the ability framing.
- Both hold: proceed with the full mechanism story.

---

## 4. Phase 2: Benchmark grid

### 4.1 Input representations

| ID | Description | Valid for |
|---|---|---|
| R20 | 20 Hz envelope EMG (original pipeline) | All methods |
| RF | 2 kHz EMG -> windowed handcrafted features (TD features + Khushaba spectral moments / TD-PSD), e.g., 150 to 250 ms windows with 50% overlap, giving a feature sequence | All methods |
| RC | 2 kHz EMG -> learned strided 1D CNN stem feeding the same encoder | Trainable methods only |

- Choose the window length on validation users only.
- Keep the temporal context (gesture duration covered) identical across representations.
- IMU uses its native rate, with the same windowing as the EMG representation it is paired with.

### 4.2 Modalities
EMG-only, IMU-only, EMG+IMU.
Rationale: at 20 Hz, IMU-only matched or beat fused. 2 kHz EMG may change this, and the paper must report it either way.

### 4.3 Methods

| Group | Method | Uses other users? | Notes |
|---|---|---|---|
| A. Per-user | $B | No | Lab's existing recognizer |
| | PCA + 1-NN | No | PCA dims chosen on validation users |
| | DTW 1-NN | No | Standard few-shot time-series baseline; run on R20/RF only |
| | Nearest class mean | No | On handcrafted features |
| | Shrinkage LDA | No | K >= 3 only; ill-posed at K = 1 |
| B. Transfer | Pretrained encoder + nearest centroid | Yes | Pretrain as (user, gesture)-ID classification on training users ("good embedding" baseline) |
| | Pretrained encoder + fine-tune on support | Yes | Chen et al. 2019 style baseline |
| C. Metric meta-learning | Prototypical Networks | Yes | |
| | Matching Networks | Yes | |
| D. Optimization meta-learning | MAML++ (dense, parameter-matched) | Yes | |
| | ANIL | Yes | Head-only adaptation |
| E. Routing (hypothesis test) | **EncoderMoE** (soft or top-k) | Yes | |
| | **Dense ensemble**: same experts, fixed uniform gates | Yes | **Most important control**: separates routing from ensembling |
| | Random fixed routing | Yes | |
| | User-level hard routing, oracle (route by clinical ability group) | Yes | Explicit user-level clustering with labels |
| | User-level hard routing, unsupervised (cluster training users, assign test user via support set) | Yes | The classical alternative EncoderMoE claims to beat |
| | MoE + ProtoNet (optional) | Yes | Tests whether routing helps outside MAML |
| F. Optional | Pretrained EMG foundation model | Yes | Only if it can be made comparable |

### 4.4 Protocol
- **Splits:** user-level k-fold cross-validation (e.g., 4 folds x 8 test users), so every user is a test user exactly once. Needed for per-user analyses (H4) and more robust than one fixed split. Inner validation users within each training fold for HPO and early stopping.
- **Episodes:** N in {3, 5, 10}, K in {1, 3, 5}; queries = all remaining repetitions of the sampled gestures. Generate episode lists once, save to disk, and have every method read the same files.
- **Meta-training way:** fix one meta-training N for all episodic methods (or train per N for all of them). Same rule for every method.
- **HPO:** equal budget per method (same number of trials), validation users only. Report the budget in the paper.
- **Seeds:** at least 5 for every trained method.
- **Unit of analysis:** the user. Average episodes within each user, then compare across users.
- **Statistics:** paired Wilcoxon signed-rank over users (or a mixed-effects model with a user random effect), Holm correction within the primary family, effect sizes, and 95% bootstrap CIs over users.

**Primary comparisons (pre-specified; everything else is exploratory):**

| ID | Comparison | Question |
|---|---|---|
| P1 | EncoderMoE vs. best per-user baseline ($B or DTW) | Does cross-user learning help at all? |
| P2 | EncoderMoE vs. dense ensemble | Is the gain from routing or from ensembling? |
| P3 | EncoderMoE vs. parameter-matched MAML++ | Does MoE add anything over dense MAML? |
| P4 | EncoderMoE vs. user-level hard routing (oracle and unsupervised) | Does sample-level beat user-level structure? |
| P5 | Best model on 2 kHz vs. 20 Hz | Was 20 Hz throwing away information? |

### 4.5 Compute tiers
The full grid (methods x representations x modalities x seeds x folds) is too large. Episode evaluation is cheap once models are trained, so the cost is in training runs.

- **Tier 1 (full grid):** all per-user methods (group A), all representations and modalities. Cheap.
- **Tier 2:** all methods on the primary representation, with EMG+IMU and the best single modality.
- **Tier 3:** representation and modality sweeps for the top 3 trained methods only.

---

## 5. Phase 3: Mechanism analysis (H3, H4)

Run on trained EncoderMoE models (every fold and seed).

### 5.1 H3: Does routing use motifs?
- **Routing decomposition:** how well gate vectors predict user ID, prompt label, ability group, and motif membership (from Phase 1), using mutual information or a linear probe. Compare each to a label-permutation null. H3 predicts motif > user ID.
- **Routing vs. input similarity:** for cross-user pairs of executions, correlate execution distance (Phase 1) with gate-vector distance. H3 predicts a positive correlation.
- **Causal gate ablations (test time):** replace gates with (a) uniform, (b) gates shuffled across samples, (c) another user's gates. H3 predicts accuracy drops, mostly for users with high motif overlap.
- **Expert knockout:** drop each expert in turn and record the per-user accuracy drop. Test whether the knockout pattern aligns with M (Mantel test).
- **Sparse routing caveat:** with top-k routing, low entropy is forced by construction, so entropy alone is not evidence of specialization. Report expert load and collapse, and rely on the tests above.

### 5.2 H4: Does MoE benefit track sharing?
- Per-user gain = EncoderMoE minus dense ensemble (and minus MAML++).
- Spearman correlation across all 32 users between gain and each user's motif-overlap score (row sum of M). Pre-registered direction: positive.
- Also report gain by ability group with CIs.

---

## 6. Outcomes and what the paper becomes

| Result | Paper framing |
|---|---|
| Per-user methods ($B / DTW) match or beat all cross-user methods | Honest domain finding: for user-defined gestures, per-user templates suffice at these K. Story shifts to *when* cross-user learning helps (higher N, lower K, 2 kHz). MoE claims dropped. |
| Cross-user methods win, but EncoderMoE is about equal to the dense ensemble | Contribution is meta-learned adaptation. MoE described as ensembling. Specialization claims dropped. |
| EncoderMoE beats the ensemble, gate ablations hurt, routing tracks motifs, H4 positive | Sample-level hypothesis supported. Strongest version of the paper. |
| User-level oracle routing about equals EncoderMoE | Ability labels carry the structure; unsupervised user clustering failed for other reasons (e.g., small N). |
| H1 fails | No motif premise. MoE claims limited to capacity. |

The 2 kHz vs. 20 Hz comparison and the modality results are reported regardless of which row applies.

---

## 7. Implementation rules

- **Pre-registration:** before Phase 2 and 3, commit a dated document to the repo with H1 to H4, predicted directions, P1 to P5, the match definition, and the statistical tests.
- **Shared inputs:** splits, episode lists and HPO configs are saved, versioned and read by every method. No method generates its own episodes.
- **Explicit validity grid:** every (method, representation, modality) cell is declared valid or invalid in config. Requesting an undeclared or invalid cell raises an error. No default substitutions.
- **Logging:** save per-episode predictions and gate vectors so every analysis can be recomputed without retraining.

---

## 8. Timeline (target: early January submission)

| Approx. end date | Milestone |
|---|---|
| Oct 30 | Phase 0 ($B reproduction), pre-registration doc, split and episode files |
| Nov 13 | Phase 1 (H1, H2) done; 2 kHz pipelines (RF, RC) built and verified |
| Dec 4 | Phase 2 Tiers 1 and 2 done |
| Dec 18 | Phase 2 Tier 3 and Phase 3 (H3, H4) done |
| Jan 8 | Draft complete, framed by the outcome table in Section 6 |

Holidays will compress December. If time runs short, cut Tier 3 and the optional methods, not the dense-ensemble control or Phase 1.

---

## 9. Open questions to resolve this week

1. Exact protocol behind the $B 80% figure.
2. Repetitions per gesture per user (sets the maximum K and the query size).
3. How the clinical ability groups were defined, and whether an IMU range-of-motion proxy is acceptable to Momona.
4. Whether segmentation is identical across the 20 Hz and 2 kHz data.
5. Whether DTW is computationally feasible on the chosen representation (use RF or downsampled sequences if not).
