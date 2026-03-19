# Train/test split: on repetition, NOT user!
# Subject specific! There is no mutli-user model. Each user trains their own KNN/Proto
# PCA (if used) is fit per user per episode
# proto_encoded allows for a pretrained multi-subject encoder to be applied followed by subject-specific Proto

"""
eval_knn_proto.py
=================
Subject-specific few-shot evaluation: KNN and Prototypical Networks.
Everything runs in pure PyTorch — no sklearn dependency.

═══════════════════════════════════════════════════════════════════════
Evaluation Modes
═══════════════════════════════════════════════════════════════════════
  1. knn_raw        : KNN on raw features (flattened or feature_fn output)
  2. knn_pca        : KNN after channel-wise PCA dimensionality reduction
  3. proto_raw      : ProtoNet on raw features (per-class mean)
  4. proto_pca      : ProtoNet after channel-wise PCA
  5. proto_encoded  : ProtoNet on outputs of a pretrained neural encoder

═══════════════════════════════════════════════════════════════════════
Subject-Specificity
═══════════════════════════════════════════════════════════════════════
  Every user is evaluated completely independently. There is no shared
  model or cross-user information for methods 1-4. Each user's support
  set defines that user's prototypes/neighbours and has zero effect on
  any other user's predictions.

  The only thing shared across users is the neural encoder (method 5),
  which is pretrained on a separate set of users and then frozen. The
  encoder generalises; the prototypes personalise. This is the correct
  decomposition for subject-adaptive BCIs.

  There is NO user-level train/test split in this file because KNN and
  ProtoNet have no learnable cross-user parameters. The only split that
  matters here is the within-user repetition split (support vs. query).

═══════════════════════════════════════════════════════════════════════
Support / Query Split Modes
═══════════════════════════════════════════════════════════════════════
  Fixed-rep (default, deployment-realistic):
      Support = first k_shot repetition indices (e.g. rep 1 for 1-shot)
      Query   = all remaining repetition indices
      Simulates real BCI calibration: collect 1 trial per gesture, then
      immediately classify everything else.

  Explicit override:
      config["support_reps"] = [1]
      config["query_reps"]   = [2, 3, ..., 10]

  Random-shuffle (matches MetaGestureDataset / MAML pipeline):
      config["use_fixed_rep_split"] = False
      All reps shuffled per gesture; first k_shot -> support, rest -> query.
      Use for apples-to-apples comparisons against MAML numbers.

═══════════════════════════════════════════════════════════════════════
Feature Engineering Hook (feature_fn)
═══════════════════════════════════════════════════════════════════════
  An optional callable that maps a raw (C, T) time-series tensor to a
  1-D feature vector. It is applied AFTER augmentation and BEFORE any
  PCA or classifier. This is the correct order:

      raw time-series -> augment -> feature_fn -> PCA / KNN / ProtoNet

  Augmenting engineered features directly (e.g. adding noise to an RMS
  value) is rarely meaningful; augmentation should always operate on
  the time-series before feature extraction.

  Usage:
      def my_features(x: torch.Tensor) -> torch.Tensor:
          # x: (C, T) float tensor
          rms  = x.pow(2).mean(dim=-1).sqrt()     # (C,)
          mav  = x.abs().mean(dim=-1)              # (C,)
          return torch.cat([rms, mav])             # (2*C,)

      results = run_all_conditions(tensor_dict, config, feature_fn=my_features)

  If feature_fn is None (default), samples are flattened: (C, T) -> (C*T,).

═══════════════════════════════════════════════════════════════════════
PCA Design  (channel-wise / spatial PCA, paper-style)
═══════════════════════════════════════════════════════════════════════
  Three formulations exist for PCA on multi-channel time series:

  (1) Sample-wise PCA  [NOT used]:
      Flatten each trial to (C*T,), stack N trials -> (N, C*T).
      PCA caps at min(N-1, C*T). For 1-shot 10-way: only 9 components,
      regardless of C or T, because your 10 data points only span a
      9-dimensional subspace of the C*T-dimensional feature space.

  (2) Channel-wise PCA + mean-pool over time  [previous version, not used]:
      Covariance is (C, C); components are spatial filters.
      After projecting (C,T) -> (nPC,T), mean-pool -> (nPC,) per trial.
      Cap: C-1 = up to 15 for 16-ch EMG. Compact but discards time.

  (3) Channel-wise PCA + keep time  [THIS implementation, matches paper]:
      Same spatial filter as (2). After projecting (C,T) -> (nPC,T),
      FLATTEN -> (nPC*T,) per trial. Full temporal structure preserved.
      Cap: C-1 per modality, independent of k_shot.
      Output dimension: nPC * T (e.g. 8 * 200 = 1600 for EMG).

  This matches the referenced paper exactly: their D matrix is (C, T),
  covariance is (C,C), eigenvectors are spatial filters, output per
  trial is (nPC * T,) flattened. They use nPC=50 on 88 channels.

  PCA fit on support ONLY. Same filters applied to query (no leakage).
  For k_shot > 1, per-trial covariances are averaged before eigen-decomp.

  Multi-modality (EMG + IMU):
    Separate PCA per modality; projected features concatenated.
    Prevents high-variance IMU from dominating the shared filter space.

  config["pca_n_components"] = 8   # components PER MODALITY (cap: C-1)

═══════════════════════════════════════════════════════════════════════
Support Augmentation (optional)
═══════════════════════════════════════════════════════════════════════
  Augmentation is applied to the raw (C, T) time-series BEFORE feature_fn
  and BEFORE PCA. Query samples are never augmented.

  Each support sample gets n_copies augmented variants appended.
    ProtoNet: augmented copies fold into the per-class prototype mean.
    KNN:      augmented copies become extra support neighbours.

  config["aug_support"]   = True
  config["aug_n_copies"]  = 4
  config["aug_noise_std"] = 0.05    # Gaussian noise (relative to signal std)
  config["aug_max_shift"] = 4       # max circular temporal shift (samples)
  config["aug_ch_drop"]   = 0.10    # per-channel zero-out probability

═══════════════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════════════
  As a script:
      python eval_knn_proto.py --tensor_dict path/to/tensor_dict.pkl

  As a library — raw baselines:
      from eval_knn_proto import build_config, run_all_conditions
      results = run_all_conditions(tensor_dict, build_config())

  With engineered features:
      def rms_mav(x):   # x: (C, T) -> (2C,)
          return torch.cat([x.pow(2).mean(-1).sqrt(), x.abs().mean(-1)])
      results = run_all_conditions(tensor_dict, config, feature_fn=rms_mav)

  With a neural encoder:
      results = run_all_conditions(tensor_dict, config, encoder=model.backbone)

  Both together (features for raw/PCA tracks, encoder for encoded track):
      results = run_all_conditions(tensor_dict, config,
                                   feature_fn=rms_mav, encoder=model.backbone)

═══════════════════════════════════════════════════════════════════════
Config Keys Reference
═══════════════════════════════════════════════════════════════════════
  eval_PIDs           : list of participant IDs to evaluate
  n_way               : int, number of gesture classes (default 10)
  knn_metric          : "euclidean" | "cosine" (default "euclidean")
  seed                : int for reproducible RNG (default 42)
  use_imu             : bool, include IMU modality (default False)
  use_fixed_rep_split : bool, fixed-rep split vs random shuffle (default True)
  support_reps        : list[int] | None, explicit 1-indexed support rep indices
  query_reps          : list[int] | None, explicit 1-indexed query rep indices
  pca_n_components    : int, PCA components PER MODALITY (default 8)
  aug_support         : bool, augment support set (default False)
  aug_n_copies        : int, augmented copies per support sample (default 4)
  aug_noise_std       : float (default 0.05)
  aug_max_shift       : int   (default 4)
  aug_ch_drop         : float (default 0.10)
  device              : "cuda" | "cpu"
"""

import pickle
import random
from typing import Optional, Callable, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SECTION 1: Distance Primitives (pure PyTorch)
# =============================================================================

def pairwise_euclidean_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Squared Euclidean distance between every row of a and every row of b.
    a : (N, D)  b : (M, D)  ->  returns (N, M)

    Uses the expansion ||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a*b^T).
    This avoids materialising the (N, M, D) difference tensor and is
    ~10x faster than looping. Clamped to suppress floating-point negatives.
    """
    a_sq = (a * a).sum(dim=1, keepdim=True)      # (N, 1)
    b_sq = (b * b).sum(dim=1, keepdim=True).t()  # (1, M)
    ab   = a @ b.t()                              # (N, M)
    return (a_sq + b_sq - 2 * ab).clamp(min=0.0)


def pairwise_cosine_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance (1 - cosine_similarity) between every row of a and b.
    a : (N, D)  b : (M, D)  ->  returns (N, M)

    Values in [0, 2]: 0 = identical direction, 1 = orthogonal, 2 = opposite.
    """
    a_n = F.normalize(a, p=2, dim=1)
    b_n = F.normalize(b, p=2, dim=1)
    return 1.0 - (a_n @ b_n.t())

def pairwise_l1(a, b):
    # a: (N, D), b: (M, D) -> (N, M)
    return (a.unsqueeze(1) - b.unsqueeze(0)).abs().sum(dim=-1)

def pairwise_l2(a, b):
    # a: (N, D), b: (M, D) -> (N, M)
    #return torch.linalg.norm(a - b)  # --> This is 1D vector version
    return torch.linalg.norm(a - b, dim=1)  # --> This is 2D matrix version --> We return a distance for EACH corresponding pair of rows in a and b

def pairwise_dist(a: torch.Tensor, b: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """Dispatcher: returns (N, M) pairwise distance matrix."""
    if metric.upper() == "COSINE":
        return pairwise_cosine_dist(a, b) 
    elif metric.upper() == "EUC" or metric == "EUCLIDEAN":
        return  pairwise_euclidean_sq(a, b)
    elif metric.upper() == "L1":
        return pairwise_l1(a, b) 
    elif metric.upper() == "L2":
        return pairwise_l2(a, b) 
    else:
        raise ValueError(f"metric {metric} not recognized!")


# =============================================================================
# SECTION 2: Core Classifiers (pure PyTorch)
# =============================================================================

def knn_classify(
    support_feats:  torch.Tensor,   # (N_support, D)
    support_labels: torch.Tensor,   # (N_support,)  -- integer class indices 0..n_classes-1
    query_feats:    torch.Tensor,   # (N_query, D)
    k:              int,
    metric:         str = "euclidean",
    n_classes:      int = 10,
) -> torch.Tensor:
    """
    Pure-PyTorch k-Nearest-Neighbour classifier.
    Returns predicted class indices: (N_query,) long tensor.

    Algorithm:
        1. Compute (N_query, N_support) pairwise distance matrix.
        2. For each query, find the k closest support indices (topk, smallest).
        3. Accumulate votes: scatter neighbour labels into (N_query, n_classes).
        4. Return argmax (majority vote).

    Fully vectorised — no Python loops over queries.

    Sanity check: with k_shot=1, knn_classify(k=1) and proto_classify must
    produce identical predictions (one neighbour == one prototype). If they
    disagree at k=1, something is wrong upstream.
    """
    dist          = pairwise_dist(query_feats, support_feats, metric)           # (Q, S)
    k_eff         = min(k, support_feats.size(0))
    _, nn_indices = dist.topk(k_eff, dim=1, largest=False, sorted=True)         # (Q, k)
    nn_labels     = support_labels[nn_indices]                                   # (Q, k)

    votes = torch.zeros(query_feats.size(0), n_classes,
                        device=query_feats.device, dtype=torch.float32)
    votes.scatter_add_(dim=1, index=nn_labels,
                       src=torch.ones_like(nn_labels, dtype=torch.float32))
    return votes.argmax(dim=1)   # (Q,)


def proto_classify(
    support_feats:  torch.Tensor,   # (N_support, D)
    support_labels: torch.Tensor,   # (N_support,)  -- integer class indices
    query_feats:    torch.Tensor,   # (N_query, D)
    metric:         str = "euclidean",
    n_classes:      int = 10,
) -> torch.Tensor:
    """
    Prototypical Network classifier.
    Returns predicted class indices: (N_query,) long tensor.

    Algorithm:
        1. Prototype_c = mean of all support embeddings with label c.
        2. Classify each query as the class whose prototype is nearest.

    vs. KNN: ProtoNet compresses within-class info into one representative
    point before comparing. With k=1 they are identical. With k>1, ProtoNet's
    averaging is more robust to outlier support samples than KNN's voting.

    With a learned encoder, the embedding space is explicitly shaped so
    within-class points cluster and between-class prototypes separate —
    this is the core ProtoNet training objective.
    """
    D          = support_feats.size(1)
    prototypes = torch.zeros(n_classes, D,
                             device=support_feats.device, dtype=support_feats.dtype)
    for c in range(n_classes):
        mask = (support_labels == c)
        if mask.any():
            prototypes[c] = support_feats[mask].mean(dim=0)

    dist = pairwise_dist(query_feats, prototypes, metric)   # (Q, n_classes)
    return dist.argmin(dim=1)


# =============================================================================
# SECTION 3: Encoders
# =============================================================================

# ── Background: three PCA formulations for multi-channel time series ─────────
#
# Given a single trial of shape (C, T):
#
# Formulation 1 — Sample-wise PCA  [NOT used here; included for reference]
#   Flatten all N support trials to (N, C*T), run PCA over the N-sample dim.
#   Output: (N, k) where k <= min(N-1, C*T).
#   Problem: for 1-shot 10-way, N=10, so k <= 9 regardless of C or T.
#   The rank of a (10, 3200) matrix is at most 10 — the extra 3190 dimensions
#   of feature space are unreachable because you only have 10 data points to
#   span them. After mean-centering you lose one more degree of freedom -> 9.
#   This is why sample count caps components: PCA can only find directions
#   that exist in the subspace your data actually occupies.
#
# Formulation 2 — Channel-wise PCA + mean-pool over time  [previous version]
#   Stack all (N*T) timestep-vectors as rows: (N*T, C).
#   PCA covariance is (C, C); components are spatial filters over channels.
#   Cap: min(N*T - 1, C) = up to 16 for EMG. Not capped by k_shot.
#   Project each trial (C, T) -> (nPC, T), then mean-pool -> (nPC,).
#   Pro: compact vector per trial. Con: temporal structure discarded.
#
# Formulation 3 — Channel-wise PCA + keep time  [what the paper does; used here]
#   Same spatial filter fit as Formulation 2 (covariance is still C×C).
#   After projecting: (C, T) -> (nPC, T), then FLATTEN -> (nPC * T,).
#   Output per trial: (nPC * T,). Cap on nPC: C-1 (single-trial covariance).
#   Pro: temporal structure fully preserved, matches paper's approach.
#   Con: output dimension is nPC * T (e.g. 8 * 200 = 1600), not as compact.
#
# The paper fits the covariance on a single TEMPLATE TRIAL (their "support"),
# so their cap is C-1 regardless of k_shot. That is what we implement below.
# For k_shot > 1 we average the per-trial covariances before computing the SVD,
# which is a natural generalisation (each extra support sample refines the
# spatial filter estimate).
# ─────────────────────────────────────────────────────────────────────────────

def _fit_spatial_pca(
    support_emg: torch.Tensor,   # (N_support, C, T)
    n_components: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a channel-wise (spatial) PCA on the support set.

    Constructs the C×C covariance matrix by averaging per-trial covariances,
    then returns the top-n_components eigenvectors (spatial filters) and the
    channel mean used for centering.

    Why average per-trial covariances (rather than stacking all timesteps)?
        Stacking (N*T, C) and taking one big covariance is equivalent but
        weights all timesteps equally regardless of which trial they came from.
        Averaging per-trial covariances weights each trial equally, which is
        more appropriate when trials may have different lengths or when you
        want the filter to represent the average trial structure.
        For fixed-length trials the two approaches give the same result.

    Returns
    -------
    components : (n_components, C)  -- rows are spatial filters (eigenvectors)
    mean       : (C,)               -- channel mean for centering
    """
    N, C, T = support_emg.shape
    n_components = min(n_components, C - 1)
    if n_components < 1:
        n_components = 1   # always return at least 1 component

    # Channel mean across all support trials and all timesteps
    # Shape: (C,)
    mean = support_emg.mean(dim=(0, 2))

    # Average covariance: mean over trials of (1/(T-1)) * X_centred @ X_centred^T
    # Each X_centred is (C, T); covariance is (C, C)
    cov = torch.zeros(C, C, device=support_emg.device, dtype=support_emg.dtype)
    for i in range(N):
        x = support_emg[i] - mean.unsqueeze(-1)   # (C, T)  -- centre each trial
        cov += (x @ x.t()) / (T - 1)
    cov /= N   # average across trials

    # Eigendecomposition of the symmetric covariance matrix.
    # torch.linalg.eigh is faster and more numerically stable than eig for
    # symmetric matrices. Returns eigenvalues in ASCENDING order, so we flip.
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)   # both (C,), (C, C)
    except Exception:
        # Fallback: identity projection (no dimensionality reduction)
        components = torch.eye(C, device=support_emg.device)[:n_components]
        return components, mean

    # Flip to descending order (largest variance first)
    eigenvectors = eigenvectors.flip(dims=[1])   # (C, C) -- columns are eigenvectors
    components   = eigenvectors[:, :n_components].t()   # (n_components, C)

    return components, mean


def _apply_spatial_pca(
    emg_batch:  torch.Tensor,   # (N, C, T)
    components: torch.Tensor,   # (n_components, C)
    mean:       torch.Tensor,   # (C,)
) -> torch.Tensor:
    """
    Apply pre-fit spatial filters to a batch of trials and return flattened vectors.

    Pipeline per trial:
        (C, T) -- centre --> (C, T) -- project channels --> (nPC, T) -- flatten --> (nPC*T,)

    This matches the paper's approach: the full temporal structure is preserved
    after the channel projection, and the result is flattened for KNN/ProtoNet.

    Returns : (N, n_components * T)
    """
    # Centre: subtract channel mean; mean shape (C,) -> broadcast to (C, 1)
    x_centred = emg_batch - mean.unsqueeze(-1)              # (N, C, T)
    # Project channels: components (nPC, C) x x_centred (N, C, T)
    # einsum 'kc,nct->nkt': for each trial n and timestep t, project C channels -> nPC dims
    projected = torch.einsum('kc,nct->nkt', components, x_centred)   # (N, nPC, T)
    # Flatten nPC*T into a single feature vector per trial
    N = emg_batch.size(0)
    return projected.reshape(N, -1)                         # (N, nPC*T)


def pca_encode(
    support_emg:        torch.Tensor,
    query_emg:          torch.Tensor,
    support_imu:        Optional[torch.Tensor],
    query_imu:          Optional[torch.Tensor],
    emg_pca_dims:       int  = 8,
    imu_pca_dims:       int  = 16,
    shared_pca:         bool = False,
    pca_level:          str  = "global",  # Options: "global", "per_class", "per_sample"
    support_labels:     Optional[torch.Tensor] = None, 
    n_classes:          int  = 10,  # This should match n_way I think?
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Channel-wise spatial PCA encode.
    
    pca_level="global": 
        Returns (sup_proj, qry_proj) -> standard projected feature tensors.
        
    pca_level="per_class": 
        Returns (class_dists, class_dists_sup) -> (N, n_classes) L1-distance matrices.
        Classify queries via argmin over the class dimension.
        
    pca_level="per_sample":
        Returns (sample_dists, sample_dists_sup) -> (N, N_support) L1-distance matrices.
        Classify queries via 1-NN (argmin over all support samples, then map to class).
    """
    assert pca_level in ["global", "per_class", "per_sample"], f"Unknown pca_level: {pca_level}"

    # 1. Setup labels and counts based on the chosen PCA level
    if pca_level == "per_class":
        assert support_labels is not None, "support_labels required for per_class_pca"
        labels = support_labels
        n_models = n_classes
    elif pca_level == "per_sample":
        # Hack: Treat every sample as its own "class" to reuse the per-class logic
        N_sup = support_emg.shape[0]
        labels = torch.arange(N_sup, device=support_emg.device)
        n_models = N_sup

    # 2. Define a helper to process a single data stream (EMG, IMU, or Concatted)
    def _process_branch(sup: torch.Tensor, qry: torch.Tensor, dims: int):
        if pca_level == "global":
            components, mean = _fit_spatial_pca(sup, dims)
            return (_apply_spatial_pca(sup, components, mean),
                    _apply_spatial_pca(qry, components, mean))
        else:
            # Applies to both "per_class" and "per_sample"
            models = _fit_per_class_pcas(sup, labels, dims, n_models)
            return (_per_class_pca_distances(sup, models),
                    _per_class_pca_distances(qry, models))

    # 3. Apply routing logic based on `shared_pca`
    if shared_pca:
        assert support_imu is not None, "IMU required for shared_pca"
        assert emg_pca_dims == imu_pca_dims, "Dims must match if projecting into a shared space"
        
        sup_cat = torch.cat([support_emg, support_imu], dim=1)
        qry_cat = torch.cat([query_emg,  query_imu],  dim=1)
        
        sup_res, qry_res = _process_branch(sup_cat, qry_cat, emg_pca_dims)
        return qry_res, sup_res  # Returning query first to match your original per_class return signature

    else:
        # Separate PCA models for EMG and IMU
        sup_emg_res, qry_emg_res = _process_branch(support_emg, query_emg, emg_pca_dims)
        
        if support_imu is None:
            return qry_emg_res, sup_emg_res

        sup_imu_res, qry_imu_res = _process_branch(support_imu, query_imu, imu_pca_dims)
        
        # Combine the separate modalities based on the PCA level
        if pca_level == "global":
            # Global returns feature vectors, so we concatenate them
            sup_proj = torch.cat([sup_emg_res, sup_imu_res], dim=-1)
            qry_proj = torch.cat([qry_emg_res, qry_imu_res], dim=-1)
            return sup_proj, qry_proj
        else:
            # Per-class/sample returns distance matrices, so we sum the distances
            # (Because L1 distance is additive over partitions)
            return (qry_emg_res + qry_imu_res), (sup_emg_res + sup_imu_res)


@torch.no_grad()
def neural_encode_batch(
    encoder:   nn.Module,
    emg_batch: torch.Tensor,            # (N, C, T)
    imu_batch: Optional[torch.Tensor],  # (N, C_imu, T) | None
    device:    torch.device,
) -> torch.Tensor:
    """
    Run a pretrained neural encoder and return feature vectors: (N, D).

    Handles two calling conventions:
      (a) encoder(emg, imu) returns (features, aux)  -> uses features
      (b) encoder(emg, imu) returns features directly

    Set to eval() mode before inference (suppresses BatchNorm stat updates
    and Dropout). Left in eval() after — caller should manage model state.
    """
    encoder.eval()
    out   = encoder(emg_batch.to(device),
                    imu_batch.to(device) if imu_batch is not None else None)
    feats = out[0] if isinstance(out, (tuple, list)) else out
    return feats.float()


# =============================================================================
# SECTION 4: Feature Extraction (raw flatten OR feature_fn)
# =============================================================================

def extract_features(
    emg_batch:  torch.Tensor,              # (N, C, T)
    imu_batch:  Optional[torch.Tensor],    # (N, C_imu, T) | None
    feature_fn: Optional[Callable],        # (C, T) -> (D,) | None
    use_imu:    bool,
) -> torch.Tensor:
    """
    Convert a batch of raw (C, T) time-series tensors into feature vectors.

    Two modes:
        feature_fn=None  : flatten the time series to (C*T,), then optionally
                           concatenate flattened IMU. Simple but very high-D.
        feature_fn given : apply it to each sample independently, then optionally
                           concatenate IMU features. The feature_fn operates on
                           the raw TIME SERIES before any reduction, which is the
                           correct place for feature engineering (see module docs).

    If use_imu=True and imu_batch is not None, IMU features are appended to EMG
    features along the last dimension. Both use the same feature_fn (or flatten).

    Returns : (N, D_total) float tensor
    """
    N = emg_batch.size(0)

    if feature_fn is None:
        emg_feats = emg_batch.reshape(N, -1).float()   # (N, C*T)
    else:
        emg_feats = torch.stack([feature_fn(emg_batch[i]) for i in range(N)]).float()

    if not use_imu or imu_batch is None:
        return emg_feats

    if feature_fn is None:
        imu_feats = imu_batch.reshape(N, -1).float()
    else:
        imu_feats = torch.stack([feature_fn(imu_batch[i]) for i in range(N)]).float()

    return torch.cat([emg_feats, imu_feats], dim=-1)   # (N, D_emg + D_imu)


# =============================================================================
# SECTION 5: Data Augmentation Helpers
# =============================================================================

def _aug_gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """
    Add zero-mean Gaussian noise scaled to std * signal_std.
    Simulates electrode contact noise and inter-trial EMG amplitude variation.
    x : (C, T)
    """
    return x + torch.randn_like(x) * (x.std() * std + 1e-8)


def _aug_temporal_shift(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    """
    Circular shift along the time axis by a random amount in [-max_shift, max_shift].
    Simulates slight timing differences in gesture onset between trials.
    x : (C, T)
    """
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(x, shifts=shift, dims=-1)


def _aug_channel_dropout(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """
    Zero out entire channels independently with probability drop_prob.
    Simulates poor electrode contact or transient signal loss.
    x : (C, T) -- mask shape (C, 1) broadcasts over time.
    """
    mask = (torch.rand(x.size(0), 1, device=x.device) > drop_prob).float()
    return x * mask


def augment_sample(x: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Apply the full augmentation pipeline to a single (C, T) EMG tensor.
    Applied in order: noise -> temporal shift -> channel dropout.

    IMPORTANT: This operates on the RAW TIME SERIES before feature_fn or PCA.
    Augmenting engineered features directly (e.g. adding noise to an RMS
    value) is only meaningful if the augmentation has a clear physical
    interpretation in that feature space. When in doubt, augment here.

    Query samples must NEVER be passed through this function at eval time.
    """
    x = _aug_gaussian_noise(x,  config.get("aug_noise_std", 0.05))
    x = _aug_temporal_shift(x,  config.get("aug_max_shift", 4))
    x = _aug_channel_dropout(x, config.get("aug_ch_drop",   0.10))
    return x


def expand_support_with_aug(
    sup_emg:    torch.Tensor,   # (N_sup, C, T)
    sup_labels: torch.Tensor,   # (N_sup,)
    config:     dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create n_copies augmented variants of each support sample and append them.

    Returned shapes:
        sup_emg    : (N_sup * (1 + n_copies), C, T)
        sup_labels : (N_sup * (1 + n_copies),)

    Originals are always at indices 0..N_sup-1; augmented copies follow.
    Augmented copies carry the same label as their source sample.

    Effect on classifiers:
        ProtoNet -- augmented copies fold into the per-class prototype mean,
                    making the prototype more robust to signal variability.
                    This is the more principled use.
        KNN      -- augmented copies become extra support neighbours,
                    effectively inflating k without adding real data.

    NOTE: Augmentation happens here on the raw time series, BEFORE feature_fn
    or PCA. The subsequent pipeline sees augmented (C, T) tensors and processes
    them identically to the real support samples.
    """
    n_copies        = int(config.get("aug_n_copies", 4))
    aug_emg_list    = [sup_emg]
    aug_labels_list = [sup_labels]

    for _ in range(n_copies):
        batch_aug = torch.stack([
            augment_sample(sup_emg[i], config) for i in range(sup_emg.size(0))
        ])
        aug_emg_list.append(batch_aug)
        aug_labels_list.append(sup_labels)

    return torch.cat(aug_emg_list, dim=0), torch.cat(aug_labels_list, dim=0)


# =============================================================================
# SECTION 6: Episode Builder
# =============================================================================

def build_episode(
    user_data:                dict,
    k_shot:                   int,
    n_way:                    int,
    rng:                      random.Random,
    available_gesture_classes: List[int],        # enc_gesture_ids (0-indexed class labels)
    support_reps:             Optional[List[int]], # 1-indexed rep numbers; None -> random shuffle
    query_reps:               Optional[List[int]], # 1-indexed rep numbers; None -> random shuffle
) -> Tuple[dict, dict]:
    """
    Build one support / query episode for a single user.

    FIXED vs original:
      - `all_reps` renamed to `available_gesture_classes` — these are
        enc_gesture_ids (0-indexed class labels, 0..N_classes-1), NOT
        repetition indices. Callers must pass gesture class IDs here.
      - `all_reps` parameter removed entirely — rep pool is now inferred
        from the data itself (entry['rep_indices']), so it can't drift
        out of sync with what was actually saved.
      - Added label consistency assert (enc_gesture_id key == stored label).
      - Added gesture_ids (strings) to output for debugging.

    UNCHANGED vs original:
      - support_reps / query_reps remain 1-indexed rep numbers — this is
        correct since Gesture_Num is 1-indexed in the source data.
      - label_map assigns episode-local labels 0..n_way-1. This is correct
        for both KNN and meta-learning; distances don't depend on label
        magnitude and the relative assignment is consistent within an episode.
      - Random-shuffle mode (None) still works as before.

    Gesture selection:
        n_way gestures randomly sampled from available_gesture_classes,
        sorted for deterministic local label assignment (0..n_way-1).

    Split modes
    -----------
    Fixed-rep (support_reps / query_reps are lists):
        Rep numbers are 1-indexed and converted to 0-indexed array positions.
        k_shot is implicitly len(support_reps).
        Default: rep 1 = calibration support, reps 2-10 = query.

    Random-shuffle (support_reps / query_reps are None):
        All reps shuffled per gesture; first k_shot -> support, rest -> query.
        Matches MetaGestureDataset / MAML pipeline behavior exactly.

    Returns
    -------
    support, query : dicts with keys
        "emg"        : (N, C_emg, T) float tensor  (channel-first)
        "imu"        : (N, C_imu, T) float tensor | None
        "labels"     : (N,) long tensor, episode-local 0..n_way-1
        "gesture_ids": List[str], human-readable labels (for debugging)
    """
    # ── Gesture class selection ──────────────────────────────────────────
    # available_gesture_classes contains enc_gesture_ids (class labels, 0-indexed)
    # We filter to only those that exist for this specific user
    valid_classes = sorted([g for g in available_gesture_classes if g in user_data])
    if len(valid_classes) < n_way:
        raise ValueError(
            f"Only {len(valid_classes)} gesture classes available "
            f"({valid_classes}), need n_way={n_way}."
        )

    # Sort after sampling for deterministic local label assignment
    selected  = sorted(rng.sample(valid_classes, n_way))
    label_map = {enc_gid: i for i, enc_gid in enumerate(selected)}

    sup_e, sup_i_list, sup_l, sup_gids = [], [], [], []
    qry_e, qry_i_list, qry_l, qry_gids = [], [], [], []

    for enc_gid in selected:
        entry = user_data[enc_gid]

        # ── Sanity check: stored label must match the key ────────────────
        assert entry['enc_gesture_id'] == enc_gid, (
            f"Label mismatch: key={enc_gid} but stored "
            f"enc_gesture_id={entry['enc_gesture_id']}"
        )

        lbl       = label_map[enc_gid]
        emg_data  = entry["emg"]           # (N_reps, T, C_emg) — time-last added in _pack
        imu_data  = entry.get("imu")       # (N_reps, T, C_imu) or None
        gid_str   = entry.get("gesture_id", str(enc_gid))
        n_reps    = emg_data.shape[0]

        # ── Rep index resolution ─────────────────────────────────────────
        # support_reps / query_reps are 1-indexed Gesture_Num values.
        # Convert to 0-indexed positions into the tensor's first dimension.
        if support_reps is not None:
            # Clamp to valid range; warn if a requested rep is missing
            sup_idx = [r - 1 for r in support_reps if 0 <= r - 1 < n_reps]
            qry_idx = [r - 1 for r in query_reps   if 0 <= r - 1 < n_reps]
            if len(sup_idx) != len(support_reps):
                print(f"  Warning: enc_gid {enc_gid} — some support_reps out of range "
                      f"(n_reps={n_reps}, requested={support_reps})")
            if len(qry_idx) != len(query_reps):
                print(f"  Warning: enc_gid {enc_gid} — some query_reps out of range "
                      f"(n_reps={n_reps}, requested={query_reps})")
        else:
            idxs = list(range(n_reps))
            rng.shuffle(idxs)
            sup_idx = idxs[:k_shot]
            qry_idx = idxs[k_shot:]

        for idx_list, e_list, i_list, l_list, gid_list in [
            (sup_idx, sup_e, sup_i_list, sup_l, sup_gids),
            (qry_idx, qry_e, qry_i_list, qry_l, qry_gids),
        ]:
            for i in idx_list:
                e_list.append(emg_data[i])          # (T, C_emg)
                if imu_data is not None:
                    i_list.append(imu_data[i])      # (T, C_imu)
                l_list.append(lbl)
                gid_list.append(gid_str)

    def _pack(e_list, i_list, l_list) -> dict:
        """Stack sample lists and enforce (N, C, T) channel-first layout."""
        emg = torch.stack(e_list).float()
        if emg.dim() == 3 and emg.shape[-1] in [16, 72]:   # (N,T,C) -> (N,C,T)
            emg = emg.permute(0, 2, 1).contiguous()
        imu = None
        if i_list:
            imu = torch.stack(i_list).float()
            if imu.dim() == 3 and imu.shape[-1] in [16, 72]:
                imu = imu.permute(0, 2, 1).contiguous()
        return {"emg": emg, "imu": imu,
                "labels": torch.tensor(l_list, dtype=torch.long)}

    return _pack(sup_e, sup_i_list, sup_l), _pack(qry_e, qry_i_list, qry_l)


# =============================================================================
# SECTION 7: Per-User Evaluation (single shot condition)
# =============================================================================

def eval_one_user(
    user_data:                 dict,
    k_shot:                    int,
    n_way:                     int,
    support_reps:              Optional[List[int]],
    query_reps:                Optional[List[int]],
    available_gesture_classes: List[int],
    rng:                       random.Random,
    device:                    torch.device,
    metric:                    str,
    encoder:                   Optional[nn.Module],
    feature_fn:                Optional[Callable],
    use_imu:                   bool,
    aug_support:               bool,
    emg_pca_dims:              int,
    imu_pca_dims:              int,
    shared_pca:                bool,
    config:                    dict,
) -> dict:
    """
    Run all evaluation modes for a single user under one shot condition.
    support_reps/query_reps (1-indexed rep numbers) are passed directly and unchanged.
    """

    # ── 1. Episode Building ──────────────────────────────────────────────────
    support, query = build_episode(
        user_data                 = user_data,
        k_shot                    = k_shot,
        n_way                     = n_way,
        rng                       = rng,
        available_gesture_classes = available_gesture_classes,
        support_reps              = support_reps,
        query_reps                = query_reps,
    )

    sup_emg    = support["emg"].to(device)
    qry_emg    = query["emg"].to(device)
    sup_labels = support["labels"].to(device)
    qry_labels = query["labels"].to(device)
    sup_imu    = support["imu"].to(device) if (use_imu and support["imu"] is not None) else None
    qry_imu    = query["imu"].to(device)   if (use_imu and query["imu"] is not None)   else None

    n_support_orig = sup_emg.size(0)

    # ── 2. Support Augmentation ──────────────────────────────────────────────
    aug_sup_emg, aug_sup_labels = sup_emg, sup_labels
    # Note: If using IMU, you'd need to augment IMU here as well if config allows
    if aug_support:
        aug_sup_emg, aug_sup_labels = expand_support_with_aug(sup_emg, sup_labels, config)

    # ── 3. Raw Feature Track (KNN & ProtoNet) ────────────────────────────────
    sup_flat = extract_features(aug_sup_emg, sup_imu, feature_fn, use_imu)
    qry_flat = extract_features(qry_emg,     qry_imu, feature_fn, use_imu)

    knn_raw_acc = (knn_classify(sup_flat, aug_sup_labels, qry_flat, k=1, 
                                metric=metric, n_classes=n_way) == qry_labels).float().mean().item()
    
    proto_raw_acc = (proto_classify(sup_flat, aug_sup_labels, qry_flat, 
                                    metric=metric, n_classes=n_way) == qry_labels).float().mean().item()

    # ── 4. PCA Track ─────────────────────────────────────────────────────────
    pca_level = config.get("pca_level", "global") # "global", "per_class", or "per_sample"
    
    # Initialize all PCA metrics to None
    knn_pca_acc = proto_pca_acc = dollar_B_acc = per_sample_acc = None

    # Call the updated pca_encode
    pca_out_1, pca_out_2 = pca_encode(
        aug_sup_emg, qry_emg,
        sup_imu if use_imu else None,
        qry_imu if use_imu else None,
        emg_pca_dims, imu_pca_dims, shared_pca,
        pca_level      = pca_level,
        support_labels = aug_sup_labels,
        n_classes      = n_way
    )

    if pca_level == "global":
        # Returns (sup_proj, qry_proj)
        sup_pca, qry_pca = pca_out_1, pca_out_2
        
        knn_pca_pred = knn_classify(sup_pca, aug_sup_labels, qry_pca, k=1, metric=metric, n_classes=n_way)
        knn_pca_acc  = (knn_pca_pred == qry_labels).float().mean().item()

        proto_pca_pred = proto_classify(sup_pca, aug_sup_labels, qry_pca, metric=metric, n_classes=n_way)
        proto_pca_acc  = (proto_pca_pred == qry_labels).float().mean().item()

    elif pca_level == "per_class":
        # Returns (qry_dists, sup_dists) where dists are (N, n_classes)
        qry_dists = pca_out_1
        dollar_B_pred = qry_dists.argmin(dim=1)
        dollar_B_acc  = (dollar_B_pred == qry_labels).float().mean().item()

    elif pca_level == "per_sample":
        # Returns (qry_dists, sup_dists) where dists are (N, n_support)
        qry_dists = pca_out_1
        # 1-Nearest Neighbor logic: find closest sample index, then map to that sample's label
        best_sample_idx = qry_dists.argmin(dim=1)
        per_sample_pred = aug_sup_labels[best_sample_idx]
        per_sample_acc  = (per_sample_pred == qry_labels).float().mean().item()

    # ── 5. Neural Track ──────────────────────────────────────────────────────
    proto_enc_acc = None
    if encoder is not None:
        sup_enc = neural_encode_batch(encoder, aug_sup_emg, sup_imu, device)
        qry_enc = neural_encode_batch(encoder, qry_emg,     qry_imu, device)
        proto_enc_pred = proto_classify(sup_enc, aug_sup_labels, qry_enc, metric=metric, n_classes=n_way)
        proto_enc_acc  = (proto_enc_pred == qry_labels).float().mean().item()

    return {
        "knn_raw":       knn_raw_acc,
        "proto_raw":     proto_raw_acc,
        "knn_pca":       knn_pca_acc,       # Only populated if level="global"
        "proto_pca":     proto_pca_acc,     # Only populated if level="global"
        "dollar_B":      dollar_B_acc,      # Only populated if level="per_class"
        "per_sample_pca":per_sample_acc,    # Only populated if level="per_sample"
        "proto_encoded": proto_enc_acc,
        "n_support":     n_support_orig,
        "n_query":       qry_emg.size(0),
    }

# =============================================================================
# SECTION 8: Single-Shot-Condition Runner (across all users)
# =============================================================================

def run_one_shot_condition(
    tensor_dict: dict,
    config:      dict,
    k_shot:      int,
    encoder:     Optional[nn.Module]  = None,
    feature_fn:  Optional[Callable]   = None,
    verbose:     bool = True,
) -> dict:
    """
    Run all evaluation modes for a given k_shot across all eval_PIDs.
    """

    device         = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    n_way          = int(config.get("n_way",         10))
    metric         = config.get("knn_metric",       "euclidean")
    seed           = int(config.get("seed",         42))
    use_imu        = bool(config.get("use_imu",     False))
    aug_support    = bool(config.get("aug_support", False))
    emg_pca_dims   = int(config.get("emg_pca_dims", 8))
    imu_pca_dims   = int(config.get("imu_pca_dims", 8))
    shared_pca     = bool(config.get("shared_pca",  False))
    pca_level      = config.get("pca_level", "global") # "global", "per_class", "per_sample"
    
    eval_pids      = config.get("eval_PIDs", config.get("val_PIDs", []))
    available_gesture_classes = config["available_gesture_classes"]

    # ── Support / query rep split ─────────────────────────────────────────────
    if "support_reps" in config and "query_reps" in config:
        support_reps = config["support_reps"]
        query_reps   = config["query_reps"]
        split_mode   = f"fixed-explicit  sup={support_reps}"
    elif config.get("use_fixed_rep_split", True):
        all_rep_indices = config.get("all_rep_indices", list(range(1, 11)))
        sorted_reps     = sorted(all_rep_indices)
        support_reps    = sorted_reps[:k_shot]
        query_reps      = sorted_reps[k_shot:]
        split_mode      = (f"fixed-auto  sup={support_reps}  "
                           f"qry={query_reps[:3]}{'...' if len(query_reps) > 3 else ''}")
    else:
        support_reps = query_reps = None
        split_mode   = "random-shuffle"

    rng = random.Random(seed)

    if verbose:
        fn_name = feature_fn.__name__ if feature_fn else "flatten (C*T)"
        print(f"\n{'='*65}")
        print(f"  {k_shot}-shot  {n_way}-way  |  metric={metric}  |  pca_level={pca_level}")
        print(f"  split      : {split_mode}")
        print(f"  feature_fn : {fn_name}  |  shared_pca: {shared_pca}")
        if encoder: print(f"  encoder    : {type(encoder).__name__}")
        print(f"{'='*65}")

    # Initialize accuracy lists for all possible tracks
    accs: Dict[str, List[float]] = {
        "knn_raw": [], "proto_raw": [],
        "knn_pca": [], "proto_pca": [],
        "dollar_B": [], "per_sample_pca": [],
        "proto_encoded": [],
    }
    per_user = {}

    for pid in eval_pids:
        if pid not in tensor_dict: continue
        user_data = tensor_dict[pid]
        available = [g for g in available_gesture_classes if g in user_data]
        if len(available) < n_way: continue

        try:
            res = eval_one_user(
                user_data=user_data, k_shot=k_shot, n_way=n_way,
                support_reps=support_reps, query_reps=query_reps,
                available_gesture_classes=available_gesture_classes,
                rng=rng, device=device, metric=metric,
                encoder=encoder, feature_fn=feature_fn,
                use_imu=use_imu, aug_support=aug_support,
                emg_pca_dims=emg_pca_dims, imu_pca_dims=imu_pca_dims,
                shared_pca=shared_pca, config=config
            )

            # ── Dynamic Accumulation ──────────────────────────────────────────
            accs["knn_raw"].append(res["knn_raw"])
            accs["proto_raw"].append(res["proto_raw"])
            
            if pca_level == "global":
                accs["knn_pca"].append(res["knn_pca"])
                accs["proto_pca"].append(res["proto_pca"])
                pca_log = f"knn_pca={res['knn_pca']*100:.1f}%  proto_pca={res['proto_pca']*100:.1f}%"
            elif pca_level == "per_class":
                accs["dollar_B"].append(res["dollar_B"])
                pca_log = f"$B={res['dollar_B']*100:.1f}%"
            elif pca_level == "per_sample":
                accs["per_sample_pca"].append(res["per_sample_pca"])
                pca_log = f"sample_pca={res['per_sample_pca']*100:.1f}%"

            if encoder and res["proto_encoded"] is not None:
                accs["proto_encoded"].append(res["proto_encoded"])

            per_user[pid] = res

            if verbose:
                enc_str = f"  enc={res['proto_encoded']*100:.1f}%" if encoder else ""
                print(f"  PID {str(pid):>4} | knn={res['knn_raw']*100:.1f}%  {pca_log}{enc_str}")

        except Exception as e:
            if verbose: print(f"  [ERROR] PID {pid}: {e}")
            continue

    # ── Summary Table ─────────────────────────────────────────────────────────
    def _summarise(key: str) -> dict:
        data = accs[key]
        if not data: return {}
        t = torch.tensor(data)
        return {
            "mean_acc": t.mean().item(),
            "std_acc":  t.std().item(),
            "all_accs": data,
        }

    results = {key: _summarise(key) for key in accs}

    if verbose:
        print(f"\n  {'Method':<30}  {'Mean':>8}  {'Std':>7}  {'Users':>6}")
        print(f"  {'─'*30}  {'─'*8}  {'─'*7}  {'─'*6}")
        
        # Determine which rows to show based on pca_level
        report_rows = [("knn_raw", "KNN (raw)"), ("proto_raw", "ProtoNet (raw)")]
        if pca_level == "global":
            report_rows += [("knn_pca", "KNN (Global PCA)"), ("proto_pca", "ProtoNet (Global PCA)")]
        elif pca_level == "per_class":
            report_rows.append(("dollar_B", "$B (Per-Class PCA)"))
        elif pca_level == "per_sample":
            report_rows.append(("per_sample_pca", "1-NN (Per-Sample PCA)"))
            
        if encoder:
            report_rows.append(("proto_encoded", f"ProtoNet ({type(encoder).__name__})"))

        for key, label in report_rows:
            r = results[key]
            if r:
                print(f"  {label:<30}  {r['mean_acc']*100:>7.2f}%  {r['std_acc']*100:>6.2f}%  {len(r['all_accs']):>6}")
    
    return results


# =============================================================================
# SECTION 9: Multi-Shot Runner + Comparison Table
# =============================================================================

def run_all_conditions(
    tensor_dict:     dict,
    config:          dict,
    shot_conditions: List[int]           = [1, 3, 5],
    encoder:         Optional[nn.Module] = None,
    feature_fn:      Optional[Callable]  = None,
    verbose:         bool                = True,
) -> Dict[int, dict]:
    """
    Run run_one_shot_condition for every k in shot_conditions and print a
    unified comparison table across all methods and shot levels.

    Parameters
    ----------
    tensor_dict     : {pid: {gesture: {"emg": Tensor, "imu": Tensor|None, ...}}}
    config          : see build_config() and module docstring
    shot_conditions : list of k values to evaluate (default [1, 3, 5])
    encoder         : optional pretrained nn.Module backbone
    feature_fn      : optional callable (C,T) -> (D,) for raw/PCA tracks
                      If None, trials are flattened to (C*T,).
    verbose         : print per-user and summary tables

    Returns
    -------
    all_results : {k_shot: result_dict}
    """
    all_results = {}
    for k in shot_conditions:
        print(f"\n{'#'*65}")
        print(f"#  {k}-SHOT EVALUATION")
        print(f"{'#'*65}")
        all_results[k] = run_one_shot_condition(
            tensor_dict, config, k_shot=k,
            encoder=encoder, feature_fn=feature_fn, verbose=verbose,
        )

    method_rows = [
        ("knn_raw",   "KNN (raw)"),
        ("knn_pca",   "KNN (PCA)"),
        ("proto_raw", "ProtoNet (raw)"),
        ("proto_pca", "ProtoNet (PCA)"),
    ]
    if encoder is not None:
        method_rows.append(("proto_encoded", f"ProtoNet ({type(encoder).__name__})"))

    n_way  = config.get("n_way", 10)
    chance = 100.0 / n_way
    col_w  = 20

    print(f"\n\n{'='*75}")
    print(f"  FINAL COMPARISON TABLE  ({n_way}-way | chance={chance:.1f}%)")
    if feature_fn is not None:
        print(f"  feature_fn: {feature_fn.__name__}")
    print(f"{'='*75}")

    header = f"  {'Method':<30}"
    for k in shot_conditions:
        header += f"  {f'{k}-shot':>{col_w}}"
    print(header)
    print(f"  {'─'*30}" + "".join([f"  {'─'*col_w}"] * len(shot_conditions)))

    for key, label in method_rows:
        row = f"  {label:<30}"
        for k in shot_conditions:
            r = all_results[k].get(key, {})
            cell = f"{r['mean_acc']*100:.2f}+/-{r['std_acc']*100:.2f}%" if r else "N/A"
            row += f"  {cell:>{col_w}}"
        print(row)

    print(f"{'='*75}\n")
    return all_results


# =============================================================================
# SECTION 10: Config Builder + Convenience Wrappers
# =============================================================================

def build_config(
    eval_pids:           List  = None,
    n_way:               int   = 10,
    metric:              str   = "euclidean",
    seed:                int   = 42,
    use_imu:             bool  = False,
    aug_support:         bool  = False,
    aug_n_copies:        int   = 4,
    emg_pca_dims:        int   = 8,       # <--- Changed
    imu_pca_dims:        int   = 8,       # <--- Added
    shared_pca:          bool  = False,   # <--- Added
    use_fixed_rep_split: bool  = True,
    device:              str   = None,
) -> dict:
    """
    Build a standard config dict with sensible defaults.

    Defaults: 24 pretrain users (PIDs 1-24), 10 reps (1-indexed),
    10-way, euclidean distance, no augmentation, 8 PCA dims per modality.

    After building, override any key directly:
        config = build_config()
        config["support_reps"] = [1]
        config["query_reps"]   = list(range(2, 11))
        config["pca_n_components"] = 16
    """
    return {
        "eval_PIDs":           eval_pids,
        "n_way":               n_way,
        "knn_metric":          metric,
        "seed":                seed,
        "use_imu":             use_imu,
        "aug_support":         aug_support,
        "aug_n_copies":        aug_n_copies,
        "aug_noise_std":       0.05,
        "aug_max_shift":       4,
        "aug_ch_drop":         0.10,
        "emg_pca_dims":        emg_pca_dims,  # <--- Changed
        "imu_pca_dims":        imu_pca_dims,  # <--- Added
        "shared_pca":          shared_pca,    # <--- Added
        "use_fixed_rep_split": use_fixed_rep_split,
        "device":              device or ("cuda" if torch.cuda.is_available() else "cpu"),
    }


def eval_from_path(
    tensor_dict_path: str,
    config:           dict,
    encoder:          Optional[nn.Module] = None,
    feature_fn:       Optional[Callable]  = None,
    shot_conditions:  List[int]           = [1, 3, 5],
) -> Dict[int, dict]:
    """
    Load tensor_dict from pickle and run the full multi-shot evaluation.
    Convenience wrapper for notebooks / quick one-liners.

    Examples:
        # Baseline
        results = eval_from_path("data/tensor_dict.pkl", build_config())

        # With engineered features
        def rms_mav(x):
            return torch.cat([x.pow(2).mean(-1).sqrt(), x.abs().mean(-1)])
        results = eval_from_path("data/tensor_dict.pkl", build_config(), feature_fn=rms_mav)

        # With neural encoder
        results = eval_from_path("data/tensor_dict.pkl", build_config(), encoder=model.backbone)
    """
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
        tensor_dict = full_dict['data']
    return run_all_conditions(tensor_dict, config,
                              shot_conditions=shot_conditions,
                              encoder=encoder, feature_fn=feature_fn)


# =============================================================================
# SECTION 11: Example Feature Functions
# =============================================================================
# These are provided as starting points. Pass any of these (or your own)
# as feature_fn to run_all_conditions / eval_from_path.
# All functions take (C, T) float tensor and return a 1-D float tensor.

def feat_rms_mav(x: torch.Tensor) -> torch.Tensor:
    """RMS and MAV per channel: (C, T) -> (2*C,)"""
    rms = x.pow(2).mean(dim=-1).sqrt()
    mav = x.abs().mean(dim=-1)
    return torch.cat([rms, mav])


def feat_rms_mav_var(x: torch.Tensor) -> torch.Tensor:
    """RMS, MAV, variance per channel: (C, T) -> (3*C,)"""
    rms = x.pow(2).mean(dim=-1).sqrt()
    mav = x.abs().mean(dim=-1)
    var = x.var(dim=-1)
    return torch.cat([rms, mav, var])


def feat_waveform_length(x: torch.Tensor) -> torch.Tensor:
    """Waveform length (sum of abs diff) per channel: (C, T) -> (C,)"""
    return x.diff(dim=-1).abs().sum(dim=-1)


def feat_full_td(x: torch.Tensor) -> torch.Tensor:
    """
    Full time-domain feature set (commonly used in EMG literature):
    RMS, MAV, variance, waveform length, zero-crossings.
    (C, T) -> (5*C,)
    """
    rms = x.pow(2).mean(-1).sqrt()
    mav = x.abs().mean(-1)
    var = x.var(-1)
    wl  = x.diff(dim=-1).abs().sum(-1)
    # Zero crossings: count sign changes
    zc  = ((x[:, :-1] * x[:, 1:]) < 0).float().sum(-1)
    return torch.cat([rms, mav, var, wl, zc])

##################################################################
# Plotting to determine how many PCs to take

import matplotlib.pyplot as plt

def plot_pca_variance(emg_data: Optional[torch.Tensor] = None, 
                      imu_data: Optional[torch.Tensor] = None, 
                      save_path: str = "pca_variance_analysis.png"):
    """
    Plots cumulative explained variance for EMG, IMU, and Shared modalities.
    Expects tensors of shape (N, C, T).
    """
    def get_cum_variance(x):
        # Flatten N and T to get (N*T, C) as done in your _fit_spatial_pca
        N, C, T = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, C)
        
        # Center the data
        x_centered = x_flat - torch.mean(x_flat, dim=0)
        
        # Calculate Covariance and Eigenvalues
        cov = torch.matmul(x_centered.T, x_centered) / (x_centered.shape[0] - 1)
        eigenvalues, _ = torch.linalg.eigh(cov)
        
        # Sort descending and calculate cumulative ratio
        eigenvalues = torch.flip(eigenvalues, dims=[0])
        var_ratio = eigenvalues / torch.sum(eigenvalues)
        return torch.cumsum(var_ratio, dim=0).cpu().numpy()

    plots = []
    if emg_data is not None: plots.append(('EMG', emg_data))
    if imu_data is not None: plots.append(('IMU', imu_data))
    if emg_data is not None and imu_data is not None:
        shared = torch.cat([emg_data, imu_data], dim=1)
        plots.append(('Shared (EMG+IMU)', shared))

    if not plots:
        print("No data provided to plot."); return

    fig, axes = plt.subplots(1, len(plots), figsize=(6 * len(plots), 5))
    if len(plots) == 1: axes = [axes]

    for i, (label, data) in enumerate(plots):
        cum_var = get_cum_variance(data)
        comps = range(1, len(cum_var) + 1)
        
        axes[i].plot(comps, cum_var, 'o-', markersize=4, label='Variance')
        axes[i].set_title(f'{label} PCA Analysis')
        axes[i].set_xlabel('Number of Components')
        axes[i].set_ylabel('Cumulative Explained Variance')
        
        # Add reference lines
        axes[i].axhline(y=0.95, color='r', linestyle='--', alpha=0.6, label='95%')
        axes[i].axhline(y=0.90, color='g', linestyle='--', alpha=0.6, label='90%')
        
        # Find exact 95% threshold for the user
        idx_95 = (cum_var >= 0.95).argmax() + 1
        axes[i].annotate(f'95% @ {idx_95} PCs', xy=(idx_95, 0.95), xytext=(idx_95, 0.80),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4))
        
        axes[i].grid(True, which='both', linestyle='--', alpha=0.5)
        axes[i].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Analysis plot saved to {save_path}")

# Example Usage:
# plot_pca_variance(my_emg_tensor, my_imu_tensor)

###########################################################
# $B per-class PCA... requires some finagling...

def _fit_per_class_pcas(
    support:    torch.Tensor,   # (N_support, C, T)
    labels:     torch.Tensor,   # (N_support,)
    n_components: int,
    n_classes:  int,
) -> List[dict]:
    """
    Fit one spatial PCA per class on the support set.

    Returns a list of length n_classes, each entry:
        {
          "components": (n_pc, C)  -- spatial filters (rows = filters)
          "mean":       (C,)       -- channel mean for centering
          "template":   (n_pc*T,)  -- mean projected support vector for this class
        }

    "template" is the per-class prototype in PCA space — the thing you
    compare each query against. For k-shot > 1 it's the mean of all
    k projected support trials, which is $B's natural k-shot extension.
    """
    C, T = support.shape[1], support.shape[2]
    models = []

    for c in range(n_classes):
        mask  = (labels == c)
        sup_c = support[mask]           # (k_shot, C, T)

        components, mean = _fit_spatial_pca(sup_c, n_components)   # existing fn
        # components: (n_pc, C),  mean: (C,)

        # Project each support trial and average -> template
        projected = _apply_spatial_pca(sup_c, components, mean)    # (k_shot, n_pc*T)
        template  = projected.mean(dim=0)                          # (n_pc*T,)

        models.append({
            "components": components,   # (n_pc, C)
            "mean":       mean,         # (C,)
            "template":   template,     # (n_pc*T,)
        })

    return models


def _per_class_pca_distances(
    query:      torch.Tensor,   # (N_query, C, T)
    pca_models: List[dict],     # output of _fit_per_class_pcas
) -> torch.Tensor:
    """
    Project each query into every class's PCA space and compute L1 distance
    to that class's template.

    Returns (N_query, n_classes) distance matrix.
    Classify by argmin over classes.
    """
    N_qry     = query.shape[0]
    n_classes = len(pca_models)
    dists     = torch.zeros(N_qry, n_classes, device=query.device)

    for c, model in enumerate(pca_models):
        proj     = _apply_spatial_pca(query, model["components"], model["mean"])
        # proj: (N_qry, n_pc*T)
        dists[:, c] = (proj - model["template"].unsqueeze(0)).abs().sum(dim=-1)

    return dists   # (N_query, n_classes) — argmin gives predicted class
