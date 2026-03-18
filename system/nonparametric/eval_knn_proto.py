"""
eval_knn_proto.py
=================
Subject-specific few-shot evaluation: KNN and Prototypical Networks.
Everything runs in pure PyTorch — no sklearn dependency.

This is the single unified file combining the primitive classifiers,
episode builder, augmentation helpers, PCA encoder, and the full
multi-shot evaluation runner.

─────────────────────────────────────────────────────────────────────
Evaluation Modes
─────────────────────────────────────────────────────────────────────
  1. knn_raw        : KNN on flattened raw EMG features (input space)
  2. knn_pca        : KNN on PCA-projected features (fit on support only)
  3. proto_raw      : ProtoNet on raw EMG features (per-class mean in input space)
  4. proto_pca      : ProtoNet on PCA-projected features (fit on support only)
  5. proto_encoded  : ProtoNet on outputs of a pretrained neural encoder

─────────────────────────────────────────────────────────────────────
Setup
─────────────────────────────────────────────────────────────────────
  - One episode per user (no episodic meta-training — pure evaluation).
  - Support set : k_shot samples per class
  - Query  set  : all remaining samples (never augmented at eval time)

─────────────────────────────────────────────────────────────────────
Support / Query Split Modes
─────────────────────────────────────────────────────────────────────
  Fixed-rep (default, deployment-realistic):
      Support = first k_shot repetition indices (e.g. rep 1 for 1-shot)
      Query   = all remaining repetition indices
      This simulates real BCI calibration: collect 1 trial per gesture,
      then classify everything else immediately.

  Explicit override:
      config["support_reps"] = [1]
      config["query_reps"]   = [2, 3, ..., 10]

  Random-shuffle (matches MetaGestureDataset / MAML pipeline):
      config["use_fixed_rep_split"] = False
      All reps shuffled; first k_shot -> support, rest -> query.
      Use this for apples-to-apples comparisons against MAML numbers.

─────────────────────────────────────────────────────────────────────
Support Augmentation (optional)
─────────────────────────────────────────────────────────────────────
  Augmentation is applied to support samples only (never query).
  Each support sample gets n_copies augmented variants appended.
  For ProtoNet: augmented copies are folded into the per-class mean,
                making the prototype more robust to signal variability.
  For KNN:      augmented copies act as extra support neighbours,
                effectively increasing k without adding real data.

  config["aug_support"]   = True
  config["aug_n_copies"]  = 4       # augmented copies per support sample
  config["aug_noise_std"] = 0.05    # Gaussian noise relative to signal std
  config["aug_max_shift"] = 4       # max circular temporal shift (samples)
  config["aug_ch_drop"]   = 0.10    # per-channel zero-out probability

─────────────────────────────────────────────────────────────────────
PCA Encoder
─────────────────────────────────────────────────────────────────────
  PCA is fit on the support set only (per-user, per-episode).
  Fitting on query would be data leakage and is never done here.
  With k=1 and 10-way, support has 10 points -> PCA components are
  automatically capped at min(n_support - 1, pca_n_components).
  Even 9 components can be highly informative for separating 10 classes.

  config["pca_n_components"] = 16   # target dimensionality after PCA

─────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────
  As a script:
      python eval_knn_proto.py --tensor_dict path/to/tensor_dict.pkl

  As a library:
      from eval_knn_proto import build_config, run_all_conditions

      # Raw + PCA baselines, no neural encoder
      config  = build_config()
      results = run_all_conditions(tensor_dict, config)

      # Add a pretrained neural encoder
      results = run_all_conditions(tensor_dict, config, encoder=model.backbone)

─────────────────────────────────────────────────────────────────────
Config Keys Reference
─────────────────────────────────────────────────────────────────────
  eval_PIDs           : list of participant IDs to evaluate
  maml_reps           : list of 1-indexed repetition indices in the full pool
  n_way               : int, number of gesture classes (default 10)
  knn_metric          : "euclidean" | "cosine" (default "euclidean")
  seed                : int for reproducible RNG (default 42)
  use_imu             : bool, include IMU modality (default False)
  use_fixed_rep_split : bool, use fixed-rep split vs random shuffle (default True)
  support_reps        : list[int] | None, explicit 1-indexed support rep indices
  query_reps          : list[int] | None, explicit 1-indexed query rep indices
  pca_n_components    : int, PCA output dimension (default 16)
  aug_support         : bool, augment support set (default False)
  aug_n_copies        : int, augmented copies per support sample (default 4)
  aug_noise_std       : float (default 0.05)
  aug_max_shift       : int   (default 4)
  aug_ch_drop         : float (default 0.10)
  device              : "cuda" | "cpu"
"""

import pickle
import random
import argparse
from typing import Optional, Dict, Any, List, Tuple

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

    Uses the expansion ||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a*b^T),
    which is ~10x faster than looping and avoids materialising (N, M, D).
    Clamped to zero to suppress floating-point negatives near the diagonal.
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
    L2-normalises both matrices before the dot product for numerical stability.
    """
    a_n = F.normalize(a, p=2, dim=1)
    b_n = F.normalize(b, p=2, dim=1)
    return 1.0 - (a_n @ b_n.t())


def pairwise_dist(a: torch.Tensor, b: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """
    Dispatcher returning an (N, M) pairwise distance matrix.
    metric : "euclidean" (default) | "cosine"
    """
    if metric == "cosine":
        return pairwise_cosine_dist(a, b)
    return pairwise_euclidean_sq(a, b)


# =============================================================================
# SECTION 2: Core Classifiers (pure PyTorch)
# =============================================================================

def knn_classify(
    support_feats:  torch.Tensor,   # (N_support, D)
    support_labels: torch.Tensor,   # (N_support,)  -- integer class indices
    query_feats:    torch.Tensor,   # (N_query, D)
    k:              int,
    metric:         str = "euclidean",
    n_classes:      int = 10,
) -> torch.Tensor:
    """
    Pure-PyTorch k-Nearest-Neighbour classifier.
    Returns predicted class indices for each query sample: (N_query,).

    Algorithm:
        1. Compute (N_query, N_support) pairwise distance matrix.
        2. For each query, select the k closest support indices.
        3. Accumulate one vote per neighbour into a (N_query, n_classes) count matrix.
        4. Return argmax over classes (majority vote).

    This is fully vectorised -- no Python loops over queries.
    The scatter_add_ vote accumulation is also differentiable w.r.t. feature
    vectors if you ever want to fine-tune through the classifier.

    Note on k=1 equivalence with ProtoNet:
        When k_shot=1, each class has exactly one support sample.
        knn_classify(k=1) and proto_classify give identical predictions
        because the single neighbour IS the prototype. This is a useful
        sanity check: if they disagree at k=1, something is wrong.
    """
    dist          = pairwise_dist(query_feats, support_feats, metric)           # (Q, S)
    k_eff         = min(k, support_feats.size(0))
    _, nn_indices = dist.topk(k_eff, dim=1, largest=False, sorted=True)         # (Q, k)
    nn_labels     = support_labels[nn_indices]                                   # (Q, k)

    # Majority vote: scatter neighbour labels into a per-class count matrix
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
    Returns predicted class indices for each query sample: (N_query,).

    Algorithm:
        1. Prototype_c = mean of all support embeddings with label c.
        2. Classify each query as the class whose prototype is nearest.

    Compared to KNN, ProtoNet compresses within-class information into a
    single representative point before comparing. With k=1 they are identical.
    With k>1, ProtoNet's averaging makes it more robust to outlier support
    samples than KNN's individual-neighbour voting.

    When the encoder is learned end-to-end with episodic cross-entropy loss,
    the embedding space is explicitly shaped so that within-class points cluster
    and between-class prototypes separate -- this is the core ProtoNet insight.
    """
    D          = support_feats.size(1)
    prototypes = torch.zeros(n_classes, D,
                             device=support_feats.device, dtype=support_feats.dtype)
    for c in range(n_classes):
        mask = (support_labels == c)
        if mask.any():
            prototypes[c] = support_feats[mask].mean(dim=0)

    dist = pairwise_dist(query_feats, prototypes, metric)   # (Q, n_classes)
    return dist.argmin(dim=1)   # (Q,)


# =============================================================================
# SECTION 3: Encoders
# =============================================================================

def pca_encode(
    support_feats: torch.Tensor,   # (N_support, D_in) -- flat features
    query_feats:   torch.Tensor,   # (N_query,   D_in)
    n_components:  int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PCA dimensionality reduction, fit on support only, applied to both sets.

    Critically, PCA is fit on the SUPPORT SET ONLY. Fitting on query (or on
    the combined set) would be data leakage, since PCA would use query signal
    structure to define the projection axes.

    With k=1 and 10-way, support has 10 points in a potentially very high-
    dimensional space (e.g. 16 channels x 200 timesteps = 3200 dims). PCA
    components are automatically capped at min(n_support - 1, n_components)
    to keep the SVD well-defined. Even 9 components is often sufficient to
    separate 10 gesture classes if the signal is well-structured.

    Algorithm:
        1. Compute support mean; subtract from both support and query.
           (Query is centred using support mean -- NOT its own mean.)
        2. Thin SVD of centred support -> principal components (right singular vectors).
        3. Project both support and query onto the top-n_components components.

    Returns
    -------
    sup_proj : (N_support, n_components)
    qry_proj : (N_query,   n_components)
    """
    # Cap components: SVD requires n_components <= min(n_samples - 1, n_features)
    n_components = min(n_components, support_feats.size(0) - 1, support_feats.size(1))
    if n_components < 1:
        # Degenerate case (e.g. only 1 support sample) -- return features unchanged
        return support_feats, query_feats

    # Step 1: centre using support mean only (applying same shift to query)
    mean        = support_feats.mean(dim=0, keepdim=True)   # (1, D_in)
    sup_centred = support_feats - mean
    qry_centred = query_feats   - mean                      # same shift, NOT query mean

    # Step 2: thin SVD of centred support
    # sup_centred = U @ diag(S) @ Vt,  Vt shape: (min(N,D), D_in)
    try:
        _, _, Vt = torch.linalg.svd(sup_centred, full_matrices=False)
    except Exception:
        # If SVD fails (e.g. all-zero support after centering), return unchanged
        return support_feats, query_feats

    components = Vt[:n_components]   # (n_components, D_in) -- principal axes

    # Step 3: project onto principal components
    sup_proj = sup_centred @ components.t()   # (N_support, n_components)
    qry_proj = qry_centred @ components.t()   # (N_query,   n_components)

    return sup_proj, qry_proj


@torch.no_grad()
def neural_encode_batch(
    encoder:   nn.Module,
    emg_batch: torch.Tensor,            # (N, C, T)
    imu_batch: Optional[torch.Tensor],  # (N, C_imu, T) or None
    device:    torch.device,
) -> torch.Tensor:
    """
    Run a pretrained neural encoder and return feature vectors: (N, D).

    Handles two common backbone calling conventions:
      (a) encoder(emg, imu) returns (features, aux_output)  -> takes features[0]
      (b) encoder(emg, imu) returns features directly

    The encoder is set to eval() mode before inference. This suppresses
    BatchNorm running-stat updates and Dropout stochasticity.
    """
    encoder.eval()
    out   = encoder(emg_batch.to(device),
                    imu_batch.to(device) if imu_batch is not None else None)
    feats = out[0] if isinstance(out, (tuple, list)) else out
    return feats.float()


# =============================================================================
# SECTION 4: Data Augmentation Helpers
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
    Zero out entire EMG channels independently with probability drop_prob.
    Simulates poor electrode contact or transient signal loss.
    x : (C, T)  ->  mask shape (C, 1) broadcasts over time dimension.
    """
    mask = (torch.rand(x.size(0), 1, device=x.device) > drop_prob).float()
    return x * mask


def augment_sample(x: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Apply the full augmentation pipeline to a single (C, T) EMG tensor.
    All three augmentations are applied in sequence.
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
    Create augmented copies of every support sample and append them,
    returning an expanded (sup_emg, sup_labels) pair.

    Returned shapes:
        sup_emg    : (N_sup * (1 + n_copies), C, T)
        sup_labels : (N_sup * (1 + n_copies),)

    Augmented copies carry the same label as their source sample.
    The original samples are always kept at indices 0..N_sup-1,
    so indexing [:N_sup] always recovers the originals.

    Effect on classifiers:
        ProtoNet -- augmented copies fold into the per-class mean, making
                    the prototype more robust to noise and inter-trial variability.
                    This is the more principled use since it pools information
                    before the distance comparison.
        KNN      -- augmented copies become extra support neighbours, effectively
                    raising the pool size without adding real calibration data.
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
# SECTION 5: Episode Builder
# =============================================================================

def build_episode(
    user_data:    dict,
    k_shot:       int,
    n_way:        int,
    rng:          random.Random,
    support_reps: Optional[List[int]],   # 1-indexed rep indices; None -> random shuffle
    query_reps:   Optional[List[int]],   # 1-indexed rep indices; None -> random shuffle
    all_reps:     List[int],             # full rep pool (1-indexed)
) -> Tuple[dict, dict]:
    """
    Build one support / query episode for a single user.

    Gesture selection:
        n_way gestures are randomly sampled from those available in all_reps
        and sorted for deterministic local label assignment (0..n_way-1).

    Split modes
    -----------
    Fixed-rep (support_reps / query_reps are lists):
        Indices come directly from the specified 1-indexed repetition numbers.
        k_shot is implicitly len(support_reps).
        Deployment-realistic default: rep 1 = calibration, reps 2-10 = workload.

    Random-shuffle (support_reps / query_reps are None):
        All repetitions in all_reps are shuffled per gesture;
        first k_shot -> support, remainder -> query.
        Matches MetaGestureDataset / MAML pipeline exactly.

    Returns
    -------
    support, query : dicts with keys
        "emg"    : (N, C, T) float tensor
        "imu"    : (N, C_imu, T) float tensor | None
        "labels" : (N,) long tensor of episode-local class indices 0..n_way-1
    """
    available = sorted([g for g in all_reps if g in user_data])
    if len(available) < n_way:
        raise ValueError(f"Only {len(available)} gestures available, need {n_way}")

    selected  = sorted(rng.sample(available, n_way))
    label_map = {g: i for i, g in enumerate(selected)}

    sup_e, sup_i_list, sup_l = [], [], []
    qry_e, qry_i_list, qry_l = [], [], []

    for gest in selected:
        lbl      = label_map[gest]
        emg_data = user_data[gest]["emg"]       # (n_trials, ...)
        imu_data = user_data[gest].get("imu")
        n_trials = emg_data.shape[0]

        if support_reps is not None:
            # Fixed-rep: convert 1-indexed repetition numbers to 0-indexed positions
            sup_idx = [r - 1 for r in support_reps if 0 <= r - 1 < n_trials]
            qry_idx = [r - 1 for r in query_reps   if 0 <= r - 1 < n_trials]
        else:
            # Random-shuffle: matches MetaGestureDataset behavior
            idxs = list(range(n_trials))
            rng.shuffle(idxs)
            sup_idx = idxs[:k_shot]
            qry_idx = idxs[k_shot:]

        for idx_list, e_list, i_list, l_list in [
            (sup_idx, sup_e, sup_i_list, sup_l),
            (qry_idx, qry_e, qry_i_list, qry_l),
        ]:
            for i in idx_list:
                e_list.append(emg_data[i])
                if imu_data is not None:
                    i_list.append(imu_data[i])
                l_list.append(lbl)

    def _pack(e_list, i_list, l_list) -> dict:
        """Stack sample lists into tensors and enforce (N, C, T) channel-first layout."""
        emg = torch.stack(e_list).float()
        # Detect (N, T, C) layout: if the last dim matches known channel counts, permute
        if emg.dim() == 3 and emg.shape[-1] in [16, 72]:
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
# SECTION 6: Per-User Evaluation (single shot condition)
# =============================================================================

def eval_one_user(
    user_data:        dict,
    k_shot:           int,
    n_way:            int,
    support_reps:     Optional[List[int]],
    query_reps:       Optional[List[int]],
    all_reps:         List[int],
    rng:              random.Random,
    device:           torch.device,
    metric:           str,
    encoder:          Optional[nn.Module],
    use_imu:          bool,
    aug_support:      bool,
    pca_n_components: int,
    config:           dict,
) -> dict:
    """
    Run all evaluation modes for a single user under one shot condition.

    Returns a flat dict:
        "knn_raw"       : float accuracy
        "knn_pca"       : float accuracy
        "proto_raw"     : float accuracy
        "proto_pca"     : float accuracy
        "proto_encoded" : float accuracy | None  (None if encoder=None)
        "n_support"     : int (un-augmented support count)
        "n_support_aug" : int (support count after augmentation)
        "n_query"       : int
    """
    support, query = build_episode(
        user_data, k_shot, n_way, rng,
        support_reps, query_reps, all_reps,
    )

    sup_emg    = support["emg"].to(device)     # (N_sup, C, T)
    qry_emg    = query["emg"].to(device)       # (N_qry, C, T)
    sup_labels = support["labels"].to(device)
    qry_labels = query["labels"].to(device)
    sup_imu    = support["imu"].to(device) if (use_imu and support["imu"] is not None) else None
    qry_imu    = query["imu"].to(device)   if (use_imu and query["imu"] is not None)   else None

    n_support_orig = sup_emg.size(0)

    # ── Optional support augmentation ────────────────────────────────────────
    # Augmented copies are appended; originals remain at indices 0..n_support_orig-1.
    # Query is NEVER augmented.
    aug_sup_emg, aug_sup_labels = sup_emg, sup_labels
    if aug_support:
        aug_sup_emg, aug_sup_labels = expand_support_with_aug(sup_emg, sup_labels, config)

    # ── Flatten (C, T) -> D for raw-feature methods ──────────────────────────
    sup_flat = aug_sup_emg.reshape(aug_sup_emg.size(0), -1)   # (N_sup_aug, C*T)
    qry_flat = qry_emg.reshape(qry_emg.size(0), -1)           # (N_qry, C*T)

    # Use the full support pool as neighbours (k = all available support)
    knn_k = aug_sup_emg.size(0)

    # ── 1. KNN on raw features ───────────────────────────────────────────────
    knn_raw_pred = knn_classify(sup_flat, aug_sup_labels, qry_flat,
                                k=knn_k, metric=metric, n_classes=n_way)
    knn_raw_acc  = (knn_raw_pred == qry_labels).float().mean().item()

    # ── 2. ProtoNet on raw features ──────────────────────────────────────────
    proto_raw_pred = proto_classify(sup_flat, aug_sup_labels, qry_flat,
                                    metric=metric, n_classes=n_way)
    proto_raw_acc  = (proto_raw_pred == qry_labels).float().mean().item()

    # ── 3 & 4. PCA projection -> KNN and ProtoNet ────────────────────────────
    # PCA is fit on (possibly augmented) support features; same projection applied to query.
    # n_components is capped inside pca_encode to keep the SVD well-defined.
    sup_pca, qry_pca = pca_encode(sup_flat, qry_flat, n_components=pca_n_components)

    knn_pca_pred = knn_classify(sup_pca, aug_sup_labels, qry_pca,
                                k=knn_k, metric=metric, n_classes=n_way)
    knn_pca_acc  = (knn_pca_pred == qry_labels).float().mean().item()

    proto_pca_pred = proto_classify(sup_pca, aug_sup_labels, qry_pca,
                                    metric=metric, n_classes=n_way)
    proto_pca_acc  = (proto_pca_pred == qry_labels).float().mean().item()

    # ── 5. ProtoNet on neural encoder outputs ────────────────────────────────
    proto_enc_acc = None
    if encoder is not None:
        sup_enc = neural_encode_batch(encoder, aug_sup_emg, sup_imu, device)
        qry_enc = neural_encode_batch(encoder, qry_emg,    qry_imu, device)
        proto_enc_pred = proto_classify(sup_enc, aug_sup_labels, qry_enc,
                                        metric=metric, n_classes=n_way)
        proto_enc_acc  = (proto_enc_pred == qry_labels).float().mean().item()

    return {
        "knn_raw":       knn_raw_acc,
        "knn_pca":       knn_pca_acc,
        "proto_raw":     proto_raw_acc,
        "proto_pca":     proto_pca_acc,
        "proto_encoded": proto_enc_acc,
        "n_support":     n_support_orig,
        "n_support_aug": aug_sup_emg.size(0),
        "n_query":       qry_emg.size(0),
    }


# =============================================================================
# SECTION 7: Single-Shot-Condition Runner (across all users)
# =============================================================================

def run_one_shot_condition(
    tensor_dict: dict,
    config:      dict,
    k_shot:      int,
    encoder:     Optional[nn.Module] = None,
    verbose:     bool = True,
) -> dict:
    """
    Run all evaluation modes for a given k_shot across all eval_PIDs.

    Support / query split is determined from config (see module docstring).
    Users with insufficient gestures or trials are skipped with a warning.

    Returns
    -------
    results : dict with one key per method ("knn_raw", "knn_pca", "proto_raw",
              "proto_pca", "proto_encoded"), each containing:
        {
          "per_user" : {pid: acc_float},
          "mean_acc" : float,
          "std_acc"  : float,
          "all_accs" : [float, ...]   # same order as valid eval_PIDs
        }
    """
    device           = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    n_way            = int(config.get("n_way",            10))
    metric           = config.get("knn_metric",           "euclidean")
    seed             = int(config.get("seed",             42))
    use_imu          = bool(config.get("use_imu",         False))
    aug_support      = bool(config.get("aug_support",     False))
    pca_n_components = int(config.get("pca_n_components", 16))
    all_reps         = config["maml_reps"]
    eval_pids        = config.get("eval_PIDs", config.get("val_PIDs", []))

    if not eval_pids:
        raise ValueError("config must contain 'eval_PIDs' (or 'val_PIDs') -- got empty list.")

    # ── Determine support / query rep split ──────────────────────────────────
    # Priority: (1) explicit lists > (2) fixed-auto > (3) random shuffle
    if "support_reps" in config and "query_reps" in config:
        support_reps = config["support_reps"]
        query_reps   = config["query_reps"]
        split_mode   = f"fixed-explicit (sup={support_reps})"
    elif config.get("use_fixed_rep_split", True):
        sorted_reps  = sorted(all_reps)
        support_reps = sorted_reps[:k_shot]
        query_reps   = sorted_reps[k_shot:]
        preview      = query_reps[:3]
        ellipsis     = "..." if len(query_reps) > 3 else ""
        split_mode   = f"fixed-auto (sup={support_reps}, qry={preview}{ellipsis})"
    else:
        support_reps = None
        query_reps   = None
        split_mode   = "random-shuffle (matches MAML pipeline)"

    # Isolated RNG: does not perturb global random state, so calling this
    # function cannot affect reproducibility of your training runs.
    rng = random.Random(seed)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  {k_shot}-shot  {n_way}-way  |  metric={metric}")
        print(f"  split   : {split_mode}")
        print(f"  aug     : {aug_support}" +
              (f"  (n_copies={config.get('aug_n_copies',4)})" if aug_support else ""))
        print(f"  PCA dims: {pca_n_components}  |  device={device}")
        print(f"  encoder : {type(encoder).__name__ if encoder else 'None'}")
        print(f"  users   : {len(eval_pids)}")
        print(f"{'='*65}")

    # Per-method accumulation
    accs: Dict[str, List[float]] = {
        "knn_raw": [], "knn_pca": [],
        "proto_raw": [], "proto_pca": [],
        "proto_encoded": [],
    }
    per_user: Dict[Any, dict] = {}

    for pid in eval_pids:
        if pid not in tensor_dict:
            if verbose:
                print(f"  [WARN] PID {pid} not in tensor_dict -- skipping.")
            continue

        user_data = tensor_dict[pid]
        available = [g for g in all_reps if g in user_data]

        if len(available) < n_way:
            if verbose:
                print(f"  [WARN] PID {pid}: only {len(available)} gestures (need {n_way}) -- skipping.")
            continue

        min_trials = min(user_data[g]["emg"].shape[0] for g in available[:n_way])
        if min_trials < k_shot + 1:
            if verbose:
                print(f"  [WARN] PID {pid}: only {min_trials} trial(s) per gesture "
                      f"(need >= {k_shot + 1}) -- skipping.")
            continue

        try:
            res = eval_one_user(
                user_data        = user_data,
                k_shot           = k_shot,
                n_way            = n_way,
                support_reps     = support_reps,
                query_reps       = query_reps,
                all_reps         = all_reps,
                rng              = rng,
                device           = device,
                metric           = metric,
                encoder          = encoder,
                use_imu          = use_imu,
                aug_support      = aug_support,
                pca_n_components = pca_n_components,
                config           = config,
            )
        except Exception as e:
            if verbose:
                print(f"  [ERROR] PID {pid}: {e} -- skipping.")
            continue

        for key in ["knn_raw", "knn_pca", "proto_raw", "proto_pca"]:
            accs[key].append(res[key])
        if encoder is not None and res["proto_encoded"] is not None:
            accs["proto_encoded"].append(res["proto_encoded"])

        per_user[pid] = res

        if verbose:
            enc_str = (f"  enc={res['proto_encoded']*100:.1f}%"
                       if encoder and res["proto_encoded"] is not None else "")
            aug_str = (f"  aug_sup={res['n_support_aug']}" if aug_support else "")
            print(f"  PID {str(pid):>4} | "
                  f"knn={res['knn_raw']*100:.1f}%  "
                  f"knn_pca={res['knn_pca']*100:.1f}%  "
                  f"proto={res['proto_raw']*100:.1f}%  "
                  f"proto_pca={res['proto_pca']*100:.1f}%"
                  f"{enc_str}  "
                  f"[sup={res['n_support']}, qry={res['n_query']}]"
                  f"{aug_str}")

    def _summarise(key: str) -> dict:
        """Aggregate per-user accuracies for one method into summary statistics."""
        a = accs[key]
        if not a:
            return {}
        t = torch.tensor(a)
        return {
            "per_user": {pid: per_user[pid][key]
                         for pid in eval_pids
                         if pid in per_user and per_user[pid].get(key) is not None},
            "mean_acc": t.mean().item(),
            "std_acc":  t.std().item(),
            "all_accs": a,
        }

    results = {key: _summarise(key) for key in accs}

    if verbose:
        chance  = 100.0 / n_way
        n_users = len(accs["knn_raw"])
        rows = [
            ("knn_raw",   "KNN (raw)"),
            ("knn_pca",   "KNN (PCA)"),
            ("proto_raw", "ProtoNet (raw)"),
            ("proto_pca", "ProtoNet (PCA)"),
        ]
        if encoder is not None:
            rows.append(("proto_encoded", f"ProtoNet ({type(encoder).__name__})"))
        print(f"\n  {'Method':<30}  {'Mean':>8}  {'Std':>7}  {'Users':>6}")
        print(f"  {'─'*30}  {'─'*8}  {'─'*7}  {'─'*6}")
        for key, label in rows:
            r = results[key]
            if r:
                print(f"  {label:<30}  {r['mean_acc']*100:>7.2f}%  "
                      f"{r['std_acc']*100:>6.2f}%  {n_users:>6}")
        print(f"  Chance = {chance:.1f}%")

    return results


# =============================================================================
# SECTION 8: Multi-Shot Runner + Comparison Table
# =============================================================================

def run_all_conditions(
    tensor_dict:     dict,
    config:          dict,
    shot_conditions: List[int]           = [1, 3, 5],
    encoder:         Optional[nn.Module] = None,
    verbose:         bool                = True,
) -> Dict[int, dict]:
    """
    Run run_one_shot_condition for each k in shot_conditions and print a
    unified comparison table across all methods and shot levels.

    Returns
    -------
    all_results : {k_shot (int): result_dict}
        result_dict has the same structure as run_one_shot_condition's output.

    Typical usage:
        config      = build_config()
        all_results = run_all_conditions(tensor_dict, config)

        # With a pretrained neural encoder:
        all_results = run_all_conditions(tensor_dict, config, encoder=model.backbone)
    """
    all_results = {}
    for k in shot_conditions:
        print(f"\n{'#'*65}")
        print(f"#  {k}-SHOT EVALUATION")
        print(f"{'#'*65}")
        all_results[k] = run_one_shot_condition(
            tensor_dict, config, k_shot=k,
            encoder=encoder, verbose=verbose,
        )

    # ── Final comparison table ────────────────────────────────────────────────
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
# SECTION 9: Config Builder + Convenience Wrappers
# =============================================================================

def build_config(
    eval_pids:           List  = None,
    maml_reps:           List  = None,
    n_way:               int   = 10,
    metric:              str   = "euclidean",
    seed:                int   = 42,
    use_imu:             bool  = False,
    aug_support:         bool  = False,
    aug_n_copies:        int   = 4,
    pca_n_components:    int   = 16,
    use_fixed_rep_split: bool  = True,
    device:              str   = None,
) -> dict:
    """
    Build a standard config dict with sensible defaults.

    Defaults assume: 24 pretrain users (PIDs 1-24), 10 repetitions (1-indexed),
    10-way classification, euclidean distance, no augmentation, PCA to 16 dims.

    You can override any key after building:
        config = build_config()
        config["support_reps"] = [1]              # always use rep 1 as support
        config["query_reps"]   = list(range(2, 11))
        config["pca_n_components"] = 32
    """
    return {
        "eval_PIDs":           eval_pids  if eval_pids  is not None else list(range(1, 25)),
        "maml_reps":           maml_reps  if maml_reps  is not None else list(range(1, 11)),
        "n_way":               n_way,
        "knn_metric":          metric,
        "seed":                seed,
        "use_imu":             use_imu,
        "aug_support":         aug_support,
        "aug_n_copies":        aug_n_copies,
        "aug_noise_std":       0.05,
        "aug_max_shift":       4,
        "aug_ch_drop":         0.10,
        "pca_n_components":    pca_n_components,
        "use_fixed_rep_split": use_fixed_rep_split,
        "device":              device or ("cuda" if torch.cuda.is_available() else "cpu"),
    }


def eval_from_path(
    tensor_dict_path: str,
    config:           dict,
    encoder:          Optional[nn.Module] = None,
    shot_conditions:  List[int]           = [1, 3, 5],
) -> Dict[int, dict]:
    """
    Load tensor_dict from a pickle file and run the full multi-shot evaluation.
    Convenience wrapper for use in notebooks or quick one-liners.

    Example:
        results = eval_from_path("data/tensor_dict.pkl", build_config())
        results = eval_from_path("data/tensor_dict.pkl", build_config(), encoder=model.backbone)
    """
    with open(tensor_dict_path, "rb") as f:
        tensor_dict = pickle.load(f)
    return run_all_conditions(tensor_dict, config,
                              shot_conditions=shot_conditions, encoder=encoder)


# =============================================================================
# SECTION 10: CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subject-specific KNN + ProtoNet evaluation (raw, PCA, encoded)"
    )
    parser.add_argument("--tensor_dict",  type=str, required=True,
                        help="Path to maml_tensor_dict.pkl")
    parser.add_argument("--shots",        type=int, nargs="+", default=[1, 3, 5],
                        help="Shot conditions to evaluate (default: 1 3 5)")
    parser.add_argument("--n_way",        type=int, default=10,
                        help="Number of gesture classes (default: 10)")
    parser.add_argument("--metric",       type=str, default="euclidean",
                        choices=["euclidean", "cosine"],
                        help="Distance metric (default: euclidean)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--pca_dims",     type=int, default=16,
                        help="PCA output dimensionality (default: 16)")
    parser.add_argument("--aug",          action="store_true",
                        help="Enable support-set augmentation")
    parser.add_argument("--aug_copies",   type=int, default=4,
                        help="Augmented copies per support sample (default: 4)")
    parser.add_argument("--random_split", action="store_true",
                        help="Random-shuffle split instead of fixed-rep split")
    parser.add_argument("--eval_pids",    type=int, nargs="+", default=None,
                        help="Participant IDs to evaluate (default: 1-24)")
    args = parser.parse_args()

    config = build_config(
        eval_pids           = args.eval_pids,
        n_way               = args.n_way,
        metric              = args.metric,
        seed                = args.seed,
        pca_n_components    = args.pca_dims,
        aug_support         = args.aug,
        aug_n_copies        = args.aug_copies,
        use_fixed_rep_split = not args.random_split,
    )

    # No encoder -- raw + PCA baselines only.
    # To add a neural encoder:
    #   results = eval_from_path(..., encoder=model.backbone)
    results = eval_from_path(
        args.tensor_dict,
        config,
        encoder         = None,
        shot_conditions = args.shots,
    )
