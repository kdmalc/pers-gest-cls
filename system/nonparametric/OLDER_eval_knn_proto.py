"""
eval_knn_proto.py
=================
Subject-specific few-shot evaluation: KNN and Prototypical Networks.
Everything runs in pure PyTorch — no sklearn dependency.

Three evaluation modes
----------------------
1. knn_raw        : KNN on raw EMG features (flattened input space)
2. proto_raw      : ProtoNet on raw EMG features (mean of support samples per class)
3. proto_encoded  : ProtoNet on encoder outputs (plug in your pretrained backbone)

Setup: 1-shot, 10-way per user (configurable via config dict).
  - Support set : k_shot samples per class  (e.g. 1)
  - Query  set  : all remaining samples per class
  - One "episode" per user — no episodic training, pure evaluation.

Usage
-----
    from eval_knn_proto import run_knn_proto_eval

    # --- Raw feature baselines (no model needed) ---
    results = run_knn_proto_eval(tensor_dict, config)

    # --- Encoded ProtoNet (plug in pretrained backbone) ---
    results = run_knn_proto_eval(tensor_dict, config, encoder=model.backbone)

    # Results dict:
    #   results["knn_raw"]       -> {"per_user": {pid: acc}, "mean_acc": float, "std_acc": float}
    #   results["proto_raw"]     -> same structure
    #   results["proto_encoded"] -> same structure (only if encoder is provided)

Config keys used
----------------
    eval_PIDs      : list of participant IDs to evaluate
    maml_reps      : list of 1-indexed repetition indices available per gesture
    k_shot         : int, support samples per class (default 1)
    n_way          : int, number of classes (default 10)
    knn_k          : int, number of neighbors for KNN vote (default k_shot * n_way, i.e. all support)
    knn_metric     : "euclidean" | "cosine" (default "euclidean")
    seed           : int for reproducible support/query splits (default 42)
    device         : "cuda" | "cpu"
    use_imu        : bool (default False)
"""

import random
import pickle
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Distance helpers (pure PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_euclidean_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Squared Euclidean distance between every row of a and every row of b.
    a : (N, D)  b : (M, D)  →  (N, M)

    Uses the identity  ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b^T
    which is ~10x faster than looping and numerically equivalent.
    """
    a_sq = (a * a).sum(dim=1, keepdim=True)      # (N, 1)
    b_sq = (b * b).sum(dim=1, keepdim=True).t()  # (1, M)
    ab   = a @ b.t()                              # (N, M)
    dist_sq = a_sq + b_sq - 2 * ab
    # Clamp to avoid tiny negative values from floating-point noise
    return dist_sq.clamp(min=0.0)


def pairwise_cosine_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Cosine *distance* (1 - cosine_similarity) between every row of a and b.
    a : (N, D)  b : (M, D)  →  (N, M)
    """
    a_n = F.normalize(a, p=2, dim=1)
    b_n = F.normalize(b, p=2, dim=1)
    return 1.0 - (a_n @ b_n.t())


def pairwise_dist(a: torch.Tensor, b: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """Dispatcher. Returns (N, M) distance matrix."""
    if metric == "cosine":
        return pairwise_cosine_dist(a, b)
    else:  # euclidean (default)
        return pairwise_euclidean_sq(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# KNN classifier (pure PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def knn_classify(
    support_feats: torch.Tensor,   # (N_support, D)
    support_labels: torch.Tensor,  # (N_support,)   int
    query_feats: torch.Tensor,     # (N_query, D)
    k: int,
    metric: str = "euclidean",
    n_classes: int = 10,
) -> torch.Tensor:
    """
    Pure-PyTorch KNN.
    
    Returns predicted labels for each query sample: (N_query,) int tensor.

    Implementation note: for each query we find its k nearest support neighbors,
    then do a soft-count vote (sum indicator per class) and take the argmax.
    This is identical to standard KNN majority vote and is fully differentiable
    w.r.t. the feature vectors if you ever want to fine-tune through it.
    """
    dist = pairwise_dist(query_feats, support_feats, metric)  # (N_query, N_support)

    # kNN: take the k closest support indices per query
    # torch.topk with largest=False gives us the k smallest distances
    k_eff = min(k, support_feats.size(0))
    _, nn_indices = dist.topk(k_eff, dim=1, largest=False, sorted=True)  # (N_query, k)

    # Gather labels of neighbors
    nn_labels = support_labels[nn_indices]  # (N_query, k)

    # Majority vote via one-hot accumulation
    # Shape: (N_query, n_classes)
    votes = torch.zeros(query_feats.size(0), n_classes,
                        device=query_feats.device, dtype=torch.float32)
    votes.scatter_add_(
        dim=1,
        index=nn_labels,
        src=torch.ones_like(nn_labels, dtype=torch.float32),
    )

    return votes.argmax(dim=1)  # (N_query,)


# ─────────────────────────────────────────────────────────────────────────────
# ProtoNet classifier (pure PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def proto_classify(
    support_feats: torch.Tensor,   # (N_support, D)
    support_labels: torch.Tensor,  # (N_support,)   int
    query_feats: torch.Tensor,     # (N_query, D)
    metric: str = "euclidean",
    n_classes: int = 10,
) -> torch.Tensor:
    """
    Prototypical Network classifier.

    1. Compute per-class prototype = mean of support embeddings for that class.
    2. Classify each query by nearest prototype.

    Returns predicted labels: (N_query,) int tensor.

    Note on k=1 equivalence:
        With k_shot=1, each class has exactly one support sample, so the
        prototype IS that sample. ProtoNet and KNN give identical predictions
        in this case (assuming same distance metric). With k>1, ProtoNet
        pools within-class info before comparing, which is its key advantage.
    """
    # Build prototype matrix: (n_classes, D)
    D = support_feats.size(1)
    prototypes = torch.zeros(n_classes, D,
                             device=support_feats.device, dtype=support_feats.dtype)
    counts = torch.zeros(n_classes, device=support_feats.device, dtype=torch.float32)

    for c in range(n_classes):
        mask = (support_labels == c)
        if mask.any():
            prototypes[c] = support_feats[mask].mean(dim=0)
            counts[c] = mask.sum().float()

    # Distance from each query to each prototype: (N_query, n_classes)
    dist = pairwise_dist(query_feats, prototypes, metric)

    return dist.argmin(dim=1)  # (N_query,) — nearest prototype wins


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def flatten_emg(emg: torch.Tensor) -> torch.Tensor:
    """
    Flatten a single EMG trial to a 1-D feature vector.
    Accepts either (C, T) or (T, C) — we just flatten everything.
    """
    return emg.reshape(-1)


@torch.no_grad()
def encode_batch(
    encoder: nn.Module,
    emg_batch: torch.Tensor,   # (N, C, T)
    imu_batch: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Run encoder.forward(emg, imu) and return features.
    
    Handles two common backbone calling conventions:
      (a) backbone returns (features, aux)  → we take features
      (b) backbone returns features directly
    
    Returned tensor: (N, D) float32.
    """
    encoder.eval()
    emg_batch = emg_batch.to(device)
    if imu_batch is not None:
        imu_batch = imu_batch.to(device)

    out = encoder(emg_batch, imu_batch)

    # Unwrap tuple output (backbone typically returns (feat, something))
    if isinstance(out, (tuple, list)):
        feats = out[0]
    else:
        feats = out

    return feats.float()


# ─────────────────────────────────────────────────────────────────────────────
# Per-user episode builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_episode(
    user_data: Dict[int, Dict],
    target_reps: list,
    k_shot: int,
    n_way: int,
    rng: random.Random,
) -> Tuple[Dict, Dict]:
    """
    For a single user, build one support/query episode.

    Returns
    -------
    support : {"emg": (N_sup, C, T), "imu": (N_sup, ...) | None, "labels": (N_sup,)}
    query   : {"emg": (N_qry, C, T), "imu": (N_qry, ...) | None, "labels": (N_qry,)}

    Label mapping is deterministic: sorted gesture IDs → 0..n_way-1.
    The caller must ensure the user has at least n_way gestures with enough reps.
    """
    available_gestures = [g for g in target_reps if g in user_data]
    # Randomly sample n_way classes for this episode (matches MetaGestureDataset behavior)
    selected_classes = rng.sample(available_gestures, min(n_way, len(available_gestures)))
    # Sort for deterministic label assignment within an episode
    selected_classes = sorted(selected_classes)
    label_map = {g: i for i, g in enumerate(selected_classes)}

    sup_emg_list, sup_imu_list, sup_labels = [], [], []
    qry_emg_list, qry_imu_list, qry_labels = [], [], []

    for gest in selected_classes:
        local_label = label_map[gest]
        emg_data = user_data[gest]["emg"]  # (n_trials, C, T) or (n_trials, T, C)
        imu_data = user_data[gest].get("imu")

        n_trials = emg_data.shape[0]
        indices  = list(range(n_trials))
        rng.shuffle(indices)

        sup_idx = indices[:k_shot]
        qry_idx = indices[k_shot:]  # All remaining → query

        for i in sup_idx:
            sup_emg_list.append(emg_data[i])
            if imu_data is not None:
                sup_imu_list.append(imu_data[i])
            sup_labels.append(local_label)

        for i in qry_idx:
            qry_emg_list.append(emg_data[i])
            if imu_data is not None:
                qry_imu_list.append(imu_data[i])
            qry_labels.append(local_label)

    def _pack(emg_list, imu_list, labels):
        # Stack and ensure (N, C, T) layout
        emg = torch.stack(emg_list).float()
        if emg.dim() == 3 and emg.shape[-1] in [16, 72]:
            emg = emg.permute(0, 2, 1).contiguous()  # (N,T,C) → (N,C,T)
        imu = None
        if imu_list:
            imu = torch.stack(imu_list).float()
            if imu.dim() == 3 and imu.shape[-1] in [16, 72]:
                imu = imu.permute(0, 2, 1).contiguous()
        return {"emg": emg, "imu": imu, "labels": torch.tensor(labels, dtype=torch.long)}

    support = _pack(sup_emg_list, sup_imu_list, sup_labels)
    query   = _pack(qry_emg_list, qry_imu_list, qry_labels)

    return support, query, label_map


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def _eval_one_user(
    user_data: Dict,
    target_reps: list,
    k_shot: int,
    n_way: int,
    knn_k: int,
    metric: str,
    rng: random.Random,
    device: torch.device,
    encoder: Optional[nn.Module],
    use_imu: bool,
) -> Dict[str, float]:
    """
    Run all three evaluation modes for a single user.
    Returns dict: {"knn_raw": acc, "proto_raw": acc, "proto_encoded": acc | None}
    """
    support, query, _ = build_user_episode(
        user_data, target_reps, k_shot, n_way, rng
    )

    sup_emg    = support["emg"].to(device)   # (N_sup, C, T)
    qry_emg    = query["emg"].to(device)     # (N_qry, C, T)
    sup_labels = support["labels"].to(device)
    qry_labels = query["labels"].to(device)
    n_classes  = n_way  # label space is always 0..n_way-1

    sup_imu = support["imu"].to(device) if (use_imu and support["imu"] is not None) else None
    qry_imu = query["imu"].to(device)   if (use_imu and query["imu"] is not None)   else None

    # ── 1. Raw features: flatten (C,T) → D ──────────────────────────────────
    sup_flat = sup_emg.reshape(sup_emg.size(0), -1)  # (N_sup, C*T)
    qry_flat = qry_emg.reshape(qry_emg.size(0), -1)  # (N_qry, C*T)

    # KNN on raw features
    knn_preds = knn_classify(sup_flat, sup_labels, qry_flat,
                             k=knn_k, metric=metric, n_classes=n_classes)
    knn_acc = (knn_preds == qry_labels).float().mean().item()

    # ProtoNet on raw features
    proto_raw_preds = proto_classify(sup_flat, sup_labels, qry_flat,
                                     metric=metric, n_classes=n_classes)
    proto_raw_acc = (proto_raw_preds == qry_labels).float().mean().item()

    # ── 2. Encoded features (optional) ──────────────────────────────────────
    proto_enc_acc = None
    if encoder is not None:
        sup_enc = encode_batch(encoder, sup_emg, sup_imu, device)    # (N_sup, D)
        qry_enc = encode_batch(encoder, qry_emg, qry_imu, device)    # (N_qry, D)
        proto_enc_preds = proto_classify(sup_enc, sup_labels, qry_enc,
                                         metric=metric, n_classes=n_classes)
        proto_enc_acc = (proto_enc_preds == qry_labels).float().mean().item()

    return {
        "knn_raw":        knn_acc,
        "proto_raw":      proto_raw_acc,
        "proto_encoded":  proto_enc_acc,
        "n_support":      sup_emg.size(0),
        "n_query":        qry_emg.size(0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_knn_proto_eval(
    tensor_dict: Dict,
    config: Dict[str, Any],
    encoder: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Run subject-specific KNN + ProtoNet evaluation over all eval_PIDs.

    Parameters
    ----------
    tensor_dict : dict  — same format used by MetaGestureDataset
    config      : dict  — see module docstring for keys
    encoder     : nn.Module | None
        Optional pretrained backbone. Must accept (emg, imu) and return
        either a tensor (N, D) or tuple (tensor, ...).
        If None, proto_encoded results are omitted.

    Returns
    -------
    results : dict with keys "knn_raw", "proto_raw", "proto_encoded" (if encoder given).
        Each value is:
            {
                "per_user"  : {pid: acc_float},
                "mean_acc"  : float,
                "std_acc"   : float,
                "all_accs"  : [float, ...]   # same order as eval_PIDs
            }
    """
    device    = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    k_shot    = int(config.get("k_shot",  1))
    n_way     = int(config.get("n_way",   10))
    knn_k     = int(config.get("knn_k",   k_shot))   # default: use all support samples
    metric    = config.get("knn_metric",  "euclidean")
    seed      = int(config.get("seed",    42))
    use_imu   = bool(config.get("use_imu", False))
    target_reps = config["maml_reps"]
    eval_pids = config.get("eval_PIDs", config.get("val_PIDs", []))

    if not eval_pids:
        raise ValueError("config must contain 'eval_PIDs' (or 'val_PIDs') — got empty list.")

    rng = random.Random(seed)  # Isolated, reproducible RNG — won't affect global random state

    print(f"\n{'='*65}")
    print(f"  Subject-Specific Few-Shot Evaluation")
    print(f"  {k_shot}-shot  {n_way}-way  |  kNN k={knn_k}  metric={metric}")
    print(f"  {len(eval_pids)} users  |  device={device}")
    if encoder is not None:
        print(f"  Encoder: {type(encoder).__name__}")
    print(f"{'='*65}")

    # Accumulate per-user results
    knn_raw_accs, proto_raw_accs, proto_enc_accs = [], [], []
    user_results: Dict[str, Dict] = {}

    for pid in eval_pids:
        if pid not in tensor_dict:
            print(f"  [WARN] PID {pid} not in tensor_dict — skipping.")
            continue

        user_data = tensor_dict[pid]
        available = [g for g in target_reps if g in user_data]

        # Gracefully skip users with insufficient classes or trials
        if len(available) < n_way:
            print(f"  [WARN] PID {pid}: only {len(available)} gestures available "
                  f"(need {n_way}) — skipping.")
            continue

        min_trials = min(user_data[g]["emg"].shape[0] for g in available[:n_way])
        if min_trials <= k_shot:
            print(f"  [WARN] PID {pid}: only {min_trials} trial(s) per gesture "
                  f"(need >{k_shot}) — skipping.")
            continue

        res = _eval_one_user(
            user_data   = user_data,
            target_reps = target_reps,
            k_shot      = k_shot,
            n_way       = n_way,
            knn_k       = knn_k,
            metric      = metric,
            rng         = rng,
            device      = device,
            encoder     = encoder,
            use_imu     = use_imu,
        )

        knn_raw_accs.append(res["knn_raw"])
        proto_raw_accs.append(res["proto_raw"])
        if encoder is not None:
            proto_enc_accs.append(res["proto_encoded"])

        user_results[pid] = res

        enc_str = f"  proto_enc={res['proto_encoded']*100:.1f}%" if encoder else ""
        print(f"  PID {pid:>4} | "
              f"knn_raw={res['knn_raw']*100:.1f}%  "
              f"proto_raw={res['proto_raw']*100:.1f}%"
              f"{enc_str}  "
              f"(sup={res['n_support']}, qry={res['n_query']})")

    def _summarize(accs, label):
        if not accs:
            return {}
        t = torch.tensor(accs)
        return {
            "per_user":  {pid: user_results[pid][label]
                          for pid in eval_pids if pid in user_results},
            "mean_acc":  t.mean().item(),
            "std_acc":   t.std().item(),
            "all_accs":  accs,
        }

    results = {
        "knn_raw":   _summarize(knn_raw_accs,   "knn_raw"),
        "proto_raw": _summarize(proto_raw_accs,  "proto_raw"),
    }
    if encoder is not None:
        results["proto_encoded"] = _summarize(proto_enc_accs, "proto_encoded")

    # ── Pretty summary table ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  SUMMARY  ({k_shot}-shot {n_way}-way,  n={len(knn_raw_accs)} users)")
    print(f"{'─'*65}")
    print(f"  {'Method':<25}  {'Mean Acc':>10}  {'Std':>8}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*8}")

    for key, label in [("knn_raw",  "KNN (raw)"),
                        ("proto_raw","ProtoNet (raw)")]:
        r = results[key]
        print(f"  {label:<25}  {r['mean_acc']*100:>9.2f}%  {r['std_acc']*100:>7.2f}%")

    if encoder is not None:
        r = results["proto_encoded"]
        label = f"ProtoNet ({type(encoder).__name__})"
        print(f"  {label:<25}  {r['mean_acc']*100:>9.2f}%  {r['std_acc']*100:>7.2f}%")

    print(f"{'─'*65}")
    print(f"  Chance level: {100.0/n_way:.1f}%")
    print(f"{'='*65}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load tensor_dict and run
# ─────────────────────────────────────────────────────────────────────────────

def eval_from_path(
    tensor_dict_path: str,
    config: Dict[str, Any],
    encoder: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Load tensor_dict from pickle and run evaluation. Convenience wrapper."""
    with open(tensor_dict_path, "rb") as f:
        tensor_dict = pickle.load(f)
    return run_knn_proto_eval(tensor_dict, config, encoder=encoder)


# ─────────────────────────────────────────────────────────────────────────────
# __main__ smoke test / standalone usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Subject-specific KNN + ProtoNet eval")
    parser.add_argument("--tensor_dict", type=str, required=True)
    parser.add_argument("--k_shot",   type=int,   default=1)
    parser.add_argument("--n_way",    type=int,   default=10)
    parser.add_argument("--metric",   type=str,   default="euclidean",
                        choices=["euclidean", "cosine"])
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    # Minimal config — adjust PIDs and reps to match your dataset
    _config = {
        "eval_PIDs":   list(range(1, 25)),      # Users 1-24 (pretrain users)
        "maml_reps":   list(range(1, 11)),      # Repetitions 1-10 (1-indexed)
        "k_shot":      args.k_shot,
        "n_way":       args.n_way,
        "knn_metric":  args.metric,
        "seed":        args.seed,
        "use_imu":     False,
        "device":      "cuda" if torch.cuda.is_available() else "cpu",
    }

    results = eval_from_path(args.tensor_dict, _config, encoder=None)
