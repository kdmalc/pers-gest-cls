"""
run_eval_knn_proto.py
=====================
Evaluates subject-specific KNN and Prototypical Networks under
1-shot, 3-shot, and 5-shot conditions.

Can be run as a script:
    python run_eval_knn_proto.py --tensor_dict path/to/tensor_dict.pkl

Or used cell-by-cell in a Jupyter notebook by importing:
    from run_eval_knn_proto import build_config, run_all_conditions

Evaluation modes per shot condition
-------------------------------------
  1. knn_raw        : KNN on flattened raw EMG features
  2. proto_raw      : ProtoNet on raw EMG features (per-class mean in input space)
  3. proto_encoded  : ProtoNet on encoder outputs  (only if encoder is provided)

Support / Query split
---------------------
  By default uses a FIXED split:
      support_reps : first k_shot reps (simulates real calibration deployment)
      query_reps   : all remaining reps
  You can override with explicit lists in config:
      config["support_reps"] = [1]          # for 1-shot
      config["query_reps"]   = [2,3,...,10]

  If support_reps/query_reps are not set, falls back to random shuffle split
  (matches MetaGestureDataset behavior, useful for apples-to-apples vs MAML).

Support Augmentation (optional)
--------------------------------
  config["aug_support"]    = True   # augment support samples
  config["aug_n_copies"]   = 4      # number of augmented copies per support sample
  config["aug_noise_std"]  = 0.05
  config["aug_max_shift"]  = 4
  config["aug_ch_drop"]    = 0.10
  Augmented copies are averaged INTO the prototype (not added as extra KNN neighbors
  for KNN, since that would artificially inflate k and break the k-shot framing).
  For KNN, augmented copies are added as extra support points.
"""

import os
import pickle
import random
import argparse
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Re-import core primitives from eval_knn_proto (or inline them here if needed)
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_euclidean_sq(a, b):
    a_sq = (a * a).sum(1, keepdim=True)
    b_sq = (b * b).sum(1, keepdim=True).t()
    return (a_sq + b_sq - 2 * (a @ b.t())).clamp(min=0.0)

def pairwise_cosine_dist(a, b):
    return 1.0 - (F.normalize(a, p=2, dim=1) @ F.normalize(b, p=2, dim=1).t())

def pairwise_dist(a, b, metric="euclidean"):
    return pairwise_cosine_dist(a, b) if metric == "cosine" else pairwise_euclidean_sq(a, b)

def knn_classify(sup_feats, sup_labels, qry_feats, k, metric="euclidean", n_classes=10):
    dist        = pairwise_dist(qry_feats, sup_feats, metric)           # (Q, S)
    k_eff       = min(k, sup_feats.size(0))
    _, nn_idx   = dist.topk(k_eff, dim=1, largest=False, sorted=True)  # (Q, k)
    nn_labels   = sup_labels[nn_idx]                                    # (Q, k)
    votes       = torch.zeros(qry_feats.size(0), n_classes,
                              device=qry_feats.device)
    votes.scatter_add_(1, nn_labels,
                       torch.ones_like(nn_labels, dtype=torch.float32))
    return votes.argmax(1)

def proto_classify(sup_feats, sup_labels, qry_feats, metric="euclidean", n_classes=10):
    D          = sup_feats.size(1)
    prototypes = torch.zeros(n_classes, D,
                             device=sup_feats.device, dtype=sup_feats.dtype)
    for c in range(n_classes):
        mask = sup_labels == c
        if mask.any():
            prototypes[c] = sup_feats[mask].mean(0)
    dist = pairwise_dist(qry_feats, prototypes, metric)                 # (Q, n_classes)
    return dist.argmin(1)


# ─────────────────────────────────────────────────────────────────────────────
# Data augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aug_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    return x + torch.randn_like(x) * (x.std() * std + 1e-8)

def _aug_shift(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(x, shifts=shift, dims=-1)

def _aug_ch_drop(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    mask = (torch.rand(x.size(0), 1, device=x.device) > drop_prob).float()
    return x * mask

def augment_sample(x: torch.Tensor, config: dict) -> torch.Tensor:
    """Apply all enabled augmentations to a single (C, T) EMG tensor."""
    x = _aug_noise(x, config.get("aug_noise_std", 0.05))
    x = _aug_shift(x, config.get("aug_max_shift", 4))
    x = _aug_ch_drop(x, config.get("aug_ch_drop", 0.10))
    return x

def expand_support_with_aug(
    sup_emg: torch.Tensor,     # (N_sup, C, T)
    sup_labels: torch.Tensor,  # (N_sup,)
    config: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates augmented copies of every support sample and appends them.
    Returns expanded (sup_emg, sup_labels) with shape
        ((N_sup * (1 + n_copies), C, T), (N_sup * (1 + n_copies),))
    
    Note: augmented copies have the SAME label as their source sample.
    The original samples are always kept at position 0..N_sup-1.
    """
    n_copies = int(config.get("aug_n_copies", 4))
    aug_emg_list    = [sup_emg]
    aug_labels_list = [sup_labels]

    for _ in range(n_copies):
        batch_aug = torch.stack([
            augment_sample(sup_emg[i], config) for i in range(sup_emg.size(0))
        ])
        aug_emg_list.append(batch_aug)
        aug_labels_list.append(sup_labels)

    return torch.cat(aug_emg_list, dim=0), torch.cat(aug_labels_list, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_batch(encoder, emg, imu, device):
    encoder.eval()
    out = encoder(emg.to(device), imu.to(device) if imu is not None else None)
    feats = out[0] if isinstance(out, (tuple, list)) else out
    return feats.float()


# ─────────────────────────────────────────────────────────────────────────────
# Episode builder  (supports fixed-rep and random-shuffle splits)
# ─────────────────────────────────────────────────────────────────────────────

def build_episode(
    user_data: dict,
    k_shot: int,
    n_way: int,
    rng: random.Random,
    support_reps: Optional[List[int]],   # 1-indexed; None → random shuffle
    query_reps:   Optional[List[int]],   # 1-indexed; None → random shuffle
    all_reps:     List[int],             # full rep pool (1-indexed)
) -> Tuple[dict, dict]:
    """
    Build one support/query episode for a single user.

    Fixed-rep mode  (support_reps / query_reps are lists):
        Support samples come from the specified repetition indices.
        Query   samples come from the specified repetition indices.
        k_shot is implicitly len(support_reps).

    Random-shuffle mode (support_reps / query_reps are None):
        All reps in all_reps are shuffled; first k_shot → support, rest → query.
        Matches MetaGestureDataset behavior exactly.

    Returns
    -------
    support, query : dicts with keys "emg" (N, C, T), "imu" (N,...)|None, "labels" (N,)
    """
    available = sorted([g for g in all_reps if g in user_data])
    if len(available) < n_way:
        raise ValueError(f"Only {len(available)} gestures available, need {n_way}")
    selected = sorted(rng.sample(available, n_way))
    label_map = {g: i for i, g in enumerate(selected)}

    sup_e, sup_i_list, sup_l = [], [], []
    qry_e, qry_i_list, qry_l = [], [], []

    for gest in selected:
        lbl      = label_map[gest]
        emg_data = user_data[gest]["emg"]   # (n_trials, ...)
        imu_data = user_data[gest].get("imu")
        n_trials = emg_data.shape[0]

        if support_reps is not None:
            # Fixed-rep split (1-indexed → 0-indexed)
            sup_idx = [r - 1 for r in support_reps if 0 <= r - 1 < n_trials]
            qry_idx = [r - 1 for r in query_reps   if 0 <= r - 1 < n_trials]
        else:
            # Random-shuffle split
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

    def _pack(e_list, i_list, l_list):
        emg = torch.stack(e_list).float()
        if emg.dim() == 3 and emg.shape[-1] in [16, 72]:
            emg = emg.permute(0, 2, 1).contiguous()
        imu = torch.stack(i_list).float() if i_list else None
        if imu is not None and imu.dim() == 3 and imu.shape[-1] in [16, 72]:
            imu = imu.permute(0, 2, 1).contiguous()
        return {"emg": emg, "imu": imu,
                "labels": torch.tensor(l_list, dtype=torch.long)}

    return _pack(sup_e, sup_i_list, sup_l), _pack(qry_e, qry_i_list, qry_l)


# ─────────────────────────────────────────────────────────────────────────────
# Single-user evaluation for ONE shot condition
# ─────────────────────────────────────────────────────────────────────────────

def eval_one_user_one_condition(
    user_data: dict,
    k_shot: int,
    n_way: int,
    support_reps: Optional[List[int]],
    query_reps:   Optional[List[int]],
    all_reps:     List[int],
    rng: random.Random,
    device: torch.device,
    metric: str,
    encoder: Optional[nn.Module],
    use_imu: bool,
    aug_support: bool,
    config: dict,
) -> dict:
    support, query = build_episode(
        user_data, k_shot, n_way, rng,
        support_reps, query_reps, all_reps,
    )

    sup_emg    = support["emg"].to(device)
    qry_emg    = query["emg"].to(device)
    sup_labels = support["labels"].to(device)
    qry_labels = query["labels"].to(device)
    sup_imu    = support["imu"].to(device) if (use_imu and support["imu"] is not None) else None
    qry_imu    = query["imu"].to(device)   if (use_imu and query["imu"] is not None)   else None

    # ── Optional support augmentation ───────────────────────────────────────
    aug_sup_emg, aug_sup_labels = sup_emg, sup_labels
    if aug_support:
        aug_sup_emg, aug_sup_labels = expand_support_with_aug(sup_emg, sup_labels, config)
        # Note: aug_sup_imu is not augmented here (extend if needed)

    # ── Raw features: flatten (C, T) → D ────────────────────────────────────
    sup_flat     = aug_sup_emg.reshape(aug_sup_emg.size(0), -1)
    qry_flat     = qry_emg.reshape(qry_emg.size(0), -1)
    sup_flat_raw = sup_emg.reshape(sup_emg.size(0), -1)   # un-augmented for KNN neighbor count

    # KNN — augmented support samples act as extra neighbors (increases effective k)
    knn_k    = aug_sup_emg.size(0) // n_way   # k = all support (aug or not), per-class
    knn_pred = knn_classify(sup_flat, aug_sup_labels, qry_flat,
                            k=knn_k * n_way, metric=metric, n_classes=n_way)
    knn_acc  = (knn_pred == qry_labels).float().mean().item()

    # ProtoNet raw — augmented copies fold into prototype mean automatically
    proto_raw_pred = proto_classify(sup_flat, aug_sup_labels, qry_flat,
                                    metric=metric, n_classes=n_way)
    proto_raw_acc  = (proto_raw_pred == qry_labels).float().mean().item()

    # ProtoNet encoded
    proto_enc_acc = None
    if encoder is not None:
        sup_enc = encode_batch(encoder, aug_sup_emg, sup_imu, device)
        qry_enc = encode_batch(encoder, qry_emg,    qry_imu, device)
        proto_enc_pred = proto_classify(sup_enc, aug_sup_labels, qry_enc,
                                        metric=metric, n_classes=n_way)
        proto_enc_acc  = (proto_enc_pred == qry_labels).float().mean().item()

    return {
        "knn_raw":       knn_acc,
        "proto_raw":     proto_raw_acc,
        "proto_encoded": proto_enc_acc,
        "n_support":     sup_emg.size(0),         # original (un-augmented) support size
        "n_support_aug": aug_sup_emg.size(0),      # after augmentation
        "n_query":       qry_emg.size(0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run one full shot condition across all users
# ─────────────────────────────────────────────────────────────────────────────

def run_one_shot_condition(
    tensor_dict: dict,
    config: dict,
    k_shot: int,
    encoder: Optional[nn.Module] = None,
    verbose: bool = True,
) -> dict:
    """
    Run all eval modes for a given k_shot across all eval_PIDs.

    Returns dict:
        {
          "knn_raw":       {"per_user": {pid: acc}, "mean_acc", "std_acc", "all_accs"},
          "proto_raw":     { ... },
          "proto_encoded": { ... },   # only if encoder provided
        }
    """
    device      = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    n_way       = int(config.get("n_way", 10))
    metric      = config.get("knn_metric", "euclidean")
    seed        = int(config.get("seed", 42))
    use_imu     = bool(config.get("use_imu", False))
    all_reps    = config["maml_reps"]
    aug_support = bool(config.get("aug_support", False))
    eval_pids   = config.get("eval_PIDs", config.get("val_PIDs", []))

    # ── Determine support / query rep split ─────────────────────────────────
    # Priority: explicit support_reps/query_reps > auto-fixed > random shuffle
    if "support_reps" in config and "query_reps" in config:
        # Caller provided explicit lists — use them directly regardless of k_shot
        support_reps = config["support_reps"]
        query_reps   = config["query_reps"]
        split_mode   = "fixed-explicit"
    elif config.get("use_fixed_rep_split", True):
        # Auto-fixed: first k_shot reps → support, rest → query
        # This is the deployment-realistic default
        sorted_reps  = sorted(all_reps)
        support_reps = sorted_reps[:k_shot]
        query_reps   = sorted_reps[k_shot:]
        split_mode   = f"fixed-auto (sup={support_reps}, qry={query_reps[:3]}...)"
    else:
        # Random shuffle — matches MetaGestureDataset / MAML pipeline
        support_reps = None
        query_reps   = None
        split_mode   = "random-shuffle"

    rng = random.Random(seed)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  {k_shot}-shot  {n_way}-way  |  metric={metric}  split={split_mode}")
        print(f"  aug_support={aug_support}" +
              (f"  n_copies={config.get('aug_n_copies',4)}" if aug_support else ""))
        print(f"  {len(eval_pids)} users  |  device={device}")
        print(f"{'='*65}")

    knn_accs, proto_raw_accs, proto_enc_accs = [], [], []
    per_user: Dict[Any, dict] = {}

    for pid in eval_pids:
        if pid not in tensor_dict:
            if verbose:
                print(f"  [WARN] PID {pid} not in tensor_dict — skipping.")
            continue

        user_data = tensor_dict[pid]
        available = [g for g in all_reps if g in user_data]
        if len(available) < n_way:
            if verbose:
                print(f"  [WARN] PID {pid}: {len(available)} gestures < {n_way} — skipping.")
            continue

        # Check enough query reps exist
        min_trials = min(user_data[g]["emg"].shape[0] for g in available[:n_way])
        needed = k_shot + 1  # at least 1 query sample
        if min_trials < needed:
            if verbose:
                print(f"  [WARN] PID {pid}: {min_trials} trials < {needed} needed — skipping.")
            continue

        try:
            res = eval_one_user_one_condition(
                user_data    = user_data,
                k_shot       = k_shot,
                n_way        = n_way,
                support_reps = support_reps,
                query_reps   = query_reps,
                all_reps     = all_reps,
                rng          = rng,
                device       = device,
                metric       = metric,
                encoder      = encoder,
                use_imu      = use_imu,
                aug_support  = aug_support,
                config       = config,
            )
        except Exception as e:
            if verbose:
                print(f"  [ERROR] PID {pid}: {e} — skipping.")
            continue

        knn_accs.append(res["knn_raw"])
        proto_raw_accs.append(res["proto_raw"])
        if encoder is not None and res["proto_encoded"] is not None:
            proto_enc_accs.append(res["proto_encoded"])

        per_user[pid] = res

        if verbose:
            enc_str = (f"  proto_enc={res['proto_encoded']*100:.1f}%"
                       if encoder else "")
            aug_str = (f"  (sup_aug={res['n_support_aug']})"
                       if aug_support else "")
            print(f"  PID {str(pid):>4} | "
                  f"knn={res['knn_raw']*100:.1f}%  "
                  f"proto_raw={res['proto_raw']*100:.1f}%"
                  f"{enc_str}  "
                  f"[sup={res['n_support']}, qry={res['n_query']}]"
                  f"{aug_str}")

    def _summ(accs, key):
        if not accs:
            return {}
        t = torch.tensor(accs)
        return {
            "per_user": {pid: per_user[pid][key] for pid in eval_pids if pid in per_user},
            "mean_acc": t.mean().item(),
            "std_acc":  t.std().item(),
            "all_accs": accs,
        }

    results = {
        "knn_raw":   _summ(knn_accs,       "knn_raw"),
        "proto_raw": _summ(proto_raw_accs,  "proto_raw"),
    }
    if encoder is not None:
        results["proto_encoded"] = _summ(proto_enc_accs, "proto_encoded")

    if verbose:
        chance = 100.0 / n_way
        n_users = len(knn_accs)
        print(f"\n  {'Method':<28}  {'Mean':>8}  {'Std':>7}  {'n_users':>8}")
        print(f"  {'─'*28}  {'─'*8}  {'─'*7}  {'─'*8}")
        for key, label in [("knn_raw",  "KNN (raw)"),
                            ("proto_raw","ProtoNet (raw)")]:
            r = results[key]
            if r:
                print(f"  {label:<28}  {r['mean_acc']*100:>7.2f}%  "
                      f"{r['std_acc']*100:>6.2f}%  {n_users:>8}")
        if encoder is not None and results.get("proto_encoded"):
            r = results["proto_encoded"]
            label = f"ProtoNet ({type(encoder).__name__})"
            print(f"  {label:<28}  {r['mean_acc']*100:>7.2f}%  "
                  f"{r['std_acc']*100:>6.2f}%  {n_users:>8}")
        print(f"  Chance = {chance:.1f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Run ALL shot conditions (1, 3, 5) and print comparison table
# ─────────────────────────────────────────────────────────────────────────────

def run_all_conditions(
    tensor_dict: dict,
    config: dict,
    shot_conditions: List[int] = [1, 3, 5],
    encoder: Optional[nn.Module] = None,
    verbose: bool = True,
) -> Dict[int, dict]:
    """
    Runs run_one_shot_condition for each k in shot_conditions.

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
            encoder=encoder, verbose=verbose,
        )

    # ── Final comparison table ───────────────────────────────────────────────
    methods = ["knn_raw", "proto_raw"]
    if encoder is not None:
        methods.append("proto_encoded")

    method_labels = {
        "knn_raw":       "KNN (raw)",
        "proto_raw":     "ProtoNet (raw)",
        "proto_encoded": f"ProtoNet (encoded)",
    }

    n_way  = config.get("n_way", 10)
    chance = 100.0 / n_way

    print(f"\n\n{'='*75}")
    print(f"  FINAL COMPARISON TABLE  ({n_way}-way,  chance={chance:.1f}%)")
    print(f"{'='*75}")

    # Header
    header = f"  {'Method':<28}"
    for k in shot_conditions:
        header += f"  {k}-shot (mean±std)"
    print(header)
    print(f"  {'─'*28}" + "  " + "  ".join(["─"*18] * len(shot_conditions)))

    for method in methods:
        row = f"  {method_labels[method]:<28}"
        for k in shot_conditions:
            r = all_results[k].get(method, {})
            if r:
                row += f"  {r['mean_acc']*100:>6.2f}±{r['std_acc']*100:>5.2f}%   "
            else:
                row += f"  {'N/A':>18}   "
        print(row)

    print(f"{'='*75}\n")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Config builder  (edit this to match your dataset)
# ─────────────────────────────────────────────────────────────────────────────

def build_config(
    eval_pids: List = None,
    maml_reps: List = None,
    n_way: int = 10,
    metric: str = "euclidean",
    seed: int = 42,
    use_imu: bool = False,
    aug_support: bool = False,
    aug_n_copies: int = 4,
    use_fixed_rep_split: bool = True,
    device: str = None,
) -> dict:
    """
    Build a standard config dict.
    Defaults match the typical setup: users 1-24, reps 1-10, 10-way.
    """
    return {
        "eval_PIDs":          eval_pids  if eval_pids  is not None else list(range(1, 25)),
        "maml_reps":          maml_reps  if maml_reps  is not None else list(range(1, 11)),
        "n_way":              n_way,
        "knn_metric":         metric,
        "seed":               seed,
        "use_imu":            use_imu,
        "aug_support":        aug_support,
        "aug_n_copies":       aug_n_copies,
        "aug_noise_std":      0.05,
        "aug_max_shift":      4,
        "aug_ch_drop":        0.10,
        "use_fixed_rep_split": use_fixed_rep_split,
        "device":             device or ("cuda" if torch.cuda.is_available() else "cpu"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_dict", type=str, required=True,
                        help="Path to maml_tensor_dict.pkl")
    parser.add_argument("--shots",   type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--n_way",   type=int, default=10)
    parser.add_argument("--metric",  type=str, default="euclidean",
                        choices=["euclidean", "cosine"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--aug",     action="store_true",
                        help="Enable support augmentation")
    parser.add_argument("--random_split", action="store_true",
                        help="Use random shuffle split instead of fixed-rep split")
    parser.add_argument("--eval_pids", type=int, nargs="+", default=None,
                        help="Override eval PIDs (default: 1-24)")
    args = parser.parse_args()

    with open(args.tensor_dict, "rb") as f:
        tensor_dict = pickle.load(f)

    config = build_config(
        eval_pids           = args.eval_pids,
        n_way               = args.n_way,
        metric              = args.metric,
        seed                = args.seed,
        aug_support         = args.aug,
        use_fixed_rep_split = not args.random_split,
    )

    # No encoder — raw baselines only
    # To add an encoder:  run_all_conditions(..., encoder=model.backbone)
    all_results = run_all_conditions(
        tensor_dict,
        config,
        shot_conditions = args.shots,
        encoder         = None,
    )
