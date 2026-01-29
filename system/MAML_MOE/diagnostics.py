import torch, numpy as np
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

def inspect_dataloader_batch(
    dataloader,
    *,
    name: str = "dataloader",
    expect_dict: bool = True,
    show_first_element: bool = True,
    truncate_first_repr: int = 200,
    pid_key: str = "PIDs",
    label_key: str = "label",
):
    """
    Grab one batch from a DataLoader and run sanity checks.

    Adds:
      - PID checks: NaN detection and unique PID listing.
      - Label diagnostics: shape checks, range, uniqueness, and warnings if labels
        appear to be one-hot or logits.
    """
    try:
        iterator = iter(dataloader)
    except TypeError as e:
        raise TypeError(f"❌ '{name}' is not iterable like a DataLoader: {e}")

    try:
        batch = next(iterator)
    except StopIteration:
        raise RuntimeError(f"❌ {name} is empty. No batches available.")

    # Normalize structure
    if expect_dict:
        assert isinstance(batch, dict), (
            f"❌ Expected batch to be a dict, got {type(batch)} instead."
        )
    else:
        if isinstance(batch, (list, tuple)):
            batch = {f"field_{i}": v for i, v in enumerate(batch)}
        elif not isinstance(batch, dict):
            raise TypeError(
                f"❌ Batch is {type(batch)}; set expect_dict=True or yield dict/tuple/list."
            )

    assert len(batch) > 0, "❌ Batch dictionary is empty."

    import math
    try:
        import torch
    except Exception:
        torch = None
    try:
        import numpy as np
    except Exception:
        np = None

    # Infer batch size
    batch_size = None
    for key, value in batch.items():
        if hasattr(value, "shape") and getattr(value, "shape", None) is not None:
            shp = value.shape
            if isinstance(shp, (tuple, list)) and len(shp) > 0:
                batch_size = shp[0]
                break
        try:
            ln = len(value)
            if ln is not None and ln > 0:
                batch_size = ln
                break
        except Exception:
            pass
    assert batch_size is not None, "❌ Could not infer batch size from any batch value."
    print(f"✅ [{name}] Batch size: {batch_size}")
    print(f"🔎 Keys: {list(batch.keys())}")

    # Inspect each key/value briefly
    for key, value in batch.items():
        vtype = type(value).__name__
        line = f"• Key: '{key}'  |  Type: {vtype}"

        shape_info = None
        dtype_info = None
        device_info = None

        if torch is not None and isinstance(value, torch.Tensor):
            shape_info = tuple(value.shape)
            dtype_info = str(value.dtype)
            device_info = str(value.device)
        elif np is not None and isinstance(value, np.ndarray):
            shape_info = value.shape
            dtype_info = str(value.dtype)

        if shape_info is not None:
            line += f"  |  shape: {shape_info}"
        if dtype_info is not None:
            line += f"  |  dtype: {dtype_info}"
        if device_info is not None:
            line += f"  |  device: {device_info}"
        if shape_info is None:
            try:
                line += f"  |  len: {len(value)}"
            except Exception:
                pass
        print(line)

        if show_first_element:
            try:
                first_element = value[0]
                s = repr(first_element)
                if truncate_first_repr and len(s) > truncate_first_repr:
                    s = s[:truncate_first_repr] + " ... [truncated]"
                print(f"    ↳ first element: {s}")
            except Exception as e:
                raise TypeError(
                    f"❌ Value for key '{key}' is not indexable or failed to access first element: {e}"
                )

    # -----------------------
    # PID checks (if present)
    # -----------------------
    if pid_key in batch:
        pids = batch[pid_key]
        print(f"\n🧩 PID checks on '{pid_key}':")
        # Convert to torch tensor for robust ops when possible
        pid_tensor = None
        if torch is not None and isinstance(pids, torch.Tensor):
            pid_tensor = pids
        elif np is not None and isinstance(pids, np.ndarray):
            pid_tensor = torch.from_numpy(pids) if torch is not None else None
        elif isinstance(pids, (list, tuple)):
            if torch is not None:
                try:
                    pid_tensor = torch.tensor(pids)
                except Exception:
                    pid_tensor = None

        # NaN check
        has_nan = False
        if pid_tensor is not None and torch.is_floating_point(pid_tensor):
            has_nan = torch.isnan(pid_tensor).any().item()
        elif pid_tensor is not None and pid_tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            # integers can't be NaN; treat as no-NaN
            has_nan = False
        else:
            # Fallback generic check
            try:
                if np is not None:
                    arr = np.array(pids)
                    has_nan = np.isnan(arr).any()
                else:
                    has_nan = any(isinstance(x, float) and math.isnan(x) for x in pids) if isinstance(pids, (list, tuple)) else False
            except Exception:
                has_nan = False

        if has_nan:
            raise ValueError(f"❌ Found NaN(s) in '{pid_key}' for this batch.")

        # Unique listing
        if pid_tensor is not None:
            unique_pids = torch.unique(pid_tensor).tolist()
        else:
            # Fallback for non-tensor
            try:
                unique_pids = sorted(set(pids))
            except TypeError:
                unique_pids = list(set(pids))
        print(f"✅ No NaNs in '{pid_key}'.")
        print(f"🔢 Unique PIDs in batch ({len(unique_pids)}): {unique_pids}")

    # -------------------------
    # Label diagnostics (if any)
    # -------------------------
    if label_key in batch:
        y = batch[label_key]
        print(f"\n🏷️ Label diagnostics on '{label_key}':")
        # Normalize to torch tensor for checks when possible
        y_tensor = None
        if torch is not None and isinstance(y, torch.Tensor):
            y_tensor = y
        elif np is not None and isinstance(y, np.ndarray):
            y_tensor = torch.from_numpy(y) if torch is not None else None

        # Shape-based heuristics
        if y_tensor is not None and y_tensor.ndim == 1:
            # Expected for integer class indices per example
            print(f"• Shape looks like class indices: {tuple(y_tensor.shape)} (should be (batch_size,))")
            # Basic stats
            try:
                y_min = int(y_tensor.min().item())
                y_max = int(y_tensor.max().item())
                num_unique = int(torch.unique(y_tensor).numel())
                print(f"• Value range: [{y_min}, {y_max}]  |  unique labels in batch: {num_unique}")
            except Exception:
                pass

        elif y_tensor is not None and y_tensor.ndim == 2:
            bs, d = y_tensor.shape
            print(f"• 2D labels detected: shape={bs, d}")
            # Heuristics:
            # - If d > 1 and values are all in {0,1} and rows sum to 1 -> one-hot labels
            # - If d > 1 and values are real (not just 0/1) -> logits/probabilities
            try:
                is_binary = ((y_tensor == 0) | (y_tensor == 1)).all().item()
                row_sums = y_tensor.sum(dim=1)
                looks_one_hot = is_binary and torch.all((row_sums == 1)).item()
            except Exception:
                is_binary = False
                looks_one_hot = False

            if looks_one_hot:
                print("⚠️ Detected one-hot encoded labels. If you're using CrossEntropyLoss, provide class indices (shape (B,)) not one-hot.")
            else:
                print("⚠️ Labels appear to be vectors per sample (possibly logits/probabilities).")
                print("   • For CrossEntropyLoss, expected shape is (B, C) **logits** and **integer** targets of shape (B,).")
                print("   • If this is unintended, check your collate function or dataset to ensure you’re not returning logits as 'label'.")

        else:
            # Non-tensor or other shapes
            try:
                ln = len(y)
                print(f"• Non-tensor labels with len={ln}. Verify they are class indices.")
            except Exception:
                print("• Label type/shape unclear—verify your dataset emits class indices or logits as intended.")

    print("\n✅ Batch inspection complete.")
    return batch


##############################################################################
#A) Confusion matrix + per-class precision/recall/F1 (per novel user)
## Use this to compare MoE-PEFT vs Proto vs Ridge on the same user. If the same classes dominate the errors across methods, you’ve likely hit class overlap (hard Bayes errors). If MoE-PEFT confuses classes that Proto handles, your routing/adaptation is the culprit.

@torch.no_grad()
def logits_from_model(model, loader, device, user_embed_override=None):
    model.eval()
    all_logits, all_y = [], []
    for b in loader:
        x = (b["x"] if isinstance(b, dict) else b[0]).to(device)
        y = (b["y"] if isinstance(b, dict) else b[1]).cpu().numpy()
        u_override = None
        if user_embed_override is not None:
            u_override = user_embed_override
            if u_override.ndim == 2 and u_override.shape[0] == 1:
                u_override = u_override.expand(x.size(0), -1)
        logits, _ = model(x, user_ids=None, user_embed_override=u_override)
        all_logits.append(logits.cpu().numpy()); all_y.append(y)
    return np.concatenate(all_logits,0), np.concatenate(all_y,0)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.show()

# Example usage inside your report_user function:
from sklearn.metrics import confusion_matrix, classification_report

def report_user(model, query_loader, device, y_names=None, user_embed=None):
    logits, y_true = logits_from_model(model, query_loader, device, user_embed_override=user_embed)
    y_pred = logits.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(logits.shape[1]))
    
    # Plot figure instead of text
    plot_confusion_matrix(cm, y_names if y_names else range(logits.shape[1]))
    
    # Print classification report
    print(classification_report(y_true, y_pred, target_names=y_names))


##############################################################################

# C) Adaptive (support-set) normalization check

@torch.no_grad()
def support_stats(loader):
    xs = []
    for b in loader:
        x = (b["x"] if isinstance(b, dict) else b[0]).float()
        xs.append(x)
    X = torch.cat(xs,0)  # (Ns, 16, 5)
    mu = X.mean(dim=(0,2), keepdim=True)   # (1,16,1)
    sd = X.std(dim=(0,2), keepdim=True).clamp_min(1e-6)
    return mu, sd

def apply_adaptive_norm(x, mu, sd):  # x: (B,16,5)
    return (x - mu.to(x.device)) / sd.to(x.device)

##############################################################################

# A) Return top-k from your model forward

@torch.no_grad()
def route_topk(model, x, user_ids=None, user_embed_override=None, top_k=2):
    """
    Returns:
      top_idx: (B, k) expert indices
      top_w:   (B, k) corresponding weights after softmax+mask
    """
    h = model.backbone(x)  # (B, emb)
    # Build user embedding the same way as in your forward
    if user_embed_override is not None:
        u = user_embed_override
        if u.size(0) == 1: u = u.expand(h.size(0), -1)
    elif (getattr(model, "user_table", None) is not None) and (user_ids is not None):
        u = model.user_table(user_ids)
    else:
        u = torch.zeros(h.size(0), model.gate.lin.in_features - h.size(1), device=h.device)  # USER_DIM

    g = model.gate.lin(torch.cat([h, u], dim=-1))      # (B,E)
    w = F.softmax(g, dim=-1)
    if top_k < w.size(-1):
        top = torch.topk(w, top_k, dim=-1)
        idx = top.indices
        val = top.values
        # renormalize within top-k
        val = val / (val.sum(dim=-1, keepdim=True).clamp_min(1e-9))
        return idx, val
    else:
        # dense: return all experts
        B, E = w.shape
        idx = torch.arange(E, device=w.device).expand(B, E)
        return idx, w

##############################################################################

# B) Log routing stats over a loader

@torch.no_grad()
def collect_routing_stats(model, loader, device, top_k=2, num_classes=10):
    """
    Returns:
      expert_total[e]
      expert_by_class[e][c]
      expert_by_user[e][uid]
    """
    model.eval()
    expert_total = defaultdict(int)
    expert_by_class = defaultdict(lambda: defaultdict(int))
    expert_by_user  = defaultdict(lambda: defaultdict(int))

    for b in loader:
        x = (b["x"] if isinstance(b, dict) else b[0]).to(device)
        y = (b["y"] if isinstance(b, dict) else b[1]).to(device)
        uids = (b["user_id"] if isinstance(b, dict) else (b[2] if len(b) > 2 else torch.zeros_like(y))).to(device)

        idx, val = route_topk(model, x, user_ids=uids, user_embed_override=None, top_k=top_k)  # (B,k)
        B, k = idx.shape
        for i in range(B):
            cls = int(y[i].item())
            uid = int(uids[i].item()) if uids is not None else -1
            for j in range(k):
                e = int(idx[i,j].item())
                expert_total[e] += 1
                expert_by_class[e][cls] += 1
                expert_by_user[e][uid]  += 1

    return expert_total, expert_by_class, expert_by_user

def print_routing_summary(expert_total, expert_by_class, expert_by_user, num_classes=10, top_m_users=5):
    E = len(expert_total)
    print(f"#experts with traffic: {E}")
    for e in sorted(expert_total.keys()):
        tot = expert_total[e]
        # top classes
        cls_counts = expert_by_class[e]
        top_cls = sorted(cls_counts.items(), key=lambda x: -x[1])[:5]
        # top users
        usr_counts = expert_by_user[e]
        top_usr = sorted(usr_counts.items(), key=lambda x: -x[1])[:top_m_users]
        print(f"\nExpert {e}: total={tot}")
        print("  top classes:", top_cls)
        print("  top users:  ", top_usr)
