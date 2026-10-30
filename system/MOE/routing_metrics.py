"""
routing_metrics.py
==================
Routing diagnostics that can tell routing from ensembling (X1 vs X2 in
docs/specs/SPEC_architecture_exploration.md). Replaces the numbers the old RoutingAnalyzer printed, which
were misleading (docs/run_reports/10_29_26_debug_run_results.md, issues 5-7):

  * entropy was measured on the post-top-k weights, so it sat at ln(top_k) whatever the gate learned.
    Here: per-sample entropy of the PRE-top-k softmax, normalised by ln(E) (1 = the router has no
    preference for any expert = pure ensembling);
  * imbalance ratios divided by ~0 for dead experts. Here: no ratios; load CV, max-load / uniform,
    and the fraction of dead experts;
  * per-user routing printed "?". Here: user and gesture IDs come from the episodes, and routing
    specialisation is reported as mutual information I(expert; user) and I(expert; gesture), each
    with a label-permutation null (finite samples alone give MI > 0).

All functions take numpy arrays or tensors; nothing here touches the model except
collect_episode_routing().
"""

from __future__ import annotations

import numpy as np


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _entropy(p, axis=-1):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def _mutual_info(w: np.ndarray, groups: np.ndarray) -> float:
    """I(expert; group) in nats, routing p(e | sample) = w (rows sum to 1), groups equally weighted by samples."""
    p_e = w.mean(axis=0)
    h_cond = 0.0
    for g in np.unique(groups):
        m = groups == g
        h_cond += m.mean() * _entropy(w[m].mean(axis=0))
    return float(_entropy(p_e) - h_cond)


def routing_summary(w_soft, w_hard, user_ids=None, gesture_ids=None, seed: int = 0) -> dict:
    """
    w_soft, w_hard: (N, E) pre- and post-top-k gate weights (identical for dense routing).
    user_ids, gesture_ids: optional length-N labels for the specialisation metrics.
    """
    ws, wh = _np(w_soft), _np(w_hard)
    if ws.ndim != 2 or ws.shape != wh.shape:
        raise ValueError(f"w_soft {ws.shape} and w_hard {wh.shape} must both be (N, E)")
    N, E = ws.shape
    lnE = np.log(E) if E > 1 else 1.0
    load = wh.mean(axis=0)                       # mean weight per expert
    sel = (wh > 0).mean(axis=0)                  # fraction of samples selecting each expert
    out = {
        "n_samples": int(N),
        "num_experts": int(E),
        # router sharpness per sample (1.0 = uniform soft weights = no routing preference)
        "soft_entropy_norm_mean": float(_entropy(ws).mean() / lnE),
        "soft_entropy_norm_std": float(_entropy(ws).std() / lnE),
        # load balance across the batch (1.0 = perfectly balanced usage)
        "usage_entropy_norm": float(_entropy(load / load.sum()) / lnE),
        "load_cv": float(load.std() / (load.mean() + 1e-12)),
        "max_load_over_uniform": float(load.max() * E),
        "dead_expert_frac": float((sel < 0.01).mean()),
        "selection_freq": sel.round(4).tolist(),
        "mean_load": load.round(4).tolist(),
    }
    rng = np.random.default_rng(seed)
    for name, ids in (("user", user_ids), ("gesture", gesture_ids)):
        if ids is None:
            continue
        ids = np.asarray([str(i) for i in ids])
        if len(ids) != N:
            raise ValueError(f"{name}_ids has length {len(ids)}, expected {N}")
        n_groups = len(np.unique(ids))
        mi = _mutual_info(wh, ids)
        null = float(np.mean([_mutual_info(wh, rng.permutation(ids)) for _ in range(20)]))
        denom = min(np.log(E), np.log(n_groups)) if (E > 1 and n_groups > 1) else 1.0
        out[f"mi_{name}"] = mi
        out[f"mi_{name}_null"] = null
        out[f"mi_{name}_norm"] = float(mi / denom)       # 0 = no specialisation, 1 = deterministic
        out[f"n_{name}s"] = int(n_groups)
    if user_ids is not None and gesture_ids is not None:
        # gestures are user-defined, so "gesture" only means something within a user:
        # I(expert; gesture | user), averaged over users, with a within-user permutation null
        u = np.asarray([str(i) for i in user_ids])
        g = np.asarray([str(i) for i in gesture_ids])
        mi_w, null_w = 0.0, 0.0
        for uid in np.unique(u):
            m = u == uid
            mi_w += m.mean() * _mutual_info(wh[m], g[m])
            null_w += m.mean() * float(np.mean([_mutual_info(wh[m], rng.permutation(g[m])) for _ in range(20)]))
        out["mi_gesture_within_user"] = float(mi_w)
        out["mi_gesture_within_user_null"] = float(null_w)
    return out


def format_summary(s: dict) -> str:
    lines = [
        f"  router sharpness: soft entropy / ln E = {s['soft_entropy_norm_mean']:.3f} "
        f"(+/- {s['soft_entropy_norm_std']:.3f}; 1.0 = uniform = ensembling)",
        f"  load balance: usage entropy / ln E = {s['usage_entropy_norm']:.3f}, load CV = {s['load_cv']:.3f}, "
        f"max load = {s['max_load_over_uniform']:.2f}x uniform, dead experts = {s['dead_expert_frac']*100:.0f}%",
    ]
    for name in ("user", "gesture"):
        if f"mi_{name}" in s:
            lines.append(f"  I(expert; {name}) = {s[f'mi_{name}']:.4f} nats (permutation null "
                         f"{s[f'mi_{name}_null']:.4f}; normalised {s[f'mi_{name}_norm']:.3f}, "
                         f"{s[f'n_{name}s']} {name}s)")
    if "mi_gesture_within_user" in s:
        lines.append(f"  I(expert; gesture | user) = {s['mi_gesture_within_user']:.4f} nats "
                     f"(within-user permutation null {s['mi_gesture_within_user_null']:.4f})")
    return "\n".join(lines)


def collect_episode_routing(model, config, episodic_loader, adapted: bool = True) -> dict | None:
    """
    Routing on the query sets of an episodic loader, after inner-loop adaptation (adapted=True,
    what the model actually predicts with) or at the meta-learned init (adapted=False).
    Returns routing_summary(...) or None if the model exposes no routing.
    """
    import torch
    from MAML.mamlpp import FunctionalModel, mamlpp_adapt, named_param_dict, task_support

    device = config["device"]
    ws, wh, users, gestures = [], [], [], []
    for step_item in episodic_loader:
        for ep in ([step_item] if isinstance(step_item, dict) else step_item):
            sup, qry = ep["support"], ep["query"]
            with task_support(model, sup, device):
                theta = mamlpp_adapt(model, config, sup) if adapted else named_param_dict(model)
                model.eval()
                with torch.no_grad():
                    imu = qry.get("imu")
                    out = FunctionalModel(model, theta)(
                        qry["emg"].to(device), imu.to(device) if imu is not None else None,
                        return_routing=True)
                model.train()
            if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict)):
                return None
            info = out[1]
            n, T = qry["emg"].shape[0], qry["emg"].shape[-1]
            h = info["gate_weights"].cpu()
            s = info.get("gate_weights_soft", info["gate_weights"]).cpu()
            if h.shape[0] != n:
                # token-level MoE (T4): rows are (block, sample, time step), block-major
                # (attention.merge_routing). Report per-sample routing = mean over blocks and steps.
                if h.shape[0] % (n * T) != 0:
                    raise ValueError(f"gate_weights has {h.shape[0]} rows; expected a multiple of "
                                     f"n*T = {n}*{T} for token routing")
                nb = h.shape[0] // (n * T)
                h = h.reshape(nb, n, T, -1).mean(dim=(0, 2))
                s = s.reshape(nb, n, T, -1).mean(dim=(0, 2))
            wh.append(h)
            ws.append(s)
            users += [str(ep["user_id"])] * n
            # global label = the command (shared across users); its execution is user-defined
            gestures += [str(g) for g in qry["global_labels"].tolist()]
    if not wh:
        return None
    import torch as _t
    return routing_summary(_t.cat(ws), _t.cat(wh), user_ids=users, gesture_ids=gestures)
