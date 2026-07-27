"""
portA_support_routing.py
========================
MoEMeta Port A: support-derived routing.

THE QUESTION
------------
MoEMeta (Wu & Yin, NeurIPS 2025) computes routing per SUPPORT triplet and
consumes the result as a single task-level object (their Eqs. 5-7):

    s_{i,j} = softmax(Gate(h_i, t_i))    # per support item i
    g_{i,j} = TopN({s_{i,j}}, N)         # per support item i
    r_i     = (1/N) sum_j g_{i,j} f_j(.) # per support item i
    R_T     = (1/K) sum_i r_i            # task-level aggregation HERE

Query items are never routed -- the query is scored against R'_T and the MoE
never sees it. EncoderMoE routes every sample including queries, which is the
sharpest technical distinction between the two methods and is already asserted
at L487-488 ("routing operates directly on the query input and does not depend
on support set size").

This script tests that design choice IN OUR REGIME by replacing our
query-conditioned routing with a support-derived gate vector, holding
everything else fixed. Our prediction is that query-conditioned routing wins
at K=1, precisely because support-derived routing has one example to work
from. If that holds, a novelty attack becomes a positive empirical result.

WHY THIS IS CHEAP
-----------------
No retraining. It evaluates an existing M0 checkpoint under two routing
regimes on the SAME episodes, so the comparison is paired.

WHAT THIS IS NOT
----------------
Not a MoEMeta baseline. MoEMeta cannot be run on this data (it needs a graph,
a candidate set and ranking objective, and symbolic entity embeddings, and its
held-out axis is relations while ours is users). This is a transfer of ONE of
its design decisions. Port B -- the frozen-global / adapt-a-small-local-module
regime -- is the other, and must be META-TRAINED with the restricted inner
loop rather than restricted only at eval time.

A DISCLOSED DESIGN CHOICE
-------------------------
The support-derived gate vector is computed from the PRE-adaptation model and
then held fixed through both the inner loop and query evaluation. The
alternative -- recomputing it from adapted parameters -- would make the routing
source and the adaptation interact, which muddies exactly the axis under test.
State this choice in the caption; it is a diagnostic, not a tuned baseline.

Usage (NOTS):
    python portA_support_routing.py --checkpoint /path/to/best_M0_model.pt
    python portA_support_routing.py --checkpoint ... --k-shot 5 --num-episodes 200
"""

import os
import sys
import json
import copy
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))

from ablation_config import (
    make_base_config, build_maml_moe_model, set_seeds, FIXED_SEED,
    save_results, count_parameters, NUM_TEST_EPISODES,
)

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------

def support_gate_vector(model, config, support, renormalise: bool = True,
                        reapply_topk: bool = True) -> torch.Tensor:
    """
    One gate vector per episode, derived from the support set only.

    Order of operations follows MoEMeta: per-item routing WITH top-k applied,
    then average over the K*N support items. Averaging post-top-k weights (not
    pre-softmax logits) is what makes this their aggregation rather than ours.

    Averaging destroys sparsity -- the mean of several k-sparse vectors is
    generally denser than k. `reapply_topk` restores the configured sparsity
    so the forced routing has the same utilisation as normal operation;
    without it Port A would quietly become a denser model and any accuracy
    difference would confound routing source with utilisation.
    """
    device = config["device"]
    model.eval()
    with torch.no_grad():
        _, routing = model(
            support["emg"].to(device),
            support["imu"].to(device) if (config["use_imu"] and support.get("imu") is not None) else None,
            support["demo"].to(device) if support.get("demo") is not None else None,
            return_routing=True,
        )
        w = routing["gate_weights"]                 # (B_support, E)
        v = w.mean(dim=0)                            # (E,)

        top_k = config.get("MOE_top_k", None)
        if reapply_topk and top_k is not None and int(top_k) < v.numel():
            k = int(top_k)
            _, idx = torch.topk(v, k)
            mask = torch.zeros_like(v).scatter_(0, idx, 1.0)
            v = v * mask
        if renormalise:
            s = v.sum()
            v = v / s if float(s) > 0 else torch.full_like(v, 1.0 / v.numel())
    return v.detach()


def per_sample_diagnostics(model, config, batch_part) -> dict:
    """
    Entropy of the NORMAL per-sample gate weights.

    This is the number that explains a null result. If per-sample routing is
    itself close to uniform, then "route each sample" and "route once from the
    support set" are both approximately "average the top-k experts equally", and
    the two regimes are near-identical BY CONSTRUCTION rather than because the
    routing source is unimportant.

    Compared against ln(top_k), not ln(num_experts): with top-k gating only
    top_k experts are ever active, so ln(num_experts) is unreachable and using it
    makes routing look far more peaked than it is.
    """
    device = config["device"]
    model.eval()
    with torch.no_grad():
        _, routing = model(
            batch_part["emg"].to(device),
            batch_part["imu"].to(device) if (config["use_imu"] and batch_part.get("imu") is not None) else None,
            batch_part["demo"].to(device) if batch_part.get("demo") is not None else None,
            return_routing=True,
        )
        W = routing["gate_weights"]                       # (B, E)
        P = W / W.sum(dim=1, keepdim=True).clamp_min(1e-12)
        ent = -(torch.where(P > 0, P * P.log(), torch.zeros_like(P))).sum(dim=1)
        top_k = int(config.get("MOE_top_k", W.shape[1]))
        ceil_topk = float(np.log(max(top_k, 2)))
        return {
            "entropy_mean":        float(ent.mean()),
            "entropy_vs_topk_ceil": float(ent.mean()) / ceil_topk,
            "max_weight_mean":     float(P.max(dim=1).values.mean()),
            "n_active_mean":       float((W > 0).float().sum(dim=1).mean()),
            "topk_entropy_ceiling": ceil_topk,
        }


def routing_diagnostics(v: torch.Tensor) -> dict:
    """Summarise a gate vector so the two regimes can be compared structurally."""
    p = v / v.sum().clamp_min(1e-12)
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum())
    return {
        "n_active":    int((v > 0).sum()),
        "max_weight":  float(v.max()),
        "entropy":     entropy,
        "entropy_norm": entropy / float(np.log(v.numel())) if v.numel() > 1 else 0.0,
    }


def run(args) -> dict:
    config = make_base_config(ablation_id="portA")
    config["test_procedure"] = "hpo_test_split"

    # With only 4 test users a paired test has almost no power, so allow the val
    # users in as well. Legitimate here BECAUSE this is a paired within-episode
    # diagnostic of a frozen checkpoint: both regimes see identical episodes and
    # nothing is selected or tuned on these users. It would NOT be legitimate for
    # a headline accuracy number, and the JSON records which users were used.
    eval_pids = list(config["test_PIDs"])
    if args.include_val_users:
        eval_pids = list(config["val_PIDs"]) + eval_pids
        print(f"[portA] including val users -> {len(eval_pids)} users total")
    if args.k_shot is not None:
        config["k_shot"] = args.k_shot
    if args.n_way is not None:
        config["n_way"] = args.n_way

    set_seeds(FIXED_SEED)
    device = config["device"]

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_maml_moe_model(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("best_state", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[portA] WARNING load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        print(f"        missing[:5]={list(missing)[:5]}")
        print(f"        unexpected[:5]={list(unexpected)[:5]}")
    print(f"[portA] Loaded checkpoint: {args.checkpoint}")
    print(f"[portA] Parameters: {count_parameters(model):,}")

    assert hasattr(model, "expert_cnns"), (
        "Port A requires an MoE model exposing .expert_cnns (DeepCNNLSTM_EncoderMOE). "
        f"Got {type(model).__name__}."
    )
    assert hasattr(model, "gate"), "Model has no .gate -- cannot override routing."

    # ── Data ─────────────────────────────────────────────────────────────────
    from MAML.maml_data_pipeline import (
        MetaGestureDataset, maml_mm_collate, reorient_tensor_dict,
    )
    from MAML.mamlpp import mamlpp_adapt_and_eval
    from torch.utils.data import DataLoader

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")
    with open(tensor_dict_path, "rb") as f:
        full_dict = pickle.load(f)
    tensor_dict = reorient_tensor_dict(full_dict, config)

    test_ds = MetaGestureDataset(
        tensor_dict,
        target_pids             = eval_pids,
        target_gesture_classes  = config["maml_gesture_classes"],
        target_trial_reps       = config["target_trial_reps"],
        n_way                   = config["n_way"],
        k_shot                  = config["k_shot"],
        q_query                 = config["q_query"],
        num_eval_episodes       = args.num_episodes,
        is_train                = False,
        seed                    = FIXED_SEED,
        use_label_shuf_meta_aug = False,
    )
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=4, collate_fn=maml_mm_collate)

    print(f"[portA] {config['n_way']}-way {config['k_shot']}-shot, "
          f"{args.num_episodes} episodes/user over {len(eval_pids)} users: {eval_pids}")

    # ── Paired evaluation ────────────────────────────────────────────────────
    per_user = defaultdict(lambda: {"query_routed": [], "support_routed": []})
    diag_accum = []
    persample_accum = []

    for ep_idx, batch in enumerate(test_dl):
        uid = batch["user_id"]
        support, query = batch["support"], batch["query"]

        # Regime 1: ours -- per-sample routing, queries routed.
        model._forced_gate_weights = None
        m1 = mamlpp_adapt_and_eval(model, config, support, query)
        per_user[uid]["query_routed"].append(m1["acc"])

        # Regime 2: MoEMeta-style -- one support-derived gate vector, queries
        # never routed. Same episode, same support set, same adaptation budget.
        v = support_gate_vector(model, config, support)
        model._forced_gate_weights = v
        try:
            m2 = mamlpp_adapt_and_eval(model, config, support, query)
        finally:
            # Always clear, or every later episode silently inherits this vector.
            model._forced_gate_weights = None
        per_user[uid]["support_routed"].append(m2["acc"])

        diag_accum.append(routing_diagnostics(v))
        if ep_idx < 50:      # bounded: enough for a stable mean, cheap
            persample_accum.append(per_sample_diagnostics(model, config, query))

        if ep_idx < 3 or (ep_idx + 1) % 100 == 0:
            print(f"  [ep {ep_idx+1}] user={uid}  "
                  f"query_routed={m1['acc']*100:.1f}%  "
                  f"support_routed={m2['acc']*100:.1f}%")

    # ── Aggregate, per user then across users ────────────────────────────────
    users = sorted(per_user.keys())
    q_means = np.array([np.mean(per_user[u]["query_routed"]) for u in users])
    s_means = np.array([np.mean(per_user[u]["support_routed"]) for u in users])
    deltas = q_means - s_means

    print(f"\n  Per-user means ({len(users)} users):")
    for u, a, b in zip(users, q_means, s_means):
        print(f"    {u}  query={a*100:6.2f}%  support={b*100:6.2f}%  "
              f"delta={(a-b)*100:+6.2f}")

    # Paired test across users. n is small (4 on the fixed split), so report the
    # effect size and per-user direction rather than leaning on the p-value.
    p_value = None
    t_stat = None
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(q_means, s_means)
        t_stat, p_value = float(t_stat), float(p_value)
    except Exception as e:
        print(f"  (paired t-test skipped: {e})")

    d_z = float(deltas.mean() / deltas.std(ddof=1)) if len(deltas) > 1 and deltas.std(ddof=1) > 0 else None

    diag_mean = {k: float(np.mean([d[k] for d in diag_accum])) for k in diag_accum[0]} if diag_accum else {}
    persample_mean = ({k: float(np.mean([d[k] for d in persample_accum]))
                      for k in persample_accum[0]} if persample_accum else {})

    if persample_mean:
        r = persample_mean["entropy_vs_topk_ceil"]
        print(f"\n  Normal per-sample routing: entropy {persample_mean['entropy_mean']:.4f} "
              f"= {r*100:.1f}% of the top-k ceiling "
              f"({persample_mean['topk_entropy_ceiling']:.4f})")
        print(f"  max gate weight (mean) {persample_mean['max_weight_mean']:.4f}, "
              f"active experts {persample_mean['n_active_mean']:.2f}")
        if r > 0.95:
            print("  -> Per-sample routing is itself near-UNIFORM. That largely explains")
            print("     any null: both regimes reduce to averaging the active experts")
            print("     roughly equally, so the routing SOURCE cannot matter much. Treat")
            print("     this as evidence about how much the gate specialises at all, and")
            print("     check it before defending the section 4.6 specialisation claim.")
        else:
            print("  -> Per-sample routing is meaningfully peaked, so a null result is")
            print("     NOT explained by uniform gating and is a real finding about the")
            print("     routing source.")

    result = {
        "ablation_id":   "portA_support_routing",
        "description":   ("MoEMeta Port A: routing derived from the support set only, "
                          "aggregated to one gate vector per episode, applied to all "
                          "queries. Evaluated from an existing M0 checkpoint; no retraining."),
        "checkpoint":    str(args.checkpoint),
        "n_way":         config["n_way"],
        "k_shot":        config["k_shot"],
        "num_episodes_per_user": args.num_episodes,
        "users":         users,
        "query_routed_mean":   float(q_means.mean()),
        "query_routed_std":    float(q_means.std()),
        "support_routed_mean": float(s_means.mean()),
        "support_routed_std":  float(s_means.std()),
        "delta_mean":          float(deltas.mean()),
        "n_users_favouring_query_routing": int((deltas > 0).sum()),
        "n_users":             len(users),
        "paired_t":            t_stat,
        "paired_p":            p_value,
        "cohens_d_z":          d_z,
        "forced_routing_diagnostics": diag_mean,
        "per_sample_routing_diagnostics": persample_mean,
        "eval_users": eval_pids,
        "included_val_users": bool(args.include_val_users),
        "design_notes": [
            "Support gate vector computed from the PRE-adaptation model and held "
            "fixed through the inner loop and query evaluation.",
            "Top-k re-applied after averaging so utilisation matches normal "
            "operation; without this, routing source would be confounded with "
            "expert utilisation.",
            "Paired over the SAME episodes, so the comparison is within-episode.",
            "Fixed 24/4/4 split: 4 test users, so treat the paired p-value as "
            "indicative and report per-user direction.",
        ],
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(result, config, tag=f"portA_k{config['k_shot']}_n{config['n_way']}")

    print(f"\n{'='*70}")
    print(f"[portA] query-conditioned routing (ours) : "
          f"{q_means.mean()*100:.2f}% ± {q_means.std()*100:.2f}%")
    print(f"[portA] support-derived routing (MoEMeta): "
          f"{s_means.mean()*100:.2f}% ± {s_means.std()*100:.2f}%")
    print(f"[portA] delta = {deltas.mean()*100:+.2f} points, favouring ours in "
          f"{int((deltas>0).sum())}/{len(users)} users")
    if p_value is not None:
        print(f"[portA] paired t={t_stat:.3f}  p={p_value:.4f}  d_z={d_z}")
    print(f"{'='*70}")
    return result


def main():
    ap = argparse.ArgumentParser(description="MoEMeta Port A: support-derived routing.")
    ap.add_argument("--checkpoint", required=True,
                    help="M0 checkpoint (e.g. models/final_eval_models/best_M0_model.pt)")
    ap.add_argument("--num-episodes", type=int, default=NUM_TEST_EPISODES,
                    help="Episodes per test user (default matches the ablation suite).")
    ap.add_argument("--k-shot", type=int, default=None)
    ap.add_argument("--n-way", type=int, default=None)
    ap.add_argument("--include-val-users", action="store_true",
                    help="Evaluate on val+test users (8) instead of test only (4). "
                         "Valid for this paired diagnostic; not for a headline number.")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. This must run on NOTS."
        )
    run(args)


if __name__ == "__main__":
    main()
