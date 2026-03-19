# Currently these funcs do not get called at all, all the used funcs are in run_$B.ipynb

import torch
import random
from typing import Optional, List

# =============================================================================
# Exact $B Classifier
# =============================================================================

def dollar_B_classify(
    sup_emg:    torch.Tensor,   # (N_sup, C_emg, T)
    sup_imu:    Optional[torch.Tensor], # (N_sup, C_imu, T)
    sup_labels: torch.Tensor,   # (N_sup,)
    qry_emg:    torch.Tensor,   # (N_qry, C_emg, T)
    qry_imu:    Optional[torch.Tensor], # (N_qry, C_imu, T)
    n_components: int,
    use_imu:    bool,
) -> torch.Tensor:
    """
    True $B classifier:
      1. Early fusion (concatenate EMG & IMU vertically).
      2. Per-TEMPLATE (per-sample) PCA.
      3. 1-Nearest Neighbor based on L1 distance in each template's latent space.
    """
    # 1. Early Concatenation of Modalities
    if use_imu and sup_imu is not None and qry_imu is not None:
        sup = torch.cat([sup_emg, sup_imu], dim=1)  # (N_sup, C_total, T)
        qry = torch.cat([qry_emg, qry_imu], dim=1)  # (N_qry, C_total, T)
    else:
        sup = sup_emg
        qry = qry_emg

    N_sup, C, T = sup.shape
    N_qry = qry.shape[0]
    n_pc = min(n_components, C - 1)
    
    # Distance from each query to each INDIVIDUAL support template: (N_qry, N_sup)
    all_dists = torch.zeros(N_qry, N_sup, device=qry.device)

    # 2. Per-Template Processing
    for i in range(N_sup):
        template_ts = sup[i]                     # (C, T)
        mean_c = template_ts.mean(dim=1, keepdim=True) # (C, 1)
        x_centered = template_ts - mean_c        # (C, T)
        
        # Fit PCA on this exact template
        cov = (x_centered @ x_centered.t()) / (T - 1)
        
        try:
            _, vecs = torch.linalg.eigh(cov)
            U = vecs[:, -n_pc:].flip(dims=[1])   # (C, n_pc)
        except Exception:
            U = torch.eye(C, device=sup.device)[:, :n_pc]
            
        # Transform template to its own latent space and flatten
        proj_t = (U.t() @ x_centered).flatten()  # (n_pc * T,)
        
        # 3. Project all queries into THIS template's latent space
        for q in range(N_qry):
            xq_centered = qry[q] - mean_c        # (C, T)
            proj_q = (U.t() @ xq_centered).flatten() # (n_pc * T,)
            
            # L1 Distance
            all_dists[q, i] = (proj_q - proj_t).abs().sum()

    # 1-NN: find the single closest template index
    best_template_idx = all_dists.argmin(dim=1)  # (N_qry,)
    
    # Return the label of the winning template
    return sup_labels[best_template_idx]         # (N_qry,)


# =============================================================================
# Per-User Evaluation for $B
# =============================================================================

def eval_one_user_dollar_B(
    user_data:    dict,
    k_shot:       int,
    n_way:        int,
    support_reps: Optional[List[int]],
    query_reps:   Optional[List[int]],
    all_reps:     List[int],
    rng:          random.Random,
    device:       torch.device,
    use_imu:      bool,
    n_components: int,
    config:       dict,
) -> dict:
    
    # (Assuming build_episode is imported from your existing file)
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

    n_support_orig = sup_emg.size(0)

    # Note: I removed augmentation here, $B normally doesn't augment the templates!
    
    # Forward Pass: Get predictions directly from our $B classifier
    preds = exact_dollar_B_classify(
        sup_emg=sup_emg, sup_imu=sup_imu, sup_labels=sup_labels,
        qry_emg=qry_emg, qry_imu=qry_imu,
        n_components=n_components, use_imu=use_imu
    )
    
    # Score the predictions
    acc = (preds == qry_labels).float().mean().item()

    return {
        "dollar_B":  acc,
        "n_support": n_support_orig,
        "n_query":   qry_emg.size(0),
    }

# =============================================================================
# Single-Shot-Condition Runner (across all users)
# =============================================================================

def run_one_shot_condition_dollar_B(
    tensor_dict: dict,
    config:      dict,
    k_shot:      int,
    verbose:     bool = True,
) -> dict:
    
    device       = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    n_way        = int(config.get("n_way", 10))
    seed         = int(config.get("seed", 42))
    use_imu      = bool(config.get("use_imu", False))
    n_components = int(config.get("n_components", 50)) # $B paper used 50
    all_reps     = config["maml_reps"]
    eval_pids    = config.get("eval_PIDs", config.get("val_PIDs", []))

    if not eval_pids:
        raise ValueError("config must contain 'eval_PIDs'.")

    # ── Support / query rep split ──
    if "support_reps" in config and "query_reps" in config:
        support_reps = config["support_reps"]
        query_reps   = config["query_reps"]
    elif config.get("use_fixed_rep_split", True):
        sorted_reps  = sorted(all_reps)
        support_reps = sorted_reps[:k_shot]
        query_reps   = sorted_reps[k_shot:]
    else:
        support_reps = None
        query_reps   = None

    rng = random.Random(seed)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  EXACT $B EVAL: {k_shot}-shot  {n_way}-way")
        print(f"  n_components : {n_components} (EMG+IMU Early Fusion)")
        print(f"  users        : {len(eval_pids)}")
        print(f"{'='*65}")

    accs = []
    
    for pid in eval_pids:
        if pid not in tensor_dict:
            continue
            
        user_data = tensor_dict[pid]
        available = [g for g in all_reps if g in user_data]
        if len(available) < n_way: continue

        res = eval_one_user_dollar_B(
            user_data=user_data, k_shot=k_shot, n_way=n_way,
            support_reps=support_reps, query_reps=query_reps, all_reps=all_reps,
            rng=rng, device=device, use_imu=use_imu, n_components=n_components,
            config=config
        )
        
        accs.append(res["dollar_B"])
        
        if verbose:
            print(f"  PID {str(pid):>4} | exact_$B = {res['dollar_B']*100:>5.1f}%  "
                  f"[sup={res['n_support']}, qry={res['n_query']}]")

    mean_acc = torch.tensor(accs).mean().item()
    std_acc  = torch.tensor(accs).std().item()

    if verbose:
        print(f"\n  {'Method':<30}  {'Mean':>8}  {'Std':>7}  {'Users':>6}")
        print(f"  {'─'*30}  {'─'*8}  {'─'*7}  {'─'*6}")
        print(f"  {'Exact $B ($B-faithful)':<30}  {mean_acc*100:>7.2f}%  {std_acc*100:>6.2f}%  {len(accs):>6}")

    return {"mean_acc": mean_acc, "std_acc": std_acc, "all_accs": accs}