import torch
import numpy as np
from collections import defaultdict

def meta_evaluate(model, episodic_loader, config, adapt_and_eval_fn, print_per_user=False):
    """
    A shared eval function.
    adapt_and_eval_fn: This will be either maml_adapt_and_eval or mamlpp_adapt_and_eval

    The val/test DataLoader is a single flat stream of episodes from ALL users in
    target_pids (MetaGestureDataset pre-computes num_eval_episodes per user and
    concatenates them). Per-user accuracy is recovered by reading episode["user_id"].

    Per-user acc is computed as the mean episode accuracy for that user (each
    episode contributes equally, regardless of q_size variation). This is the
    standard few-shot eval convention and avoids lossy int rounding.

    Args:
        print_per_user: If True, prints one line per user after aggregation.
    Returns:
        dict with keys: "loss", "acc", "per_user_acc"
    """
    model.train() # Requirement for RNN gradients
    total_loss = total_correct = total_count = n_eps = 0
    pre_adapt_accs = []

    # Per-user tracking: uid -> list of per-episode accuracy floats
    user_episode_accs: dict = defaultdict(list)

    for step_item in episodic_loader:
        episodes = [step_item] if isinstance(step_item, dict) else step_item
        for ep in episodes:

            metrics = adapt_and_eval_fn(model, config, ep["support"], ep["query"])
            
            # Global aggregate (unchanged from original)
            q_size = len(ep["query"]["labels"]) if isinstance(ep["query"], dict) else len(ep["query"][1])
            total_loss    += metrics["loss"]
            total_correct += (metrics["acc"] * q_size)
            total_count   += q_size
            n_eps         += 1

            # Per-user: accumulate one float per episode.
            # user_id is set at the episode level by maml_mm_collate.
            uid = ep.get("user_id")
            if uid is not None:
                user_episode_accs[uid].append(metrics["acc"])

            if metrics.get("pre_adapt_acc") is not None:
                pre_adapt_accs.append(metrics["pre_adapt_acc"])

    if pre_adapt_accs:
        arr = np.array(pre_adapt_accs)
        print(f"  [Debug] Pre-adapt acc: {arr.mean():.4f} ± {arr.std():.4f}  (n={len(arr)})")

    per_user_acc = {
        uid: float(np.mean(accs))
        for uid, accs in user_episode_accs.items()
    }

    if print_per_user and per_user_acc:
        for uid, acc in sorted(per_user_acc.items()):
            print(f"  [Val] user={uid}  acc={acc*100:.2f}%")

    return {
        "loss":         total_loss / max(n_eps, 1),
        "acc":          total_correct / max(total_count, 1),
        "per_user_acc": per_user_acc,
    }

def calculate_gradient_alignment(task_gradients):
    """
    task_gradients: List of tuples/lists containing gradients for each task in the meta-batch.
                    e.g., [task1_grads, task2_grads, ...]
    """
    alignments = []
    num_tasks = len(task_gradients)
    
    if num_tasks < 2:
        return 0.0

    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            grad_i_flat = []
            grad_j_flat = []
            
            # Iterate through paired parameters
            for g_i, g_j in zip(task_gradients[i], task_gradients[j]):
                # If a module is entirely unused (e.g. inactive MoE expert or disabled demo encoder), 
                # BOTH tasks will likely have None. We just skip it entirely.
                if g_i is None and g_j is None:
                    continue
                
                # If MOE is active, one task might use an expert and the other might not.
                # In this case, we MUST treat the unused one as zeros to correctly penalize the alignment.
                t_g_i = g_i.flatten() if g_i is not None else torch.zeros_like(g_j).flatten()
                t_g_j = g_j.flatten() if g_j is not None else torch.zeros_like(g_i).flatten()
                
                grad_i_flat.append(t_g_i)
                grad_j_flat.append(t_g_j)
            
            if len(grad_i_flat) == 0:
                continue

            v_i = torch.cat(grad_i_flat)
            v_j = torch.cat(grad_j_flat)
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(v_i.unsqueeze(0), v_j.unsqueeze(0)).item()
            alignments.append(cos_sim)
            
    return sum(alignments) / len(alignments) if alignments else 0.0