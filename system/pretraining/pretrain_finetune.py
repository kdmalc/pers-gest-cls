"""
pretrain_finetune.py
====================
Few-shot finetuning evaluation for a pretrained (non-MAML) DeepCNNLSTM.

Mirrors the MAML evaluation protocol so results are directly comparable:
  - 1-shot, N-way (default 3-way, same as the MAML HPO)
  - Per-user episodic evaluation: sample episodes, finetune, evaluate on query set
  - Two finetuning modes:
      'full'      — update all parameters
      'head_only' — freeze backbone, update only the classification head

This is intentionally a minimal, faithful translation of mamlpp_adapt_and_eval()
into standard SGD/Adam finetuning.  The goal is a fair ablation: same task
structure, same number of support samples, same query set, different optimizer.

Usage:
    from pretrain_finetune import finetune_and_eval_user, evaluate_all_val_users

    # Evaluate one user, one episode
    metrics = finetune_and_eval_user(
        model, config, support_emg, support_imu, support_labels,
        query_emg, query_imu, query_labels,
        mode='full',
    )
    print(metrics['acc'])

    # Evaluate all val users (episodic, averaged)
    user_metrics = evaluate_all_val_users(model, config, tensor_dict, mode='full')
"""

import copy
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Core: finetune on a single support set, evaluate on query set
# ─────────────────────────────────────────────────────────────────────────────

def finetune_and_eval_user(
    model: nn.Module,
    config: dict,
    support_emg:    torch.Tensor,   # (k_shot * n_way, C, T)
    support_imu:    torch.Tensor | None,
    support_labels: torch.Tensor,   # (k_shot * n_way,) — 0-indexed, remapped to 0..n_way-1
    query_emg:      torch.Tensor,   # (q_query * n_way, C, T)
    query_imu:      torch.Tensor | None,
    query_labels:   torch.Tensor,   # (q_query * n_way,)
    mode: Literal['full', 'head_only'] = 'full',
) -> dict:
    """
    Clone model, finetune on support set, evaluate on query set.

    The model is NOT modified in-place — we work on a deep copy so the
    caller's model state is preserved across episodes.

    Args:
        model       : pretrained model (DeepCNNLSTM or compatible)
        config      : must contain:
                        'ft_lr'           — finetuning learning rate
                        'ft_steps'        — number of gradient steps on the support set
                        'ft_optimizer'    — 'adam' or 'sgd' (default 'adam')
                        'ft_weight_decay' — weight decay (default 0.0)
                        'label_smooth'    — label smoothing (default 0.0 for finetuning)
                        'device'          — torch device
        support_*   : support set tensors (already on device is fine; will be moved if not)
        query_*     : query set tensors
        mode        : 'full' — all params trainable
                      'head_only' — only model.head parameters trainable

    Returns:
        dict with keys:
            'acc'        : float, query accuracy
            'loss'       : float, query CE loss (no grad)
            'ft_loss_log': list[float], support CE loss per step (diagnostics)
    """
    device = config['device']

    # Move data to device
    support_emg    = support_emg.to(device)
    query_emg      = query_emg.to(device)
    support_labels = support_labels.to(device)
    query_labels   = query_labels.to(device)
    if support_imu is not None:
        support_imu = support_imu.to(device)
    if query_imu is not None:
        query_imu = query_imu.to(device)

    # Work on a copy — never mutate the caller's model
    ft_model = copy.deepcopy(model)
    ft_model.train()
    ft_model.to(device)

    if mode == 'head_only':
        # Freeze everything except the classification head
        for p in ft_model.parameters():
            p.requires_grad_(False)
        for p in ft_model.head.parameters():
            p.requires_grad_(True)
    elif mode == 'full':
        for p in ft_model.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"Unknown finetuning mode: '{mode}'. Must be 'full' or 'head_only'.")

    trainable_params = [p for p in ft_model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(f"No trainable parameters found for mode='{mode}'.")

    ft_lr    = float(config['ft_lr'])
    ft_steps = int(config['ft_steps'])
    ft_wd    = float(config.get('ft_weight_decay', 0.0))
    ft_opt   = config.get('ft_optimizer', 'adam').lower()
    ls       = float(config.get('label_smooth', 0.0))

    if ft_opt == 'adam':
        optimizer = torch.optim.Adam(trainable_params, lr=ft_lr, weight_decay=ft_wd)
    elif ft_opt == 'sgd':
        optimizer = torch.optim.SGD(trainable_params, lr=ft_lr, weight_decay=ft_wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown ft_optimizer: '{ft_opt}'. Must be 'adam' or 'sgd'.")

    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    ft_loss_log = []
    for _ in range(ft_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = ft_model(x_emg=support_emg, x_imu=support_imu)
        # MoE models may return (logits, routing_info) — unwrap if needed
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits, support_labels)
        loss.backward()
        optimizer.step()
        ft_loss_log.append(loss.item())

    # Evaluate on query set — no grad
    ft_model.eval()
    with torch.no_grad():
        q_logits = ft_model(x_emg=query_emg, x_imu=query_imu)
        if isinstance(q_logits, tuple):
            q_logits = q_logits[0]
        q_loss = criterion(q_logits, query_labels)
        preds  = q_logits.argmax(dim=-1)
        acc    = (preds == query_labels).float().mean().item()

    return {
        'acc':         acc,
        'loss':        q_loss.item(),
        'ft_loss_log': ft_loss_log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Episode sampler: build one support/query episode from tensor_dict
# ─────────────────────────────────────────────────────────────────────────────

def _sample_episode(
    tensor_dict: dict,
    pid: str,
    n_way: int,
    k_shot: int,
    q_query: int,
    gesture_classes: list,
    support_rep_nums: list,   # 1-indexed rep nums eligible for support
    query_rep_nums:   list,   # 1-indexed rep nums eligible for query
    device: torch.device,
    use_imu: bool,
) -> dict:
    """
    Sample one N-way K-shot episode for a single user from tensor_dict.

    Returns a dict with support and query tensors, with labels remapped to 0..n_way-1.
    """
    assert len(gesture_classes) >= n_way, (
        f"Not enough gesture classes ({len(gesture_classes)}) for n_way={n_way}"
    )
    sampled_classes = random.sample(gesture_classes, n_way)

    support_emgs, support_imus, support_labels = [], [], []
    query_emgs,   query_imus,   query_labels   = [], [], []

    for new_label, orig_class in enumerate(sampled_classes):
        slot    = tensor_dict[pid][orig_class]
        emg_all = slot['emg']   # (n_trials, T, C) — channel last in tensor_dict
        imu_all = slot.get('imu', None)
        n_trials = emg_all.shape[0]

        # Convert 1-indexed rep nums to 0-indexed, filter to available trials
        avail_support = [r - 1 for r in support_rep_nums if 0 <= r - 1 < n_trials]
        avail_query   = [r - 1 for r in query_rep_nums   if 0 <= r - 1 < n_trials]

        assert len(avail_support) >= k_shot, (
            f"PID={pid}, class={orig_class}: need {k_shot} support reps, "
            f"only {len(avail_support)} available from reps {support_rep_nums}"
        )
        assert len(avail_query) >= q_query, (
            f"PID={pid}, class={orig_class}: need {q_query} query reps, "
            f"only {len(avail_query)} available from reps {query_rep_nums}"
        )

        sup_idxs = random.sample(avail_support, k_shot)
        qry_idxs = random.sample(avail_query,   q_query)

        # (T, C) → (C, T) for each selected rep
        for idx in sup_idxs:
            emg_ct = emg_all[idx].float().permute(1, 0)   # (C, T)
            support_emgs.append(emg_ct)
            support_labels.append(new_label)
            if use_imu and imu_all is not None:
                support_imus.append(imu_all[idx].float().permute(1, 0))

        for idx in qry_idxs:
            emg_ct = emg_all[idx].float().permute(1, 0)
            query_emgs.append(emg_ct)
            query_labels.append(new_label)
            if use_imu and imu_all is not None:
                query_imus.append(imu_all[idx].float().permute(1, 0))

    support_emg_t = torch.stack(support_emgs).to(device)   # (k*n, C, T)
    query_emg_t   = torch.stack(query_emgs).to(device)     # (q*n, C, T)
    support_lbl_t = torch.tensor(support_labels, dtype=torch.long).to(device)
    query_lbl_t   = torch.tensor(query_labels,   dtype=torch.long).to(device)

    support_imu_t = None
    query_imu_t   = None
    if use_imu and len(support_imus) > 0:
        support_imu_t = torch.stack(support_imus).to(device)
        query_imu_t   = torch.stack(query_imus).to(device)

    return {
        'support_emg':    support_emg_t,
        'support_imu':    support_imu_t,
        'support_labels': support_lbl_t,
        'query_emg':      query_emg_t,
        'query_imu':      query_imu_t,
        'query_labels':   query_lbl_t,
        'sampled_classes': sampled_classes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot evaluation (no finetuning — just forward pass on query set)
# ─────────────────────────────────────────────────────────────────────────────

def zeroshot_eval_user(
    model: nn.Module,
    config: dict,
    query_emg:    torch.Tensor,
    query_imu:    torch.Tensor | None,
    query_labels: torch.Tensor,
) -> dict:
    """
    Evaluate a pretrained model on the query set with no finetuning.

    The model predicts over all num_classes (not just the n_way subset),
    so we remap query_labels to the full class space using sampled_classes.
    Since the model was trained on all classes, zero-shot means directly
    running the pretrained head — no adaptation.

    NOTE: For a fair n_way zero-shot comparison, we restrict the argmax
    to only the n_way sampled class logits (same protocol as MAML eval).
    The caller must pass `sampled_classes` in config or as a separate arg
    for this remapping to work.  If `sampled_classes` is absent we fall
    back to evaluating over all output logits.

    Args:
        model        : pretrained model (eval mode set internally)
        config       : must contain 'device'; optionally 'sampled_classes'
        query_*      : query tensors with labels in 0..n_way-1 (episode-local)

    Returns:
        dict with 'acc' and 'loss'
    """
    device = config['device']
    sampled_classes = config.get('sampled_classes', None)

    query_emg    = query_emg.to(device)
    query_labels = query_labels.to(device)
    if query_imu is not None:
        query_imu = query_imu.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x_emg=query_emg, x_imu=query_imu)
        if isinstance(logits, tuple):
            logits = logits[0]

        if sampled_classes is not None:
            # Restrict argmax to the n_way classes that were sampled for this episode.
            # logits shape: (B, num_classes); sampled_classes: list of original class indices
            # query_labels are already remapped to 0..n_way-1, so we reindex logits accordingly.
            restricted = logits[:, sampled_classes]   # (B, n_way)
            preds = restricted.argmax(dim=-1)         # indices into sampled_classes (0..n_way-1)
        else:
            preds = logits.argmax(dim=-1)

        acc  = (preds == query_labels).float().mean().item()
        loss = nn.CrossEntropyLoss()(
            logits[:, sampled_classes] if sampled_classes is not None else logits,
            query_labels
        ).item()

    return {'acc': acc, 'loss': loss}


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop: all val users, multiple episodes each
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_val_users(
    model:       nn.Module,
    config:      dict,
    tensor_dict: dict,
    mode:        Literal['full', 'head_only', 'zero_shot'] = 'full',
) -> dict:
    """
    Episodic evaluation over all val users.

    For each val user, sample `num_eval_episodes` N-way K-shot episodes,
    run finetuning (or zero-shot forward) on each, and average accuracy.

    This is the non-MAML analogue of the MAML adapt-and-eval loop in the
    objective() function of the HPO script.

    Config keys consumed:
        val_PIDs               : list[str]
        n_way                  : int   (default 3)
        k_shot                 : int   (default 1)
        q_query                : int   (default 9)
        num_eval_episodes      : int   (default 10)
        maml_gesture_classes   : list[int] — which gesture classes to sample from
        target_trial_indices   : list[int] — 1-indexed rep nums available overall
        ft_support_reps        : list[int] — 1-indexed rep nums to use as support
                                 (default: [1], i.e. rep 1 is the support set)
        ft_query_reps          : list[int] — 1-indexed rep nums to use as query
                                 (default: all target_trial_indices except support)
        use_imu                : bool
        device                 : torch.device

    Returns:
        dict mapping pid → {'mean_acc': float, 'all_accs': list[float]}
        plus key '__overall__' → {'mean_acc': float, 'std_acc': float}
    """
    device      = config['device']
    val_pids    = config['val_PIDs']
    n_way       = int(config.get('n_way', 3))
    k_shot      = int(config.get('k_shot', 1))
    q_query     = int(config.get('q_query', 9))
    n_episodes  = int(config.get('num_eval_episodes', 10))
    use_imu     = bool(config.get('use_imu', True))
    all_classes = list(config['maml_gesture_classes'])
    all_reps    = list(config['target_trial_indices'])  # 1-indexed

    # Support/query rep split.  Default: rep 1 → support, rest → query.
    # This mirrors 1-shot: the user provides 1 example per class.
    support_reps = list(config.get('ft_support_reps', [all_reps[0]]))
    query_reps   = list(config.get('ft_query_reps',
                                    [r for r in all_reps if r not in support_reps]))

    assert len(support_reps) >= k_shot, (
        f"k_shot={k_shot} but only {len(support_reps)} support reps defined."
    )

    model.eval()
    model.to(device)

    user_results = {}
    all_user_mean_accs = []

    for pid in val_pids:
        if pid not in tensor_dict:
            print(f"[evaluate_all_val_users] Warning: PID {pid} not in tensor_dict, skipping.")
            continue

        # Filter gesture classes to those actually present for this user
        available_classes = [
            c for c in all_classes if c in tensor_dict[pid]
        ]
        if len(available_classes) < n_way:
            print(f"[evaluate_all_val_users] Warning: PID {pid} has only "
                  f"{len(available_classes)} gesture classes < n_way={n_way}, skipping.")
            continue

        episode_accs = []
        for ep_idx in range(n_episodes):
            episode = _sample_episode(
                tensor_dict, pid, n_way, k_shot, q_query,
                available_classes, support_reps, query_reps,
                device, use_imu,
            )

            if mode == 'zero_shot':
                # Pass sampled_classes so zeroshot_eval can restrict logits to the n_way subset
                ep_config = {**config, 'sampled_classes': episode['sampled_classes']}
                metrics = zeroshot_eval_user(
                    model, ep_config,
                    query_emg    = episode['query_emg'],
                    query_imu    = episode['query_imu'],
                    query_labels = episode['query_labels'],
                )
            else:
                metrics = finetune_and_eval_user(
                    model, config,
                    support_emg    = episode['support_emg'],
                    support_imu    = episode['support_imu'],
                    support_labels = episode['support_labels'],
                    query_emg      = episode['query_emg'],
                    query_imu      = episode['query_imu'],
                    query_labels   = episode['query_labels'],
                    mode           = mode,
                )

            episode_accs.append(metrics['acc'])

        mean_acc = float(np.mean(episode_accs))
        user_results[pid] = {
            'mean_acc': mean_acc,
            'all_accs': episode_accs,
        }
        all_user_mean_accs.append(mean_acc)
        print(f"  PID {pid} | mode={mode} | "
              f"mean acc={mean_acc*100:.2f}% over {n_episodes} episodes")

    overall_mean = float(np.mean(all_user_mean_accs)) if all_user_mean_accs else float('nan')
    overall_std  = float(np.std(all_user_mean_accs))  if all_user_mean_accs else float('nan')

    user_results['__overall__'] = {
        'mean_acc': overall_mean,
        'std_acc':  overall_std,
    }

    print(f"\n[evaluate_all_val_users] mode={mode} | "
          f"Overall: {overall_mean*100:.2f}% ± {overall_std*100:.2f}% "
          f"over {len(all_user_mean_accs)} users")

    return user_results
