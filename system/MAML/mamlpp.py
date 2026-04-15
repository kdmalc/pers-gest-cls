# === MAML++ core ============================================
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from functools import partial

import sys
import os
# This finds the 'system' directory (one level up from 'pretraining') and adds it to the search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MOE.MOE_training import _model_forward_router, set_MOE_optimizer, SmoothedEarlyStopping, _to_device
from MAML.shared_maml import *
from MOE.MOE_encoder import dense_MOE_aux_loss as _dense_MOE_aux_loss
from MOE.MOE_encoder import topk_MOE_aux_loss  as _topk_MOE_aux_loss
from MOE.MOE_analysis import RoutingCollector, RoutingAnalyzer

# -----------------------------
# Functional execution helpers
# -----------------------------
def named_param_dict(module: nn.Module, *, require_grad_only: bool = True) -> OrderedDict:
    """Extracts parameters for the inner loop, explicitly ignoring LSLR variables."""
    params = OrderedDict()
    for name, p in module.named_parameters():
        if require_grad_only and not p.requires_grad:
            continue
        if name.startswith("_lslr.") or "._lslr." in name:
            continue
        params[name] = p
    return params

class FunctionalModel(nn.Module):
    """Wraps a base module so we can forward with arbitrary parameter sets via functional_call."""
    def __init__(self, base: nn.Module, params: OrderedDict):
        super().__init__()
        self.base = base
        self.params = params
    def forward(self, *args, **kwargs):
        return torch.func.functional_call(self.base, self.params, args, kwargs)

# -----------------------------
# LSLR: per-parameter, per-step
# -----------------------------
class PerParamPerStepLSLR(torch.nn.Module):
    """MAML++ LSLR: one step-size (scalar) per parameter *per inner step*."""
    def __init__(self, named_params, inner_steps: int, init_lr: float, learnable: bool, device):
        super().__init__()
        self._lrs = nn.ParameterDict()
        self.inner_steps = inner_steps
        for name, p in named_params:
            if not p.requires_grad:
                continue
            key = name.replace('.', '-')
            vec = torch.ones(inner_steps + 1, device=device) * float(init_lr)
            self._lrs[key] = nn.Parameter(vec, requires_grad=learnable)

    def lr(self, name: str, step: int) -> torch.Tensor:
        # BUG FIX: Clamp step to prevent IndexError if eval_steps > train_steps
        safe_step = min(step, self.inner_steps)
        return self._lrs[name.replace('.', '-')][safe_step]

def apply_update_repo_style(params, grads, lslr, step, fallback_alpha=0.01):
    new = OrderedDict()
    for (name, p), g in zip(params.items(), grads):
        if g is None:
            new[name] = p
            continue
        if lslr is not None:
            try:
                alpha = lslr.lr(name, step)
            except (KeyError, AttributeError):
                alpha = p.new_tensor(float(fallback_alpha))
        else:
            alpha = p.new_tensor(float(fallback_alpha))
        new[name] = p - alpha * g
    return new

# -----------------------------
# MSL weights
# -----------------------------
def repo_msl_weights(num_steps: int, epoch: int, msl_num_epochs: int, device='cpu') -> torch.Tensor:
    """Calculates step weights for Multi-Step Loss (MSL) annealing."""
    w = np.ones(num_steps) * (1.0 / num_steps)
    decay = 1.0 / num_steps / max(1, msl_num_epochs)
    min_non_final = 0.03 / num_steps
    for i in range(num_steps - 1):
        w[i] = max(w[i] - epoch * decay, min_non_final)
    w[-1] = min(w[-1] + epoch * ((num_steps - 1) * decay),
                1.0 - ((num_steps - 1) * min_non_final))
    return torch.tensor(w, device=device)

def _compute_aux_loss(routing_info, config, aux_coeff):
    """
    Compute the MOE auxiliary load-balancing loss from routing_info dict.
    Handles both dense and top-k routing.
    Returns a scalar tensor, or 0.0 if routing info is absent.
    """
    if routing_info is None or aux_coeff <= 0:
        return 0.0
    gate_w = routing_info.get('gate_weights')
    if gate_w is None:
        return 0.0
    top_k = config['top_k']
    num_experts = config['num_experts']
    if top_k is None or top_k == num_experts:   # dense / soft routing
        return _dense_MOE_aux_loss(gate_w, coeff=aux_coeff)
    else:                                        # sparse top-k routing
        gate_w_soft = routing_info.get('gate_weights_soft', gate_w)
        return _topk_MOE_aux_loss(gate_w_soft, gate_w, coeff=aux_coeff)


# -----------------------------
# Inner loop (MAML++)
# -----------------------------
def inner_loop_mamlpp(
    model: nn.Module,
    theta0: OrderedDict,
    support_batch,
    query_batch,
    config,
    lslr: PerParamPerStepLSLR | None,
    *,
    criterion,
    epoch: int,
):
    """Runs the MAML++ inner loop, calculating MSL and derivatives.

    MOE aux-loss placement is controlled by config["apply_MOE_aux_loss_inner_outer"]:
      "inner"  (default) — aux loss is added to the support loss at each inner
                           step so the gate is regularised during fast adaptation.
      "outer"            — aux loss is added to the query loss so the meta-update
                           (outer loop) shapes gate balance; inner steps are free
                           to specialise without the balancing penalty.
      "both"             — aux loss is applied at both levels.

    Tradeoff summary:
      "inner"  keeps experts balanced during per-task adaptation but may fight
               the adaptation signal when the task wants a specialised gate.
      "outer"  lets the gate specialise freely per task and regularises the
               meta-initialisation instead — usually a better default.
      "both"   is the strongest regulariser; useful when collapse is severe.
    """
    device = config['device']
    multimodal = config['multimodal']
    use_MOE    = config['use_MOE']
    aux_coeff  = float(config['MOE_aux_coeff'])
    # "inner"  — aux loss added to support loss during inner-loop adaptation steps
    # "outer"  — aux loss added to query loss so the meta-update shapes gate balance
    # "both"   — aux loss applied at both inner and outer levels
    # default is "outer"
    aux_placement = config['apply_MOE_aux_loss_inner_outer']

    # MSL Logic
    msl_use = (config['use_maml_msl'] == True) or (config['use_maml_msl'] == "hybrid" and epoch <= config['maml_msl_num_epochs'])
    msl_num_epochs = config['maml_msl_num_epochs'] if config['use_maml_msl'] != False else -1

    # Second Order Logic
    use_second_order = (config['maml_opt_order'] == "second") or (config['maml_opt_order'] == "hybrid" and epoch >= config['maml_first_order_to_second_order_epoch'])

    fallback_alpha = config['maml_alpha_init']
    inner_steps = config['maml_inner_steps']

    params_i = theta0
    per_step_query_losses = []
    last_q_logits = last_q_labels = None

    # Step 0 to N
    for step in range(inner_steps + 1):

        # 1. Query Loss at current params (for MSL)
        f_q = FunctionalModel(model, params_i)

        if use_MOE and aux_placement in ('outer', 'both'):
            outputs_q, labels_q, _, routing_info_q = _model_forward_router_MOE(
                f_q, query_batch, device, multimodal=multimodal
            )
            logits_q = outputs_q[0] if isinstance(outputs_q, tuple) else outputs_q
            q_loss_step = criterion(logits_q, labels_q) + _compute_aux_loss(routing_info_q, config, aux_coeff)
        else:
            outputs_q, labels_q, _ = _model_forward_router(f_q, query_batch, device, multimodal=multimodal)
            logits_q = outputs_q[0] if isinstance(outputs_q, tuple) else outputs_q
            q_loss_step = criterion(logits_q, labels_q)

        per_step_query_losses.append(q_loss_step)
        last_q_logits, last_q_labels = logits_q, labels_q

        if step == inner_steps:
            break

        # 2. Support Forward & Inner Update
        f_s = FunctionalModel(model, params_i)

        if use_MOE and aux_placement in ('inner', 'both'):
            outputs_s, labels_s, _, routing_info_s = _model_forward_router_MOE(
                f_s, support_batch, device, multimodal=multimodal
            )
            logits_s = outputs_s[0] if isinstance(outputs_s, tuple) else outputs_s
            s_loss = criterion(logits_s, labels_s) + _compute_aux_loss(routing_info_s, config, aux_coeff)
        else:
            outputs_s, labels_s, _ = _model_forward_router(f_s, support_batch, device, multimodal=multimodal)
            logits_s = outputs_s[0] if isinstance(outputs_s, tuple) else outputs_s
            s_loss = criterion(logits_s, labels_s)

        # Calculate gradients maintaining graph connection back to theta0
        grads = torch.autograd.grad(
            s_loss,
            list(params_i.values()),
            create_graph=use_second_order,
            retain_graph=use_second_order,
            allow_unused=True,
        )

        # Apply LSLR update (maintains graph inherently via math operations)
        params_i = apply_update_repo_style(params_i, grads, lslr, step, fallback_alpha=fallback_alpha)

    # Calculate final Meta-Loss
    if msl_use and (epoch <= msl_num_epochs):
        w = repo_msl_weights(inner_steps + 1, epoch, msl_num_epochs, device=device)
        meta_loss = torch.sum(torch.stack(per_step_query_losses) * w)
    else:
        meta_loss = per_step_query_losses[-1]

    return params_i, last_q_logits, last_q_labels, meta_loss

def _normalize_step_item(step_item):
    if isinstance(step_item, dict): return [step_item]
    if isinstance(step_item, (list, tuple)): return list(step_item)
    raise TypeError(f"Episode item must be dict or list of dicts.")


def _model_forward_router_MOE(fmodel, batch, device, multimodal=True, config=None):
    """
    Like _model_forward_router but requests routing info from MOE models.

    Returns (outputs, labels, B, routing_info).
    routing_info is None if the model doesn't return it.
    """
    labels = batch['labels']
    if labels is None:
        raise KeyError("Batch missing 'labels' key.")
    labels = _to_device(labels.long(), device)

    emg = _to_device(batch["emg"], device)

    imu = batch.get("imu", None)
    if config is not None and config["use_imu"]:
        if imu is None:
            raise ValueError(
                "config has use_imu=True but batch['imu'] is None or missing. "
                "Check your dataloader — IMU data is not being packed into the batch."
            )
    imu = _to_device(imu, device) if (imu is not None and multimodal) else None

    demo = batch.get("demo", None)
    if config is not None and config.get("use_demographics", False):
        if demo is None:
            raise ValueError(
                "config has use_demographics=True but batch['demo'] is None or missing. "
                "Check your dataloader — demographics are not being packed into the batch."
            )
    demo = _to_device(demo, device) if demo is not None else None

    B = emg.size(0)
    out = fmodel(x_emg=emg, x_imu=imu, demographics=demo, return_routing=True)

    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        logits, routing_info = out
    else:
        logits = out[0] if isinstance(out, tuple) else out
        routing_info = None

    return (logits,), labels, B, routing_info

# -----------------------------
# One epoch of MAML++ training: ie batched inner loops followed by one outer loop
# -----------------------------
def train_MAMLpp_one_epoch(model, episodic_loader, meta_opt, config, epoch_idx, criterion=None):
    device = config['device']
    model.to(device).train()

    # --- Setup Logic (Verbatim from your original) ---
    if criterion is None:
        label_smooth = float(config["label_smooth"])
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth) if label_smooth > 0 else nn.CrossEntropyLoss()

    N = int(config["maml_inner_steps"])
    use_lslr = bool(config["maml_use_lslr"])
    alpha_init = float(config["maml_alpha_init"])
    meta_batchsize = int(config["meta_batchsize"])
    episodes_per_epoch = int(config["episodes_per_epoch_train"])
    track_alignment = bool(config.get("track_gradient_alignment", False))

    # LSLR Setup
    temp_theta_for_lslr = named_param_dict(model, require_grad_only=True)
    if use_lslr:
        if not hasattr(model, "_lslr") or getattr(model._lslr, "inner_steps", None) != N:
            model._lslr = PerParamPerStepLSLR(
                temp_theta_for_lslr.items(), inner_steps=N, init_lr=alpha_init,
                learnable=True, device=device
            ).to(device)
            meta_opt.add_param_group({
                "params": list(model._lslr.parameters()),
                "lr": float(config["learning_rate"])
            })

    meta_correct = meta_total = loss_sum = n_episodes = accum_count = 0
    meta_opt.zero_grad(set_to_none=True)
    
    # Storage for diagnostic branch
    task_grads_for_batch = []

    for step_item in episodic_loader:
        episodes = _normalize_step_item(step_item)
        for episode in episodes:
            if n_episodes % 100 == 0:
                print(f"--- Episode {n_episodes}/{episodes_per_epoch} | Target Meta-Batch Size: {meta_batchsize} ---")

            current_theta0 = named_param_dict(model, require_grad_only=True)

            # 1. Inner Loop
            thetaN, q_logits, q_labels, meta_loss_task = inner_loop_mamlpp(
                model, current_theta0, episode["support"], episode["query"], 
                config, lslr=(model._lslr if use_lslr else None), 
                criterion=criterion, epoch=epoch_idx
            )

            # 2. Metrics
            with torch.no_grad():
                preds = q_logits.argmax(dim=1)
                meta_correct += (preds == q_labels).sum().item()
                meta_total += q_labels.numel()
                loss_sum += float(meta_loss_task.item())

            # 3. Backward Pass / Gradient Accumulation
            if track_alignment:
                # --- DIAGNOSTIC BRANCH ---
                # Isolate this task's gradient to measure its specific direction
                ## This completely wipes the gradients from the previous task, resetting them to None rather than tensors filled with zeros (which saves memory).
                ## Mixture of Experts (MOE) Routing: If your model uses an MOE architecture, the router decides which "experts" (sub-networks) process the data.
                ## If a specific expert is not used at all during a particular episode's forward pass, PyTorch's autograd engine will not route any gradients to it. Its .grad remains None.
                ### For gradient alignment, since the experts are zero'd out, they basically don't contribute to alignment (just cotnribute 0s)
                ### Actually I reworked this to ignore all components with gradients=None (when NOT using MOE!)
                ### Or maybe the Nones are actually handled within my function now 
                model.zero_grad(set_to_none=True)
                (meta_loss_task / meta_batchsize).backward()
                
                with torch.no_grad():
                    # MODIFIED: Keep the None values! Do not overwrite with zeros here.
                    # This allows the alignment function to distinguish between "unused" and "zero gradient".
                    this_grad = [p.grad.clone().detach() if p.grad is not None else None 
                                for p in model.parameters() if p.requires_grad]
                    task_grads_for_batch.append(this_grad)
            else:
                # --- ORIGINAL VERBATIM BRANCH ---
                (meta_loss_task / meta_batchsize).backward()

            accum_count += 1
            n_episodes += 1

            # 4. Outer Optimizer Step
            if accum_count == meta_batchsize:
                if track_alignment:
                    with torch.no_grad():
                        # Simply pass the list of lists. The function above handles the rest.
                        alignment = compute_meta_batch_alignment(task_grads_for_batch)
                        
                        if n_episodes % 100 == 0 or n_episodes == meta_batchsize:
                            print(f"Meta-Batch Alignment: {alignment:.4f}")

                        # Re-inject gradients into the model for the optimizer
                        for param_idx, p in enumerate(filter(lambda x: x.requires_grad, model.parameters())):
                            # If the first task's grad is None, the whole batch is None (Non-MOE assumption)
                            if task_grads_for_batch[0][param_idx] is None:
                                p.grad = None
                                continue
                            
                            # Sum the gradients across the batch for this parameter
                            p.grad = torch.stack([task_grad[param_idx] for task_grad in task_grads_for_batch]).sum(dim=0)
                        
                    task_grads_for_batch = [] # Reset

                # Final Step (Shared logic)
                #if n_episodes == meta_batchsize:
                #    print(f"Meta batchsize hit on ep {n_episodes}! Parameters updating!")
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                meta_opt.step()
                meta_opt.zero_grad(set_to_none=True)  # Prepare for the next meta-batch
                accum_count = 0

            if episodes_per_epoch and n_episodes >= episodes_per_epoch:
                break
        if episodes_per_epoch and n_episodes >= episodes_per_epoch:
            break

    # Residual Step logic
    if accum_count > 0:
        if track_alignment and task_grads_for_batch:
            with torch.no_grad():
                for param_idx, p in enumerate(filter(lambda x: x.requires_grad, model.parameters())):
                    # THE FIX: Filter out None gradients here as well
                    valid_grads = [task_grad[param_idx] for task_grad in task_grads_for_batch 
                                if task_grad[param_idx] is not None]
                    
                    if valid_grads:
                        summed_p_grad = torch.stack(valid_grads).sum(dim=0)
                        p.grad = summed_p_grad * (meta_batchsize / accum_count)
                    else:
                        p.grad = None
                        
            task_grads_for_batch = [] 
        else:
            # Standard PyTorch accumulation path: gradients are already in p.grad
            for p in model.parameters():
                if p.grad is not None:
                    p.grad *= (meta_batchsize / accum_count)

        # Finalize the update
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        meta_opt.step()
        meta_opt.zero_grad(set_to_none=True)

    avg_loss = loss_sum / max(n_episodes, 1)
    avg_acc  = meta_correct / max(meta_total, 1)
    return {"loss": avg_loss, "acc": avg_acc, "episodes": n_episodes}

# -----------------------------
# Pretrain: this handles the full training (ie all epochs for inner and outer loops)
## This is just the MAML training stage
# -----------------------------
def mamlpp_pretrain(model, config, episodic_train_loader, episodic_val_loader=None):
    device = config["device"]
    model.to(device)

    use_MOE       = config['use_MOE']
    MOE_log_every = int(config['MOE_log_every'])
    MOE_plot_dir  = config['MOE_plot_dir']
    num_experts   = int(config['num_experts'])
    model_name    = config['model_type']

    meta_opt = set_MOE_optimizer(
        model,
        lr=float(config["learning_rate"]),
        use_weight_decay=float(config["weight_decay"]) > 0.0,
        weight_decay=float(config["weight_decay"]),
        optimizer_name=config["optimizer"],
    )

    scheduler = None
    if bool(config.get("use_cosine_outer_lr", False)):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=int(config["num_epochs"]))

    use_es = episodic_val_loader is not None and bool(config["use_earlystopping"])
    early_stopping = None
    if use_es:
        early_stopping = SmoothedEarlyStopping(
            patience=int(config["earlystopping_patience"]),
            min_delta=float(config["earlystopping_min_delta"]),
        )

    best_val_acc, best_state, best_val_epoch = -1.0, None, 0
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []
    routing_reports = []
    num_epochs = int(config["num_epochs"])

    # Epoch 0 baseline: before ANY MAML training
    if episodic_val_loader is not None:
        print("=== Epoch 0 Baseline (before ANY MAML training) ===")
        val_start = time.time()
        adapt_fn = partial(mamlpp_adapt_and_eval, debug=True)
        val_metrics = meta_evaluate(model, episodic_val_loader, config, adapt_fn)
        print(f"Epoch 0 val loss/acc: {val_metrics['loss']:.4f}, {val_metrics['acc']*100:.2f}%")
        print(f"Epoch 0 val completed in {time.time()-val_start:.2f}s")

    for ep in range(1, num_epochs + 1):
        print(f"MAML++ Pretraining: Epoch {ep} of {num_epochs}")
        epoch_start_time = time.time()

        # Train
        train_metrics = train_MAMLpp_one_epoch(model, episodic_train_loader, meta_opt, config, epoch_idx=ep)
        train_loss_log.append(train_metrics["loss"])
        train_acc_log.append(train_metrics["acc"])
        print(f"Train completed in {time.time() - epoch_start_time:.2f}s")
        print(f'Train loss/acc: {train_metrics["loss"]:.4f}, {train_metrics["acc"]*100:.2f}%')

        if bool(config.get("maml_use_lslr")) and hasattr(model, "_lslr"):
            with torch.no_grad():
                all_lrs = torch.cat([v.flatten() for v in model._lslr._lrs.values()])
                mean_lr = all_lrs.mean().item()
                min_lr  = all_lrs.min().item()
                max_lr  = all_lrs.max().item()
                print(f"[LSLR Stats] Mean: {mean_lr:.7f} | Min: {min_lr:.7f} | Max: {max_lr:.7f}")

        # Val
        if episodic_val_loader is not None:
            val_start_time = time.time()
            if ep < 2:  # This only triggers on the first epoch (we start with epoch 1 FYI)
                adapt_fn = partial(mamlpp_adapt_and_eval, debug=True)
                val_metrics = meta_evaluate(model, episodic_val_loader, config, adapt_fn)
            else:
                val_metrics = meta_evaluate(model, episodic_val_loader, config, mamlpp_adapt_and_eval)
            cur_val_loss, cur_val_acc = val_metrics["loss"], val_metrics["acc"]
            val_loss_log.append(cur_val_loss)
            val_acc_log.append(cur_val_acc)
            print(f"Val completed in {time.time() - val_start_time:.2f}s")
            print(f"Val loss/acc: {cur_val_loss:.4f}, {cur_val_acc*100:.2f}%")

            if cur_val_acc > best_val_acc:
                best_val_acc  = cur_val_acc
                best_state    = copy.deepcopy(model.state_dict())
                best_val_epoch = ep

            if use_es and early_stopping(cur_val_loss):
                print(f"[EarlyStopping] epoch {ep}: val loss stalled. Stopping.")
                if scheduler: scheduler.step()
                break
        else:
            print("No val loader found! Skipping during-training val evals")

        # ── MOE routing analysis ─────────────────────────────────────────────
        if use_MOE and MOE_log_every > 0 and ep % MOE_log_every == 0 and episodic_val_loader is not None:
            print(f"\n[MOE] Routing analysis at epoch {ep}...")
            report = _maml_routing_analysis_epoch(
                model, episodic_val_loader, config,
                num_experts=num_experts, model_name=model_name,
                epoch=ep, plot_dir=MOE_plot_dir,
            )
            if report:
                routing_reports.append(report)

        if scheduler: scheduler.step()
        print(f"Epoch completed in {time.time() - epoch_start_time:.2f}s\n")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[MAML++] loaded best model (val acc={best_val_acc:.3f})")

    # NOTE: These do not match pretrain() in pretrain_trainer...
    return model, {
        "train_loss_log":  train_loss_log,
        "train_acc_log":   train_acc_log,
        "val_loss_log":    val_loss_log,
        "val_acc_log":     val_acc_log,
        "model":           model,
        "best_state":      best_state,
        "best_val_acc":    best_val_acc,
        "best_val_epoch":  best_val_epoch,
        "routing_reports": routing_reports,
    }


def _maml_routing_analysis_epoch(model, episodic_val_loader, config,
                                  num_experts, model_name, epoch, plot_dir):
    """
    Run one pass of the val episodic loader collecting routing info,
    then print and optionally plot a routing analysis report.

    We run with adapted parameters (after inner-loop steps on support) so
    the routing reflects what the model actually does at query time.

    Returns the report dict (or None on failure).
    """

    device     = config['device']
    multimodal = bool(config.get('multimodal', True))
    # TODO: what the heck is this, demo_dim_labels definitely does not exist in the config...
    demo_labels = config.get('demo_dim_labels', None)

    collector = RoutingCollector(num_experts=num_experts, model_name=f"{model_name}_maml_val")

    try:
        model.eval()
        with torch.no_grad():
            for step_item in episodic_val_loader:
                episodes = _normalize_step_item(step_item)
                for episode in episodes:
                    # Use adapted params from a quick inner-loop run
                    support_batch = episode['support']
                    query_batch   = episode['query']

                    # --- PID extraction ---
                    # user_id lives at the episode level (set by maml_mm_collate),
                    # NOT inside query_batch. Replicate it once per query sample.
                    user_id = episode['user_id']
                    if user_id is None:
                        # TODO: We should remove this... we should ONLY be using one of these...
                        # Flat (non-episodic) loader fallback: check both common key names
                        user_id = episode.get('pid', episode.get('pids', 'unknown'))
                    # Normalise to a flat list of strings, one entry per query sample
                    if isinstance(user_id, (list, tuple)):
                        # Already a per-sample list (flat loader); use as-is
                        episode_pids = [str(p) for p in user_id]
                    else:
                        # Single string (episodic loader) — replicate after we know B
                        episode_pids = str(user_id)   # resolved to list below

                    # Fast inner-loop adapt (no grad needed for analysis)
                    theta_prime = mamlpp_adapt(model, config, support_batch,
                                               use_lslr_at_eval=False)
                    f_q = FunctionalModel(model, theta_prime)

                    # Forward with routing
                    qemg    = query_batch['emg'].to(device)
                    qlabels = query_batch['labels'].to(device)
                    qimu    = query_batch.get('imu')
                    if qimu is not None and multimodal:
                        qimu = qimu.to(device)
                    else:
                        qimu = None

                    # Resolve scalar PID → per-sample list now that we know batch size
                    if isinstance(episode_pids, str):
                        episode_pids = [episode_pids] * qemg.size(0)

                    out = f_q(qemg, qimu, return_routing=True)
                    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                        _, routing_info = out
                        gate_w = routing_info.get('gate_weights')
                        if gate_w is not None:
                            demo = query_batch.get('demographics')
                            # Use global_labels (dataset-level gesture IDs, 0…N_classes-1)
                            # rather than local episode labels (0…n_way-1), which are
                            # arbitrary and shuffled per episode by meta-augmentation.
                            # Falls back to local labels if global_labels absent.
                            gesture_ids = query_batch.get('global_labels', qlabels).cpu()
                            collector.add(
                                gate_weights   = gate_w.cpu(),
                                gesture_labels = gesture_ids,
                                pids           = episode_pids,
                                demographics   = demo.cpu() if demo is not None else None,
                            )
    except Exception as e:
        print(f"[MOE routing analysis] Warning during MAML analysis: {e}")
        model.train()
        return None
    finally:
        model.train()

    try:
        record   = collector.finalize()
        analyzer = RoutingAnalyzer(record)
        report   = analyzer.full_report(print_report=True, demo_dim_labels=demo_labels)
        report['epoch'] = epoch

        if plot_dir is not None:
            import os
            epoch_dir = os.path.join(str(plot_dir), f"maml_epoch_{epoch:03d}")
            analyzer.plot_all(save_dir=epoch_dir)

        return report
    except Exception as e:
        print(f"[MOE routing analysis] Warning during analysis: {e}")
        return None

# -----------------------------
# Test Time Adaptation
# -----------------------------
def mamlpp_adapt(model, config, support_batch, *, use_lslr_at_eval=False):
    """
    Adaptation during Meta-Evaluation. 
    CRITICAL: Must use model.train() to allow RNN gradient computation (cuDNN requirement).
    """
    device = config['device']
    
    # FIX: Explicitly set to train mode for the inner loop updates
    model.to(device).train() 
    
    multimodal = bool(config["multimodal"])
    N_eval = int(config["maml_inner_steps_eval"])

    # Extract base weights and ensure they become decoupled leaf nodes
    theta = named_param_dict(model, require_grad_only=True)
    theta = OrderedDict((n, p.detach().requires_grad_(True)) for n, p in theta.items())

    label_smooth = float(config["label_smooth"])
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth) if label_smooth > 0 else nn.CrossEntropyLoss()
    
    # LSLR Logic
    use_lslr = bool(config["maml_use_lslr"]) and use_lslr_at_eval and hasattr(model, "_lslr")
    alpha_eval = float(config["maml_alpha_init_eval"])

    with torch.enable_grad():
        for step in range(N_eval):
            f_s = FunctionalModel(model, theta)
            outputs_s, labels_s, _ = _model_forward_router(f_s, support_batch, device, multimodal=multimodal)
            logits_s = outputs_s[0] if isinstance(outputs_s, tuple) else outputs_s
            s_loss = criterion(logits_s, labels_s)

            # create_graph=False because we don't backprop to theta0 during EVALUATION
            grads = torch.autograd.grad(s_loss, list(theta.values()), create_graph=False, allow_unused=True)
            lslr = model._lslr if use_lslr else None
            theta = apply_update_repo_style(theta, grads, lslr=lslr, step=step, fallback_alpha=alpha_eval)
            # Detach after step to reset the temporary computation graph
            theta = OrderedDict((n, p.detach().requires_grad_(True)) for n, p in theta.items())

    return theta

@torch.no_grad()
def mamlpp_predict_with_params(model, adapted_params, batch, config):
    """
    Prediction on Query Set.
    We switch to eval() here to disable Dropout/BN stats updates for the final prediction.
    """
    device = config['device']
    
    # Switch to eval for deterministic prediction
    # WARNING: If using BatchNorm, this uses global stats, not query stats!
    model.to(device).eval() 
    
    multimodal = bool(config["multimodal"])

    fmodel = FunctionalModel(model, adapted_params)
    outputs, labels, B = _model_forward_router(fmodel, batch, device, multimodal=multimodal)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    preds = torch.argmax(logits, dim=1)
    
    # Revert base model to train immediately to prevent silent state bugs in next episode
    model.train()
    
    return logits, preds, labels, B

def mamlpp_adapt_and_eval(model, config, support_batch, query_batch, debug=False):
    pre_adapt_acc = None

    if debug:
        device = next(model.parameters()).device
        multimodal = config["use_imu"]
        q_emg    = query_batch["emg"].to(device)
        q_labels = query_batch["labels"].to(device)
        s_emg    = support_batch["emg"].to(device)

        diffs = (q_emg.float().unsqueeze(1) - s_emg.float().unsqueeze(0)).abs().amax(dim=(-2, -1))
        leaking_pairs = (diffs < 1e-5).nonzero(as_tuple=False)
        assert leaking_pairs.numel() == 0, \
            f"Support/query leakage! Pairs: {leaking_pairs.tolist()}"

        model.eval()
        with torch.no_grad():
            fmodel = FunctionalModel(model, named_param_dict(model))
            outputs, labels, B = _model_forward_router(
                fmodel, query_batch, device, multimodal=multimodal, config=config
            )
            pre_adapt_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            pre_adapt_acc = (pre_adapt_logits.argmax(dim=-1) == q_labels).float().mean().item()
        model.train()

    # Adapt
    theta_prime = mamlpp_adapt(model, config, support_batch)
    # Predict
    logits, preds, labels, B = mamlpp_predict_with_params(model, theta_prime, query_batch, config)

    label_smooth = float(config["label_smooth"])
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth) if label_smooth > 0 else nn.CrossEntropyLoss()
    q_loss = criterion(logits, labels).item()
    acc    = (preds == labels).sum().item() / B

    return {"loss": q_loss, "acc": acc, "pre_adapt_acc": pre_adapt_acc}

def compute_meta_batch_alignment(task_gradients_list):
    """
    task_gradients_list: List of lists. 
                         Each inner list is [grad_p1, grad_p2, ... grad_pn]
    """
    num_tasks = len(task_gradients_list)
    if num_tasks < 2:
        return 0.0
    # 1. Identify "Active" parameters
    # We look at the first task to see which parameters actually have gradients.
    # If a parameter is None here, and you aren't using MOE, it's a zombie.
    active_indices = [
        i for i, g in enumerate(task_gradients_list[0]) if g is not None
    ]
    if not active_indices:
        return 0.0
    # 2. Flatten only the active parameters for each task
    flattened_tasks = []
    for task_grads in task_gradients_list:
        # Concatenate all active tensors into one long 1D vector
        active_tensors = [task_grads[i].flatten() for i in active_indices]
        flattened_tasks.append(torch.cat(active_tensors))
    # 3. Stack into a matrix [NumTasks, TotalDimentionality]
    # Now that they are all Tensors of the same length, stack works perfectly.
    grads_matrix = torch.stack(flattened_tasks) 
    # 4. Standard Cosine Similarity Math
    grads_norm = torch.nn.functional.normalize(grads_matrix, p=2, dim=1)
    sim_matrix = torch.mm(grads_norm, grads_norm.t())
    # Extract values above the diagonal
    triu_indices = torch.triu_indices(num_tasks, num_tasks, offset=1)
    similarities = sim_matrix[triu_indices[0], triu_indices[1]]
    return similarities.mean().item()