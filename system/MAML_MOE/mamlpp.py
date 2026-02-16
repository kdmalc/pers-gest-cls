# === MAML++ core for your codebase ============================================
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import time

from system.MAML_MOE.MOE_training import _model_forward_router, set_MOE_optimizer, SmoothedEarlyStopping

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
        return self._lrs[name.replace('.', '-')][step]

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
    """Runs the MAML++ inner loop, calculating MSL and derivatives."""
    device = config['device']
    multimodal = config['multimodal']
    
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
        outputs_q, labels_q, _ = _model_forward_router(f_q, query_batch, device, multimodal=multimodal)
        logits_q = outputs_q[0] if isinstance(outputs_q, tuple) else outputs_q
        
        q_loss_step = criterion(logits_q, labels_q)
        per_step_query_losses.append(q_loss_step)
        last_q_logits, last_q_labels = logits_q, labels_q

        if step == inner_steps:
            break  # We only do `inner_steps` updates, but we need `inner_steps + 1` query evaluations

        # 2. Support Forward & Inner Update
        f_s = FunctionalModel(model, params_i)
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
        # Weight array size must match the number of query evaluations (inner_steps + 1)
        w = repo_msl_weights(inner_steps + 1, epoch, msl_num_epochs, device=device)
        meta_loss = torch.sum(torch.stack(per_step_query_losses) * w)
    else:
        meta_loss = per_step_query_losses[-1]

    return params_i, last_q_logits, last_q_labels, meta_loss

# -----------------------------
# One epoch of MAML++ training
# -----------------------------
def train_MAMLpp_one_epoch(model, episodic_loader, meta_opt, config, epoch_idx, criterion=None):
    device = config['device']
    model.to(device).train()

    if criterion is None:
        label_smooth = float(config.get("label_smooth", 0.0))
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth) if label_smooth > 0 else nn.CrossEntropyLoss()

    N = int(config["maml_inner_steps"])
    use_lslr = bool(config["maml_use_lslr"])
    alpha_init = float(config["maml_alpha_init"])
    meta_batchsize = int(config["meta_batchsize"])
    episodes_per_epoch = int(config["episodes_per_epoch_train"])

    # ---------- Ensure LSLR exists ----------
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

    def _normalize_step_item(step_item):
        if isinstance(step_item, dict): return [step_item]
        if isinstance(step_item, (list, tuple)): return list(step_item)
        raise TypeError(f"Episode item must be dict or list of dicts.")

    meta_correct = meta_total = loss_sum = n_episodes = accum_count = 0
    meta_opt.zero_grad(set_to_none=True)

    for step_item in episodic_loader:
        episodes = _normalize_step_item(step_item)

        for episode in episodes:
            if n_episodes % 100 == 0:
                print(f"--- Episode {n_episodes}/{episodes_per_epoch} | Target Meta-Batch Size: {meta_batchsize} ---")

            # 1. Capture current weights
            current_theta0 = named_param_dict(model, require_grad_only=True)

            # 2. Run Inner Loop
            thetaN, q_logits, q_labels, meta_loss_task = inner_loop_mamlpp(
                model, current_theta0, episode["support"], episode["query"], 
                config, lslr=(model._lslr if use_lslr else None), 
                criterion=criterion, epoch=epoch_idx
            )

            # 3. Metrics
            with torch.no_grad():
                preds = q_logits.argmax(dim=1)
                meta_correct += (preds == q_labels).sum().item()
                meta_total += q_labels.numel()
                loss_sum += float(meta_loss_task.item())

            # 4. Backward Pass (Scaled strictly by target meta-batch size for stable learning rates)
            # NOTE: gradients accumulate with backward, we arent just overwriting the old gradients
            # TODO: Should this be divided by 1.0 since there is only 1 task or by meta_batchsize (which is probably 4-64...)
            (meta_loss_task / meta_batchsize).backward()
            
            accum_count += 1
            n_episodes += 1

            # 5. Outer Optimizer Step
            if accum_count == meta_batchsize:
                if n_episodes==meta_batchsize:  # Ie, only print on the first hit
                    print(f"Meta batchsize hit on ep {n_episodes}! Parameters updating!")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                meta_opt.step()
                meta_opt.zero_grad(set_to_none=True)
                accum_count = 0

            if episodes_per_epoch and n_episodes >= episodes_per_epoch:
                break
        if episodes_per_epoch and n_episodes >= episodes_per_epoch:
            break

    # Catch residual gradients if epoch cut off before batch filled
    if accum_count > 0:
        # Scale gradients up to represent a "full" step equivalent, or just step with what we have
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= (meta_batchsize / accum_count)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        meta_opt.step()
        meta_opt.zero_grad(set_to_none=True)

    avg_loss = loss_sum / max(n_episodes, 1)
    avg_acc  = meta_correct / max(meta_total, 1)
    return {"loss": avg_loss, "acc": avg_acc, "episodes": n_episodes}

# -----------------------------
# Outer loop (pretrain)
# -----------------------------
def MAMLpp_pretrain(model, config, episodic_train_loader, episodic_val_loader=None):
    device = config["device"]
    model.to(device)

    meta_opt = set_MOE_optimizer(
        model,
        lr=float(config["learning_rate"]),
        use_weight_decay=float(config.get("weight_decay", 0)) > 0.0,
        weight_decay=float(config.get("weight_decay", 0)),
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

    best_val_acc, best_state = -1.0, None
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []
    num_epochs = int(config["num_epochs"])

    for ep in range(1, num_epochs + 1):
        print(f"MAML Pretraining: Epoch {ep} of {num_epochs}")
        epoch_start_time = time.time()

        # Train
        train_metrics = train_MAMLpp_one_epoch(model, episodic_train_loader, meta_opt, config, epoch_idx=ep)
        train_loss_log.append(train_metrics["loss"])
        train_acc_log.append(train_metrics["acc"])
        print(f"Train completed in {time.time() - epoch_start_time:.2f}s")
        print(f'Train loss/acc: {train_metrics["loss"]:.4f}, {train_metrics["acc"]*100:.2f}%')

        # Val
        if episodic_val_loader is not None:
            val_start_time = time.time()
            val_metrics = meta_evaluate(model, episodic_val_loader, config)
            cur_val_loss, cur_val_acc = val_metrics["loss"], val_metrics["acc"]
            val_loss_log.append(cur_val_loss)
            val_acc_log.append(cur_val_acc)
            print(f"Val completed in {time.time() - val_start_time:.2f}s")
            print(f"Val loss/acc: {cur_val_loss:.4f}, {cur_val_acc*100:.2f}%")

            if cur_val_acc > best_val_acc:
                best_val_acc = cur_val_acc
                best_state = copy.deepcopy(model.state_dict())

            if use_es and early_stopping(cur_val_loss):
                print(f"[EarlyStopping] epoch {ep}: val loss stalled. Stopping.")
                if scheduler: scheduler.step()
                break

        if scheduler: scheduler.step()
        print(f"Epoch completed in {time.time() - epoch_start_time:.2f}s\n")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[MAML++] loaded best model (val acc={best_val_acc:.3f})")

    return model, {
        "train_loss_log": train_loss_log,
        "train_acc_log":  train_acc_log,
        "val_loss_log":   val_loss_log,
        "val_acc_log":    val_acc_log,
        "model": model,
        "best_state": best_state,
        "best_val_acc": best_val_acc,
    }

# -----------------------------
# Test Time Adaptation
# -----------------------------
def mamlpp_adapt(model, config, support_batch, *, use_lslr_at_eval=False):
    """Adaptation during Meta-Evaluation. MUST detach graph between steps to avoid modifying base weights."""
    device = config['device']; model.to(device).eval()
    multimodal = bool(config["multimodal"])
    N_eval = int(config["maml_inner_steps_eval"])

    # Extract base weights and ensure they become decoupled leaf nodes
    theta = named_param_dict(model, require_grad_only=True)
    theta = OrderedDict((n, p.detach().requires_grad_(True)) for n, p in theta.items())

    label_smooth = float(config.get("label_smooth", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth) if label_smooth > 0 else nn.CrossEntropyLoss()
    use_lslr = bool(config["maml_use_lslr"]) and use_lslr_at_eval and hasattr(model, "_lslr")
    alpha_eval = float(config["maml_alpha_init_eval"])

    with torch.enable_grad():
        for step in range(N_eval):
            f_s = FunctionalModel(model, theta)
            outputs_s, labels_s, _ = _model_forward_router(f_s, support_batch, device, multimodal=multimodal)
            logits_s = outputs_s[0] if isinstance(outputs_s, tuple) else outputs_s
            s_loss = criterion(logits_s, labels_s)

            # create_graph=False because we don't backprop to theta0
            grads = torch.autograd.grad(s_loss, list(theta.values()), create_graph=False, allow_unused=True)

            lslr = model._lslr if use_lslr else None
            theta = apply_update_repo_style(theta, grads, lslr=lslr, step=step, fallback_alpha=alpha_eval)

            # Detach after step to reset the temporary computation graph
            theta = OrderedDict((n, p.detach().requires_grad_(True)) for n, p in theta.items())

    return theta

@torch.no_grad()
def mamlpp_predict_with_params(model, adapted_params, batch, config):
    device = config['device']; model.to(device).eval()
    multimodal = bool(config["multimodal"])

    fmodel = FunctionalModel(model, adapted_params)
    outputs, labels, B = _model_forward_router(fmodel, batch, device, multimodal=multimodal)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    preds = torch.argmax(logits, dim=1)
    
    return logits, preds, labels, B

def mamlpp_adapt_and_eval(model, config, support_batch, query_batch):
    theta_prime = mamlpp_adapt(model, config, support_batch)
    logits, preds, labels, B = mamlpp_predict_with_params(model, theta_prime, query_batch, config)

    label_smooth = float(config.get("label_smooth", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth) if label_smooth > 0 else nn.CrossEntropyLoss()
    
    q_loss = criterion(logits, labels).item()
    acc = (preds == labels).sum().item() / max(1, B)

    return {"loss": q_loss, "acc": acc, "adapted_params": theta_prime}

def meta_evaluate(model, episodic_loader, config, criterion=None):
    if criterion is not None:
        raise ValueError("Inputting a criterion is currently not supported! It is not used at all rn")
    device = config["device"]
    model.to(device).eval()

    def _norm(x):
        if isinstance(x, dict): return [x]
        if isinstance(x, (list, tuple)): return list(x)
        raise TypeError()

    total_loss = total_correct = total_count = n_eps = 0

    print("META_EVALUATE START")
    step_counter = 0
    ep_counter = 0
    # I am not totally sure how this is working...
    ## Each dataloader probably only returns 1 episode (1 per user)
    ## But we have multiple users... so... do we have multiple dataloaders, or does this dataloader return more than 1... ...
    for step_item in episodic_loader:
        step_counter += 1
        print(f"Step counted! Now on step: {step_counter}")

        for ep in _norm(step_item): 
            ep_counter += 1
            print(f"Ep counted! Now on ep: {ep_counter}")

            # 1. Run inference
            metrics = mamlpp_adapt_and_eval(model, config, ep["support"], ep["query"])
            
            # 2. Get the specific count for THIS episode
            if isinstance(ep["query"], dict) and "labels" in ep["query"]:
                current_q_count = len(ep["query"]["labels"])
            else:
                #current_q_count = 1 
                print("ep[query] is NOT a dictionary!")
                print(f"type: {type(ep['query'])}")
                print(f"len: {len(ep['query'])}")
                print(f"first ele: {ep['query'][0]}")
                raise ValueError("ep['query'] is NOT a dict for some reason. Check on that")

            # 3. Aggregate correctly
            total_loss += metrics["loss"]
            total_correct += (metrics["acc"] * current_q_count) # Multiply by current episode size
            total_count += current_q_count
            n_eps += 1

    return {
        "loss": total_loss / max(n_eps, 1),
        "acc": (total_correct / max(total_count, 1)) if total_count else 0.0,
        "episodes": n_eps,
    }