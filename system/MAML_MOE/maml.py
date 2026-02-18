# === MAML core (Original) =====================================================
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import time

# Reusing your existing project-specific routers and logic
from system.MAML_MOE.MOE_training import _model_forward_router, set_MOE_optimizer, SmoothedEarlyStopping

# -----------------------------
# Functional execution helpers
# -----------------------------
def named_param_dict(module: nn.Module) -> OrderedDict:
    """Extracts parameters that require gradients."""
    return OrderedDict((name, p) for name, p in module.named_parameters() if p.requires_grad)

class FunctionalModel(nn.Module):
    """Wraps a base module for functional forward passes."""
    def __init__(self, base: nn.Module, params: OrderedDict):
        super().__init__()
        self.base = base
        self.params = params
    def forward(self, *args, **kwargs):
        return torch.func.functional_call(self.base, self.params, args, kwargs)

def apply_gradient_update(params, grads, alpha):
    """Simple SGD update: theta' = theta - alpha * grad"""
    new_params = OrderedDict()
    for (name, p), g in zip(params.items(), grads):
        if g is None:
            new_params[name] = p
        else:
            new_params[name] = p - alpha * g
    return new_params

# -----------------------------
# Inner loop (Original MAML)
# -----------------------------
def inner_loop_maml(
    model: nn.Module,
    theta0: OrderedDict,
    support_batch,
    config,
    *,
    criterion
):
    """
    Performs N steps of gradient descent on the support set.
    Returns the adapted parameters (theta_prime).
    """
    device = config['device']
    multimodal = config.get('multimodal', False)
    inner_steps = int(config['maml_inner_steps'])
    alpha = float(config['maml_alpha_init'])  # NOTE: This is our MAML learning rate! Need to unify this with learning_rate ...

    # We can still have first or second order MAML --> This is not MAML++ specific!
    use_second_order = (config['maml_opt_order'] == "second")

    params_i = theta0

    for step in range(inner_steps):
        # Forward pass on support set
        f_s = FunctionalModel(model, params_i)
        outputs_s, labels_s, _ = _model_forward_router(f_s, support_batch, device, multimodal=multimodal)
        logits_s = outputs_s[0] if isinstance(outputs_s, tuple) else outputs_s
        s_loss = criterion(logits_s, labels_s)
        
        # Calculate gradients
        # create_graph=True allows us to take gradients of gradients (Meta-Optimization)
        grads = torch.autograd.grad(
            s_loss,
            list(params_i.values()),
            create_graph=use_second_order,
            retain_graph=use_second_order,
            allow_unused=True,
        )
        
        # Manual SGD Update
        params_i = apply_gradient_update(params_i, grads, alpha)

    return params_i

# -----------------------------
# Training logic
# -----------------------------
def train_MAML_one_epoch(model, episodic_loader, meta_opt, config, epoch_idx, criterion=None):
    device = config['device']
    model.train()

    if criterion is None:
        label_smooth = float(config.get("label_smooth", 0.0))
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    meta_batchsize = int(config["meta_batchsize"])
    episodes_per_epoch = int(config.get("episodes_per_epoch_train", 0))
    multimodal = config.get('multimodal', False)

    meta_correct = meta_total = loss_sum = n_episodes = accum_count = 0
    meta_opt.zero_grad(set_to_none=True)

    for step_item in episodic_loader:
        # Standardize input format
        episodes = [step_item] if isinstance(step_item, dict) else step_item

        for episode in episodes:
            # 1. Adapt on Support Set
            theta0 = named_param_dict(model)
            theta_prime = inner_loop_maml(model, theta0, episode["support"], config, criterion=criterion)

            # 2. Evaluate on Query Set (Meta-Loss)
            f_q = FunctionalModel(model, theta_prime)
            outputs_q, labels_q, _ = _model_forward_router(f_q, episode["query"], device, multimodal=multimodal)
            logits_q = outputs_q[0] if isinstance(outputs_q, tuple) else outputs_q
            meta_loss_task = criterion(logits_q, labels_q)

            # 3. Metrics (No grad for logging)
            with torch.no_grad():
                preds = logits_q.argmax(dim=1)
                meta_correct += (preds == labels_q).sum().item()
                meta_total += labels_q.numel()
                loss_sum += float(meta_loss_task.item())

            # 4. Accumulate Gradients
            (meta_loss_task / meta_batchsize).backward()
            accum_count += 1
            n_episodes += 1

            # 5. Outer Optimizer Step
            if accum_count == meta_batchsize:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                meta_opt.step()
                meta_opt.zero_grad(set_to_none=True)
                accum_count = 0

            if episodes_per_epoch and n_episodes >= episodes_per_epoch:
                break
        if episodes_per_epoch and n_episodes >= episodes_per_epoch:
            break

    # Final step for remaining episodes
    if accum_count > 0:
        meta_opt.step()
        meta_opt.zero_grad(set_to_none=True)

    return {"loss": loss_sum / max(n_episodes, 1), "acc": meta_correct / max(meta_total, 1)}

# -----------------------------
# Top-Level Pretraining Pipeline
# -----------------------------
def MAML_pretrain(model, config, episodic_train_loader, episodic_val_loader=None):
    device = config["device"]
    model.to(device)

    # Re-using your specific MOE optimizer setter
    meta_opt = set_MOE_optimizer(
        model,
        lr=float(config["learning_rate"]),
        use_weight_decay=float(config.get("weight_decay", 0)) > 0.0,
        weight_decay=float(config.get("weight_decay", 0)),
        optimizer_name=config["optimizer"],
    )

    use_es = episodic_val_loader is not None and bool(config.get("use_earlystopping", False))
    early_stopping = SmoothedEarlyStopping(
        patience=int(config.get("earlystopping_patience", 5)),
        min_delta=float(config.get("earlystopping_min_delta", 0.001)),
    ) if use_es else None

    best_val_acc, best_state = -1.0, None
    metrics_log = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for ep in range(1, int(config["num_epochs"]) + 1):
        print(f"--- MAML Epoch {ep} ---")
        
        # Train
        t_metrics = train_MAML_one_epoch(model, episodic_train_loader, meta_opt, config, ep)
        metrics_log["train_loss"].append(t_metrics["loss"])
        metrics_log["train_acc"].append(t_metrics["acc"])
        print(f"Train Loss: {t_metrics['loss']:.4f} | Acc: {t_metrics['acc']*100:.2f}%")

        # Validation
        if episodic_val_loader:
            v_metrics = meta_evaluate(model, episodic_val_loader, config)
            metrics_log["val_loss"].append(v_metrics["loss"])
            metrics_log["val_acc"].append(v_metrics["acc"])
            print(f"Val Loss: {v_metrics['loss']:.4f} | Acc: {v_metrics['acc']*100:.2f}%")

            if v_metrics["acc"] > best_val_acc:
                best_val_acc = v_metrics["acc"]
                best_state = copy.deepcopy(model.state_dict())

            if early_stopping and early_stopping(v_metrics["loss"]):
                print("Early stopping triggered.")
                break

    if best_state:
        model.load_state_dict(best_state)
    
    return model, metrics_log

# -----------------------------
# Meta-Evaluation
# -----------------------------
@torch.no_grad()
def maml_predict(model, adapted_params, batch, config):
    """Simple prediction using adapted parameters."""
    device = config['device']
    fmodel = FunctionalModel(model, adapted_params)
    outputs, labels, B = _model_forward_router(fmodel, batch, device, multimodal=config.get('multimodal', False))
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    return logits, torch.argmax(logits, dim=1), labels, B

def meta_evaluate(model, episodic_loader, config):
    model.eval()
    device = config["device"]
    criterion = nn.CrossEntropyLoss()
    
    total_loss = total_correct = total_count = n_eps = 0

    for step_item in episodic_loader:
        episodes = [step_item] if isinstance(step_item, dict) else step_item
        for ep in episodes:
            # 1. Adapt (Inner Loop - with torch.enable_grad() so we can compute support grads)
            with torch.enable_grad():
                theta0 = named_param_dict(model)
                # We usually use more steps or a different LR at eval; check config
                # It is overwriting so that we can call inner_loop_maml and have it use the eval params without needing additional branching logic
                eval_config = config.copy()
                eval_config['maml_inner_steps'] = config.get('maml_inner_steps_eval', config['maml_inner_steps'])
                eval_config['maml_alpha_init'] = config.get('maml_alpha_init_eval', config['maml_alpha_init'])
                
                theta_prime = inner_loop_maml(model, theta0, ep["support"], eval_config, criterion=criterion)

            # 2. Predict on Query
            logits, preds, labels, B = maml_predict(model, theta_prime, ep["query"], config)
            
            total_loss += criterion(logits, labels).item()
            total_correct += (preds == labels).sum().item()
            total_count += B
            n_eps += 1

    return {
        "loss": total_loss / max(n_eps, 1),
        "acc": total_correct / max(total_count, 1),
        "episodes": n_eps
    }