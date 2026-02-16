# === MAML++ core for your codebase ============================================
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import time

#from MOE_training import _model_forward_router, set_MOE_optimizer, SmoothedEarlyStopping
from system.MAML_MOE.MOE_training import _model_forward_router, set_MOE_optimizer, SmoothedEarlyStopping

# -----------------------------
# Functional execution helpers
# -----------------------------
def named_param_dict(module: nn.Module, *, require_grad_only: bool = True,
                     exclude_bn_from_inner: bool = True) -> OrderedDict:
    params = OrderedDict()
    for name, p in module.named_parameters():
        if require_grad_only and not p.requires_grad:
            continue
        # Exclude LSLR (and any inner-optim modules) from the inner set
        if name.startswith("_lslr.") or "._lslr." in name:
            continue
        # TODO: So does this cover BN AND Layernorm?? ...
        if exclude_bn_from_inner and (
            'bn' in name.lower() or 'batchnorm' in name.lower() or 'norm_layer' in name.lower()
        ):
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
    """
    MAML++ LSLR: one step-size (scalar) per parameter *per inner step*.
    Unconstrained -> sign encodes direction. Learnable if configured.
    """
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

    def parameters(self):
        # allow optimizer to see these params when added as a param group
        return super().parameters()

def apply_update_repo_style(params, grads, lslr, step, fallback_alpha=None):
    new = OrderedDict()
    for (name, p), g in zip(params.items(), grads):
        if g is None:
            new[name] = p
            continue
        if lslr is not None:
            try:
                alpha = lslr.lr(name, step)
            except (KeyError, AttributeError):
                print("Key Error in apply_update_repo_style excepted")
                alpha = p.new_tensor(float(fallback_alpha if fallback_alpha is not None else 0.4))
        else:
            alpha = p.new_tensor(float(fallback_alpha if fallback_alpha is not None else 0.4))
        new[name] = p - alpha * g
    return new


# -----------------------------
# BN per-step (optional hooks)
# -----------------------------
#def set_bn_step(mod: nn.Module, step: int):
#    """If you have BN-per-step wrappers, expose a `.set_step(int)` and we’ll call it here."""
#    for m in mod.modules():
#        if hasattr(m, 'set_step') and callable(getattr(m, 'set_step')):
#            m.set_step(step)
#
#def backup_bn_stats(mod: nn.Module):
#    """Optional: if your BN wrapper exposes `.backup()`."""
#    for m in mod.modules():
#        if hasattr(m, 'backup') and callable(getattr(m, 'backup')):
#            m.backup()
#
#def restore_bn_stats(mod: nn.Module):
#    """Optional: if your BN wrapper exposes `.restore()`."""
#    for m in mod.modules():
#        if hasattr(m, 'restore') and callable(getattr(m, 'restore')):
#            m.restore()

# -----------------------------
# MSL weights (repo-accurate)
# -----------------------------
def repo_msl_weights(num_steps: int, epoch: int, msl_num_epochs: int, device='cpu') -> torch.Tensor:
    """
    Matches the provided repo’s get_per_step_loss_importance_vector():
    - non-final steps decay toward a small floor
    - final step ramps up
    - early epochs emphasize uniform, later ones emphasize final step
    """
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
    """
    Implements: DOA (via use_second_order), LSLR, BN per-step hooks, MSL (repo schedule).
    Returns:
      thetaN, final_query_logits, final_query_labels, meta_loss (MSL or final-only)
    """
    # NOTE: MAML++ evaluates query loss after each step (0..N-1), and either uses MSL
    # early in training or only the final-step loss later.
    # We also evaluate “step 0” (before any adaptation) as in their code path.

    device = config['device']
    multimodal = config['multimodal']
    msl_use = (config['use_maml_msl']==True) or (config['use_maml_msl']=="hybrid" and epoch >= config['maml_msl_num_epochs']) 
    if msl_use==True:
        msl_num_epochs = config['maml_msl_num_epochs']
    else:
        msl_num_epochs = -1  # MSL is used for the FIRST N epochs. So by setting to -1 in the False case it will not be used

    use_second_order = (config['maml_opt_order']=="second") or (config['maml_opt_order']=="hybrid" and epoch >= config['maml_first_order_to_second_order_epoch']) 

    fallback_alpha = config['maml_alpha_init']
    inner_steps = config['maml_inner_steps']

    # MSL weights (used only if msl_use=True and epoch < msl_num_epochs)
    w = repo_msl_weights(inner_steps, epoch, msl_num_epochs, device=device)

    params_i = theta0
    per_step_query_losses = []
    last_q_logits = last_q_labels = None

    for step in range(inner_steps + 1):
        # Here, we do a forward pass on Query, then a forward pass on Support, then update
        ## Doing this double forward pass is correct for MAML++ but is computationally expensive...

        # ---- Query loss at current params_i (for MSL) ----
        # THIS IS THE INITIAL CHECK: how does the model perform on the query BEFORE we have seen/trained on the support set?
        #set_bn_step(model, min(step, inner_steps))
        f_q = FunctionalModel(model, params_i)  # This makes the temporary copy of the model, I think
        outputs_q, labels_q, _ = _model_forward_router(f_q, query_batch, device, multimodal=multimodal)
        logits_q = outputs_q[0] if isinstance(outputs_q, tuple) else outputs_q
        q_loss_step = criterion(logits_q, labels_q)
        per_step_query_losses.append(q_loss_step)
        last_q_logits, last_q_labels = logits_q, labels_q

        if step == inner_steps:
            break

        # ---- Support loss and inner update to get params_{i+1} ----
        # SUPPORT EVALUATION + calculate loss and gradients
        #set_bn_step(model, step)
        f_s = FunctionalModel(model, params_i)
        outputs_s, labels_s, _ = _model_forward_router(f_s, support_batch, device, multimodal=multimodal)
        logits_s = outputs_s[0] if isinstance(outputs_s, tuple) else outputs_s
        s_loss = criterion(logits_s, labels_s)
        # compute grads wrt params_i
        grads = torch.autograd.grad(
            s_loss,
            list(params_i.values()),
            create_graph=use_second_order,
            retain_graph=use_second_order,
            allow_unused=True,  # match repo tolerance
        )
        # apply LSLR update
        params_i = apply_update_repo_style(params_i, grads, lslr, step, fallback_alpha=fallback_alpha)
        # TODO: May not need this? If use_second_order is False then that is already taken care of in the above autograd call?...
        #if not use_second_order:
        #    # Break graph between inner steps (FOMAML)
        #    params_i = OrderedDict((n, p.detach().requires_grad_(True)) for n, p in params_i.items())

    # ---- MSL aggregation (only early epochs), else final-only loss ----
    if msl_use and (epoch < msl_num_epochs):
        # use weights w[0..N-1]; they weight each step’s query loss as they train
        # We accumulated inner_steps+1 entries (0..N). The repo uses per-step list over steps (uses last too).
        # Match repo behavior: they weight losses per step during training; after msl window, use only final step.
        meta_loss = torch.sum(torch.stack(per_step_query_losses[:inner_steps]) * w) \
                    + per_step_query_losses[inner_steps] * w[-1]
    else:
        meta_loss = per_step_query_losses[-1]

    return params_i, last_q_logits, last_q_labels, meta_loss

# -----------------------------
# One epoch of MAML++ training
# -----------------------------
def train_MAMLpp_one_epoch(model, episodic_loader, meta_opt, config, epoch_idx, criterion=None):
    """
    Train one epoch of MAML(++) with either:
      • Single-episode updates (meta_batchsize = 1), or
      • Meta-batched updates averaging over M episodes per optimizer step
        (meta_batchsize = M > 1).
    """
    device = config['device']
    model.to(device).train()
    multimodal = bool(config["multimodal"])

    # Loss
    if criterion is None:
        label_smooth = float(config["label_smooth"])
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # Inner loop settings
    N = int(config["maml_inner_steps"])
    use_second_order = (config["maml_opt_order"]=="second") or (config["maml_opt_order"]=="hybrid" and epoch_idx > int(config["maml_first_order_to_second_order_epoch"]))
    exclude_bn_from_inner = not bool(config["enable_inner_loop_optimizable_bn_params"])

    # Per-parameter per-step learning rates (LSLR)
    use_lslr   = bool(config["maml_use_lslr"])
    alpha_init = float(config["maml_alpha_init"])

    # Episode scheduling
    meta_batchsize = int(config["meta_batchsize"])  # meta-batch size
    episodes_per_epoch = int(config["episodes_per_epoch_train"])  # 0 → no explicit cap

    # Metrics
    meta_correct = 0
    meta_total   = 0
    loss_sum     = 0.0
    n_episodes   = 0

    # ---------- NEW: Freeze inner-loop param set once per epoch ----------
    # NOTE: This could be causing a bug? I am moving this inside of the episodes so it freezes once per episode instead of once per epoch...
    ## Important: named_param_dict must already exclude _lslr.* (you added that).
    #theta0_full_epoch = named_param_dict(
    #    model, require_grad_only=True,
    #    exclude_bn_from_inner=exclude_bn_from_inner
    #)
    ## Optional safety: cache and assert stability of keys across epochs
    #if not hasattr(model, "_inner_keys_frozen"):
    #    model._inner_keys = tuple(theta0_full_epoch.keys())
    #    model._inner_keys_frozen = True
    #else:
    #    # If something changed, rebuild from frozen keys to guarantee stability
    #    if set(theta0_full_epoch.keys()) != set(model._inner_keys):
    #        # Re-map from frozen names to current parameter objects
    #        name_to_param = dict(model.named_parameters())
    #        theta0_full_epoch = OrderedDict((n, name_to_param[n]) for n in model._inner_keys)

    # ---------- NEW: Ensure/sync LSLR ONCE per epoch ----------
    def _ensure_lslr_synced_once_per_epoch(theta0_full):
        if not use_lslr:
            return
        need_new = (not hasattr(model, "_lslr")) or (getattr(model._lslr, "inner_steps", None) != N)
        if not need_new:
            keys_needed = {n.replace(".", "-") for n in theta0_full.keys()}
            have = set(getattr(model._lslr, "_lrs", {}).keys())
            if not keys_needed.issubset(have):
                need_new = True

        if need_new:
            # Build fresh LSLR exactly over the frozen inner-loop params
            new_lslr = PerParamPerStepLSLR(
                theta0_full.items(), inner_steps=N, init_lr=alpha_init,
                learnable=True, device=device
            ).to(device)

            # Identify existing LSLR group (by id of old tensors), if any
            old_group_idx = None
            if hasattr(model, "_lslr"):
                old_ids = {id(p) for p in model._lslr.parameters()}
                for gi, g in enumerate(meta_opt.param_groups):
                    if any(id(p) in old_ids for p in g["params"]):
                        old_group_idx = gi
                        break

            # Swap module
            model._lslr = new_lslr

            # Replace optimizer param list in place (or add once)
            if old_group_idx is None:
                meta_opt.add_param_group({
                    "params": list(model._lslr.parameters()),
                    "lr": float(config["learning_rate"])
                })
            else:
                meta_opt.param_groups[old_group_idx]["params"] = list(model._lslr.parameters())
        else:
            # Ensure present if it wasn’t added yet (robustness)
            if not _optimizer_has_any_params(meta_opt, model._lslr.parameters()):
                meta_opt.add_param_group({
                    "params": model._lslr.parameters(),
                    "lr": float(config["learning_rate"])
                })

    # --- Generate a temp dict just for the LSLR setup ---
    # We just need the keys and the shape of the tensors to sync the LSLR parameters
    temp_theta_for_lslr = named_param_dict(
        model, 
        require_grad_only=True, 
        exclude_bn_from_inner=exclude_bn_from_inner
    )

    _ensure_lslr_synced_once_per_epoch(temp_theta_for_lslr)

    # Optional BN snapshot/restore per epoch
    #backup_bn_stats(model)

    def _normalize_step_item(step_item):
        if isinstance(step_item, dict): return [step_item]
        if isinstance(step_item, (list, tuple)): return list(step_item)
        raise TypeError(f"Episode item must be dict or list of dicts, got {type(step_item)}")

    # -------- Training loop with optional meta-batch accumulation --------
    meta_opt.zero_grad(set_to_none=True)
    accum_count = 0
    n_episodes = 0
    for step_item in episodic_loader:
        episodes = _normalize_step_item(step_item)  # --> This is supposed to return 1! 

        # --- PARTIAL BATCH FIX START ---
        # We determine how many episodes will contribute to the NEXT update.
        # It's usually 'meta_batchsize', unless we are near the end of the loader or the epoch limit.
        # We need to do this so if we do have many less eps in the batch, we scale the lr accordingly so our gradient step isnt tiny...
        # This is related to how many episodes are... in the limit or how many are available in the data loader?...
        remaining_in_epoch = episodes_per_epoch - n_episodes if episodes_per_epoch else float('inf')
        
        # This is the actual number of episodes we will process before calling meta_opt.step()
        actual_batch_size = min(meta_batchsize, remaining_in_epoch)

        if n_episodes % 100 == 0:
            print(f"--- Episode {n_episodes}/{episodes_per_epoch} | Realized Meta-Batch Size: {actual_batch_size} ---")
        # --- PARTIAL BATCH FIX END ---

        for episode in episodes:  # This doesnt actually loop FYI
            assert isinstance(episode, dict) and "support" in episode and "query" in episode, \
                f"Bad episode structure: {type(episode)} keys={list(episode) if isinstance(episode, dict) else None}"
            
            # 1. Capture current weights
            current_theta0 = named_param_dict(model, require_grad_only=True, exclude_bn_from_inner=exclude_bn_from_inner)

            # 2. Run Inner Loop
            thetaN, q_logits, q_labels, meta_loss_task = inner_loop_mamlpp(...)

            # 3. Calculate and scale loss
            # Use the CONSTANT meta_batch_size for scaling, not the local len(episodes)
            (meta_loss_task / actual_batch_size).backward()
            accum_count += 1
            n_episodes += 1

            # 4. Step only when we hit the target meta-batch size
            if accum_count >= actual_batch_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                meta_opt.step()
                meta_opt.zero_grad(set_to_none=True)
                accum_count = 0

            if episodes_per_epoch and n_episodes >= episodes_per_epoch:
                break

        if episodes_per_epoch and (n_episodes >= episodes_per_epoch):
            break

    if accum_count > 0:
        meta_opt.step()
        meta_opt.zero_grad(set_to_none=True)
    #finally:
    #    restore_bn_stats(model)

    avg_loss = loss_sum / n_episodes
    avg_acc  = meta_correct / meta_total
    return {"loss": avg_loss, "acc": avg_acc, "episodes": n_episodes}


# -----------------------------
# Outer loop (pretrain)
# -----------------------------
def MAMLpp_pretrain(model, config, episodic_train_loader, episodic_val_loader=None):
    """
    Drop-in replacement for MOE_pretrain.
    Uses set_MOE_optimizer to build the OUTER optimizer over model params (+ LSLR if enabled).
    Adds smoothed early stopping on validation loss (if a val loader is provided).
    """
    device = config["device"]
    model.to(device)

    # --- outer/meta optimizer
    meta_opt = set_MOE_optimizer(
        model,
        lr=float(config["learning_rate"]),
        use_weight_decay=float(config["weight_decay"]) > 0.0,
        weight_decay=float(config["weight_decay"]),
        optimizer_name=config["optimizer"],
    )

    # --- optional cosine schedule
    scheduler = None
    if bool(config["use_cosine_outer_lr"]):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            meta_opt, T_max=int(config["num_epochs"])
        )

    # --- early stopping: only if you have a validation loader
    use_es = episodic_val_loader is not None and bool(config["use_earlystopping"])
    early_stopping = None
    if use_es:
        early_stopping = SmoothedEarlyStopping(
            patience=int(config["earlystopping_patience"]),
            min_delta=float(config["earlystopping_min_delta"]),
            #smoothing_window=int(config["earlystopping_smoothing_window"]),  # I haven't been HPOing this one for whatever reason
        )

    # --- logs & best tracking
    best_val_acc, best_state = -1.0, None
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []

    num_epochs = int(config["num_epochs"])
    for ep in range(1, num_epochs + 1):
        print(f"MAML Pretraining: Epoch {ep} of {num_epochs}")
        epoch_start_time = time.time()

        print("Checking params to see if they are causing blow up:")
        print("meta_opt.param_groups:", len(meta_opt.param_groups))
        lslr_params = sum(p.numel() for p in (model._lslr.parameters() if hasattr(model, "_lslr") else []))
        print("lslr_params:", lslr_params)

        # ---- train over episodes
        train_start_time = time.time()
        train_metrics = train_MAMLpp_one_epoch(
            model, episodic_train_loader, meta_opt, config, epoch_idx=ep
        )
        train_loss_log.append(train_metrics["loss"])
        train_acc_log.append(train_metrics["acc"])
        total_train_time = time.time() - train_start_time
        print(f"Train completed in {total_train_time:.2f}s")
        print(f'Train loss/acc: {train_metrics["loss"]:.4f}, {train_metrics["acc"]*100:.2f}%')

        # ---- validation (if provided)
        cur_val_acc, cur_val_loss = None, None
        if episodic_val_loader is not None:
            val_start_time = time.time()
            # NOTE: This is not a user-specific evaluation. And I think that's fine here
            ## The meta-train is not user-specific, I dont see any reason why I care about specific user perf during val (grouped seems fine)
            val_metrics = meta_evaluate(model, episodic_val_loader, config)
            cur_val_loss, cur_val_acc = val_metrics["loss"], val_metrics["acc"]
            # NOTE: So we do have and maintain val_loss and val_acc logs! What happens to these? 
            ## Maybe they are saved and I just dont know where... I dont think I've had a successful completed HPO run (since it takes too long)
            ## Is the saving code for this already existing from the non-hpo runs??
            val_loss_log.append(cur_val_loss)
            val_acc_log.append(cur_val_acc)

            total_val_time = time.time() - val_start_time
            print(f"Val completed in {total_val_time:.2f}s")
            print(f"Val loss/acc: {cur_val_loss:.4f}, {cur_val_acc*100:.2f}%")
            #print()

            # Track best by accuracy (keeps your original behavior)
            if cur_val_acc is not None and cur_val_acc > best_val_acc:
                best_val_acc = cur_val_acc
                best_state = copy.deepcopy(model.state_dict())

            # Early stopping by smoothed validation loss
            if use_es and early_stopping(cur_val_loss):
                print(f"[EarlyStopping] epoch {ep}: smoothed val loss stalled "
                      f"(best={early_stopping.best_loss:.6f}, "
                      f"patience={early_stopping.patience}). Stopping.")
                # step the scheduler once more for bookkeeping (optional)
                if scheduler is not None:
                    scheduler.step()
                break
        else:
            raise ValueError("Why is Val Data Loader None?")
            # Keep list lengths aligned for consistency
            val_loss_log.append(None)
            val_acc_log.append(None)

        if scheduler is not None:
            scheduler.step()

        total_epoch_time =  time.time() - epoch_start_time
        print(f"Epoch completed in {total_epoch_time:.2f}s\n")

    # ---- load best by validation accuracy (if available)
    if episodic_val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[MAML++] loaded best model (val acc={best_val_acc:.3f})")

    return model, {
        "train_loss_log": train_loss_log,
        "train_acc_log":  train_acc_log,
        "val_loss_log":   val_loss_log if episodic_val_loader is not None else None,
        "val_acc_log":    val_acc_log  if episodic_val_loader is not None else None,
        "model": model,
        "best_state": best_state,
        "best_val_acc": best_val_acc,
    }
# === End MAML++ core ==========================================================

def _optimizer_has_any_params(opt, params_iterable):
    existing = {id(p) for g in opt.param_groups for p in g["params"]}
    return any(id(p) in existing for p in params_iterable)

def _replace_or_add_param_group(opt, old_params_iterable, new_params_iterable, *, lr):
    new_params = list(new_params_iterable)
    if old_params_iterable is not None:
        old_ids = {id(p) for p in old_params_iterable}
        for g in opt.param_groups:
            if any(id(p) in old_ids for p in g["params"]):
                g["params"] = new_params
                g["lr"] = lr
                return
    # no old group found → add new
    if not _optimizer_has_any_params(opt, new_params):
        opt.add_param_group({"params": new_params, "lr": lr})


############################################################################################

# HOW TO ADAPT OUR META MODEL

def mamlpp_adapt(model, config, support_batch, *, use_lslr_at_eval=False):
    """This function does the inner loop only, and uses .detach() and False graph bc we do NOT want to update the baes (meta) model"""
    device = config['device']; model.to(device).eval()
    multimodal = bool(config["multimodal"])
    N_eval = int(config["maml_inner_steps_eval"])

    # --- guards ---
    #if hasattr(torch, "is_inference_mode_enabled"):
    #    assert not torch.is_inference_mode_enabled(), \
    #        "mamlpp_adapt cannot run under torch.inference_mode(). Remove that decorator/context."

    # build theta and ensure leaf tensors require grad
    exclude_bn = not bool(config["enable_inner_loop_optimizable_bn_params"])
    theta = named_param_dict(model, require_grad_only=True, exclude_bn_from_inner=exclude_bn)
    theta = OrderedDict((n, p.detach().requires_grad_(True)) for n, p in theta.items())
    assert all(p.requires_grad for p in theta.values()), "theta params do not require_grad (grad disabled?)"

    label_smooth = float(config["label_smooth"])
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    use_lslr = bool(config["maml_use_lslr"]) and use_lslr_at_eval and hasattr(model, "_lslr")
    alpha_eval = float(config["maml_alpha_init_eval"])

    #backup_bn_stats(model)

    # ADAPTATION MUST RUN WITH GRADS ENABLED
    with torch.enable_grad():
        for step in range(N_eval):
            #set_bn_step(model, step)
            f_s = FunctionalModel(model, theta)
            outputs_s, labels_s, _ = _model_forward_router(f_s, support_batch, device, multimodal=multimodal)
            logits_s = outputs_s[0] if isinstance(outputs_s, tuple) else outputs_s
            s_loss = criterion(logits_s, labels_s)

            grads = torch.autograd.grad(
                s_loss, list(theta.values()),
                create_graph=False, allow_unused=True
            )

            lslr = model._lslr if use_lslr else None
            theta = apply_update_repo_style(theta, grads, lslr=lslr, step=step, fallback_alpha=alpha_eval)

            # FOMAML: cut graph between steps, keep trainable
            theta = OrderedDict((n, p.detach().requires_grad_(True)) for n, p in theta.items())

    #restore_bn_stats(model)
    return theta


@torch.no_grad()
def mamlpp_predict_with_params(model, adapted_params, batch, config):
    """
    Run the model with a given adapted parameter set (theta') to get logits/preds on any batch.
    Does not modify model weights.
    """
    device = config['device']; model.to(device).eval()
    multimodal = bool(config["multimodal"])

    # If you use BN-per-step wrappers, you can select a stable bank, e.g. the last step index:
    N_eval = int(config["maml_inner_steps_eval"])
    #set_bn_step(model, N_eval)
    #set_bn_step(model, max(N_eval - 1, 0))

    fmodel = FunctionalModel(model, adapted_params)
    outputs, labels, B = _model_forward_router(fmodel, batch, device, multimodal=multimodal)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs

    preds = torch.argmax(logits, dim=1)
    return logits, preds, labels, B


#@torch.no_grad()
def mamlpp_finetune_and_eval(model, config, support_batch, query_batch):
    """
    Convenience helper: (1) adapt on support, (2) evaluate on query, (3) return metrics and preds.
    """
    theta_prime = mamlpp_adapt(model, config, support_batch)
    logits, preds, labels, B = mamlpp_predict_with_params(model, theta_prime, query_batch, config)

    # CE loss for reporting (same criterion as above)
    label_smooth = float(config["label_smooth"])
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth) if label_smooth > 0 else nn.CrossEntropyLoss()
    q_loss = criterion(logits, labels).item()
    acc = (preds == labels).sum().item() / max(1, B)

    return {
        "loss": q_loss,
        "acc": acc,
        "preds": preds.detach().cpu(),
        "labels": labels.detach().cpu(),
        "adapted_params": theta_prime,  # in case you want to reuse them for this user
    }


# Example usage at test time
# with torch.no_grad():
#     result = mamlpp_finetune_and_eval(
#         model,
#         config,
#         support_batch=new_user_support_batch,
#         query_batch=new_user_query_batch,
#     )

# print("new user acc:", result["acc"])
# # If you want to reuse the adapted parameters (e.g., to run inference on more data from this user):
# theta_user = result["adapted_params"]

# # Later, for more batches from the same user (no re-adaptation):
# logits, preds, labels, B = mamlpp_predict_with_params(model, theta_user, another_batch_from_user, config)


# -----------------------------
# Meta-evaluation (no graph)
# -----------------------------
# How much adaptation are we doing in the test set? 
## We dont really need to do every single possible task on our 4 test users...
## It seems to be pretty low, meta-eval is only taking like 20s
## That is fine/expected, it is only doing a few gradient steps on the fast inner loop, no outer loop (thus no Hessian)
def meta_evaluate(model, episodic_loader, config, criterion=None):
    device = config["device"]; model.to(device).eval()
    if criterion is None:
        ls = float(config["label_smooth"])
        criterion = nn.CrossEntropyLoss(label_smoothing=ls) if ls > 0 else nn.CrossEntropyLoss()

    def _norm(x):
        if isinstance(x, dict): return [x]
        if isinstance(x, (list, tuple)): return list(x)
        raise TypeError(f"Episode item must be dict or list/tuple; got {type(x)}")

    total_loss = total_correct = total_count = n_eps = 0

    for step_item in episodic_loader:
        for ep in _norm(step_item):
            # Just call the helper!
            metrics = mamlpp_finetune_and_eval(model, config, ep["support"], ep["query"])
            
            # Aggregate
            total_loss    += metrics["loss"]
            # Note: finetune returns a ratio (acc), so you might need to grab 
            # raw counts if you want perfect global accuracy, but averaging per-task acc is also common.
            total_correct += (metrics["acc"] * len(ep["query"])) # approximated reverse engineering
            total_count   += len(ep["query"])
            n_eps         += 1

    #This is actually the standard convention in Meta-Learning papers:
    #Loss: Reported as "Mean Loss per Task." (How confused is the model usually?)
    #Accuracy: Reported as "Global Accuracy" (Total Correct / Total Queries) OR "Mean Accuracy per Task.
    return {
        "loss": total_loss / max(n_eps, 1),
        "acc":  (total_correct / max(total_count, 1)) if total_count else 0.0,
        "episodes": n_eps,
    }
