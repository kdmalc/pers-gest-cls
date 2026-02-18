# So I think this is the "traditional" altnerative to MAML
## If I wasnt doing meta learnig then I would need to do this stuff for normal ML training, and finetuning, etc
## I think this file tries to implement PEFT but it doesnt do it very well (None in WithUserOverride results in all zeros? Or maybe the weighted init vector? Either way useless)
## MAML might pull some data processing funcs from here? Idk tbh

import random, math, copy, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

#from MOE_model_classes import WithUserOverride
from system.MAML_MOE.MOE_model_classes import WithUserOverride

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from contextlib import contextmanager


# -----------------
# Freeze/unfreeze helpers
# -----------------
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

@contextmanager
def temporarily_frozen(module: nn.Module):
    prev = [p.requires_grad for p in module.parameters()]
    set_requires_grad(module, False)
    try:
        yield
    finally:
        for p, f in zip(module.parameters(), prev):
            p.requires_grad_(f)

def _check_trainable(model, printed_message, config):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        print(f"[{printed_message}]: No trainable parameters! Strategy: {config['finetune_strategy']}")
        for name, param in model.named_parameters():
            print(f"  {name}: requires_grad={param.requires_grad}")
        raise RuntimeError("No trainable parameters - freezing strategy failed")

def apply_dropout_flag(mod, enabled):
    for m in mod.modules():
        if isinstance(m, nn.Dropout):
            m.p = m.p if enabled else 0.0

def _rebuild_ft_opt_sched_if_needed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    config: dict,
    opt_class=None,  # kept for signature compatibility; unused now
    lr: float | None = None,
    wd: float | None = None,
):
    """
    Rebuild optimizer/scheduler after (re)setting requires_grad masks.
    Always returns a FRESH optimizer bound to current trainable params.
    """
    # Resolve hyperparams
    lr = config["ft_learning_rate"] if lr is None else lr
    wd = config["ft_weight_decay"] if wd is None else wd
    opt_name = config["optimizer"]
    use_wd = (wd > 0.0)

    # Build fresh optimizer tied to *current* trainable params
    optimizer = set_MOE_optimizer(
        model=model,
        lr=lr,
        use_weight_decay=use_wd,
        weight_decay=wd,
        optimizer_name=opt_name,
    )

    # Rebuild scheduler if requested/present
    if scheduler is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["ft_lr_scheduler_factor"],
            patience=config["ft_lr_scheduler_patience"],
        )

    return optimizer, scheduler


def handle_MOE_batches(batch_idx, batch, num_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Add some config[verbose] or something here for data aug version
    ## Don't use config[verbose] verbatim bc it probably is used elsewhere in places idc about...
    #if num_batches is not None and (batch_idx==0 or batch_idx%500==0):
    #    print(f"Starting batch {batch_idx}/{num_batches}!")

    # Unpack depending on if batch is a tuple or list
    if len(batch) == 4:
        #features, (now encoded) gesture_names, participant_ids, gesture_nums = batch
        batch_features, batch_labels, _, _ = batch
        batch_features = batch_features.reshape(-1, 16, 5)

        # If labels are numpy or wrong type, convert to torch.long and move to device
        if not torch.is_tensor(batch_labels):
            batch_labels = torch.tensor(batch_labels, dtype=torch.LongTensor)
        else:
            if batch_labels.dtype != torch.long:
                batch_labels = batch_labels.long()
    elif len(batch) == 2:
        batch_features, batch_labels = batch
    else:
        raise ValueError(f"Expected either 2 or 4 elements from dataset, got {len(batch)}")
    
    batch_features = batch_features.to(device)
    batch_labels = batch_labels.to(device)
    #print("Batch set up complete!")
    return batch_features, batch_labels


def _to_device(x, device):
    return x.to(device) if torch.is_tensor(x) else x

def _model_forward_router(model, batch, device, multimodal: bool):
    """
    Returns: (logits_or_tuple, labels_tensor, batch_size)
    - If `multimodal` is True we call model with named args.
      Otherwise we call model(features) like the old path. --> This is quite old and can probably be removed...
    """
    # Dict batch (new or old)
    if isinstance(batch, dict):
        # labels
        labels = batch['labels']
        if labels is None:
            raise KeyError("Batch missing 'labels' key.")
        labels = _to_device(labels.long(), device)
        B = labels.size(0)

        if multimodal:
            emg = _to_device(batch["emg"], device)
            # TODO: Is this how I want to handle this if IMU / DEMO are missing... not sure... fail quietly...
            # TODO: pass in config and then add print outs if demo or imu (or emg) are empty/None but use_imu/etc are turned on
            # NOTE: These should NOT be pulling use_imu and use_demographic (not directly anyways); batch['imu'] is its own thing
            imu = _to_device(batch["imu"], device) if batch.get("imu", None) is not None else None
            demo = _to_device(batch["demo"], device) if batch.get("demo", None) is not None else None
            #pids = _to_device(batch["PIDs"], device) if batch.get("PIDs", None) is not None else None

            outputs = model(
                x_emg=emg,
                x_imu=imu,
                demographics=demo,
            )
            return outputs, labels, B
        else:
            # Unimodal (EMG-only) legacy path (old MOE or other single-input models)
            x = _to_device(batch["emg"], device)
            # TODO: Should this be logits, aux?
            outputs = model(x)
            return outputs, labels, B

    # Tuple/list batch (legacy)
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, labels = batch
        x = _to_device(x, device)
        labels = _to_device(labels.long(), device)
        B = labels.size(0)
        outputs = model(x)
        return outputs, labels, B

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def train_MOE_one_epoch(model, train_loader, optimizer, config, criterion=None):
    """
    One epoch of training.

    Pulls multimodal from the config.
    - `multimodal=True` -> expects dict batches from your new dataloader and calls model with named args.
    - `multimodal=False` -> stays compatible with your old unimodal pipeline.
    """
    device = config['device']
    model.to(device).train()

    if criterion is None:
        # Use label smoothing if provided
        label_smooth = 0.0
        if config is not None and "label_smooth" in config:
            label_smooth = float(config["label_smooth"])
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    total_loss_sum = 0.0   # accumulate as sum over samples
    correct = 0
    total = 0

    for batch in train_loader:
        optimizer.zero_grad()

        outputs, labels, B = _model_forward_router(model, batch, device, multimodal=config['multimodal'])

        # Accept either (logits, aux) or logits only
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        # Helpful sanity checks
        if not any(p.requires_grad for p in model.parameters()):
            fs = config["finetune_strategy"]
            raise AssertionError(f"All model params are frozen (finetune_strategy={fs}).")

        if not logits.requires_grad:
            fs = config["finetune_strategy"]
            raise AssertionError(f"Logits are detached (finetune_strategy={fs}).")

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # metrics
        total_loss_sum += loss.item() * B          # true per-sample average at the end
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += B

    return {
        "loss": total_loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
    }


def evaluate_MOE_model(model, dataloader, config, criterion=None, device=None):
    """
    Evaluation with no gradient.
    - `multimodal=True` -> named multimodal call path.
    - `multimodal=False` -> legacy unimodal path.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if device is None:
        device = config['device']
    else:
        device = device
    model.to(device).eval()

    total_loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs, labels, B = _model_forward_router(model, batch, device, multimodal=config['multimodal'])

            # Accept either (logits, aux) or logits only
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits, labels)
            total_loss_sum += loss.item() * B

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += B

    return {
        "loss": total_loss_sum / total,
        "acc": correct / total,
    }


def freeze_module(module: nn.Module, freeze: bool):
     for p in module.parameters():
         p.requires_grad = not freeze
     for child in module.children():
         freeze_module(child, freeze)

def apply_finetune_freezing(model, strategy: str):
    """
    Freezing policy for MOE models.
    Expects model to have attributes:
      - model.backbone (feature extractor)
      - model.experts (iterable of expert modules)
      - model.gate (gating module)
    And each expert to expose submodules: fc1, norm, drop, fc2
    """
    # backbone, experts, gate
    if strategy == "linear_probing":
        # TODO: This only freezes everything, it doesn't add a final linear layer...
        # freeze everything except the final heads (fc2 / cosine)
        freeze_module(model.backbone, True)
        for exp in model.experts:
            freeze_module(exp.fc1, True)
            freeze_module(exp.norm, True)
            freeze_module(exp.drop, True)
            freeze_module(exp.fc2, False)  # train only last layer
        freeze_module(model.gate, True)

    elif strategy == "experts_only":
        freeze_module(model.backbone, True)
        freeze_module(model.gate, True)
        for exp in model.experts:
            freeze_module(exp, False)

    elif strategy == "experts_plus_gate":
        freeze_module(model.backbone, True)
        freeze_module(model.gate, False)
        for exp in model.experts:
            freeze_module(exp, False)

    elif strategy == "full":
        # everything trainable
        freeze_module(model, False)
    else:
        raise ValueError(f"[apply_finetune_freezing] Unknown strategy: {strategy}")
    
class SmoothedEarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, smoothing_window=5):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.best_loss = np.inf
        self.num_bad_epochs = 0
        self.loss_buffer = deque(maxlen=smoothing_window)

    def __call__(self, current_loss):
        self.loss_buffer.append(current_loss)
        smoothed_loss = np.mean(self.loss_buffer)

        if smoothed_loss < self.best_loss - self.min_delta:
            self.best_loss = smoothed_loss
            self.num_bad_epochs = 0  # Reset patience if improvement
        else:
            self.num_bad_epochs += 1  # Increment if no improvement

        return self.num_bad_epochs >= self.patience  # Stop if patience exceeded

#def set_MOE_optimizer(model, config, optimizer_name=None, lr=None, use_weight_decay=None, weight_decay=None)
def set_MOE_optimizer(
    model: torch.nn.Module,
    lr: float,
    use_weight_decay: bool,
    weight_decay: float,
    optimizer_name: str,
) -> torch.optim.Optimizer:
    """
    Build and return an optimizer bound to the model's *current trainable* parameters.
    Only parameters with requires_grad=True are optimized.

    Raises:
        RuntimeError: if no trainable parameters are found.
        ValueError: for unsupported optimizer names.
    """
    # Filter to trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError(
            "No trainable parameters found (all `requires_grad=False`). "
            "Check your freezing/unfreezing logic before building the optimizer."
        )

    name = optimizer_name.upper()
    wd = float(weight_decay) if use_weight_decay else 0.0

    if name == "ADAM":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif name == "ADAMW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "SGD":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Only ADAM, ADAMW, and SGD are supported; got {optimizer_name}")
    
def MOE_fine_tune_model(finetuned_model, fine_tune_loader, config: dict, timestamp: str, val_loader=None, pid=None, num_epochs=None):
    """
    Fine-tune a Mixture-of-Experts model with MOE-specific freezing strategies
    and return the same structure as your original fine_tune_model().
    """

    ####################################################################################################
    trainable_params = sum(p.numel() for p in finetuned_model.parameters() if p.requires_grad)
    if trainable_params == 0:
        print(f"MOE_fine_tune_model 1: No trainable parameters! Strategy: {finetune_strategy}")
        for name, param in finetuned_model.named_parameters():
            print(f"  {name}: requires_grad={param.requires_grad}")
        raise RuntimeError("No trainable parameters - freezing strategy failed")
    ####################################################################################################

    # --- Setup / bookkeeping ---
    finetune_strategy = config["finetune_strategy"]  # expected to be one of:
                                                     # "full", "experts_only",
                                                     # "experts_plus_gate", "linear_probing"
    pid_str = "" if pid is None else f"{pid}_"

    max_epochs = config["num_ft_epochs"] if num_epochs is None else num_epochs
    finetuned_model.train()

    # --- Apply MOE-specific freezing ---
    apply_finetune_freezing(finetuned_model, finetune_strategy)

    optimizer = set_MOE_optimizer(
        finetuned_model,
        lr=config["ft_learning_rate"],
        use_weight_decay=config["ft_weight_decay"] > 0,
        weight_decay=config["ft_weight_decay"],
        optimizer_name=config["optimizer"],
    )
    scheduler = None
    if config["ft_lr_scheduler_factor"] > 0.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config["ft_lr_scheduler_patience"],
            factor=config["ft_lr_scheduler_factor"],
        )

    early_stopping = None
    if config["use_ft_earlystopping"]:
        early_stopping = SmoothedEarlyStopping(
            patience=config["ft_earlystopping_patience"],
            min_delta=config["ft_earlystopping_min_delta"],
        )

    # --- Logs ---
    train_loss_log = []
    intra_test_loss_log = []
    train_acc_log = []
    intra_test_acc_log = []

    # Optional per-PID text log
    log_each = bool(config["log_each_pid_results"])
    log_fh = None
    if log_each:
        log_fh = open(f"{timestamp}_{pid_str}ft_log.txt", "w")

    ####################################################################################################
    trainable_params = sum(p.numel() for p in finetuned_model.parameters() if p.requires_grad)
    if trainable_params == 0:
        print(f"MOE_fine_tune_model 7: No trainable parameters! Strategy: {finetune_strategy}")
        for name, param in finetuned_model.named_parameters():
            print(f"  {name}: requires_grad={param.requires_grad}")
        raise RuntimeError("No trainable parameters - freezing strategy failed")
    ####################################################################################################

    # Optional: compute a statistic-based warm start once
    #support_stat_init = None
    #if config["gate_requires_u_user"] and config["use_u_init_warm_start"]:
    support_stat_init = uservec_from_support_mean(finetuned_model, fine_tune_loader, config)

    # --- Training loop ---

    if config["alt_or_seq_MOE_user_emb_ft"].upper() == "SEQUENTIAL":
        # ---- Stage A: PEFT (optimize u_user once; model frozen inside PEFT) ----
        if config["gate_requires_u_user"]:
            # TODO: This seems really redundant. I wrap the model, pass in support_stat_init
            # Then in peft I pass in the wrapped model and the same init again
            # Then I have to call .set_u_user on the model to make sure it is updated (this probably could be done inside peft_user_emb_vec)
            finetuned_model = WithUserOverride(finetuned_model, support_stat_init, multimodal=config['multimodal'])
            u_user, moe_logs = peft_user_emb_vec(
                copy.deepcopy(finetuned_model), fine_tune_loader, config, u_init=support_stat_init
            )
            # Inject once; ensure model is back in train mode afterwards
            finetuned_model.set_u_user(u_user)
            finetuned_model.train()

        # If your strategy changes which layers are trainable, apply masks now (before optimizer)
        # e.g., set_requires_grad_masks(finetuned_model, strategy=config["finetune_strategy"])
        # (Assumes you already froze/unfroze earlier; if not, call your mask setter here.)
        _check_trainable(finetuned_model, "sequential-before-opt", config)

        # Rebuild optimizer/scheduler to reflect current requires_grad
        optimizer, scheduler = _rebuild_ft_opt_sched_if_needed(finetuned_model, optimizer, scheduler, config)

        # ---- Stage B: Fine-tune model weights with fixed u_user ----
        for epoch in range(1, max_epochs + 1):
            finetuned_model.train()
            train_metrics = train_MOE_one_epoch(finetuned_model, fine_tune_loader, optimizer, config)
            train_loss_log.append(train_metrics["loss"])
            train_acc_log.append(train_metrics["acc"])

            # ADDING A SKIP SO IT DOESNT EVAL EVERY EPOCH DURING FT
            if epoch%3==0:
                if val_loader is not None:
                    finetuned_model.eval()
                    intra_test_metrics = evaluate_MOE_model(finetuned_model, val_loader, config)
                    intra_test_loss_log.append(intra_test_metrics["loss"])
                    intra_test_acc_log.append(intra_test_metrics["acc"])

                if scheduler is not None:
                    if len(intra_test_loss_log) > 0:
                        scheduler.step(intra_test_loss_log[-1])
                    else:
                        scheduler.step(train_loss_log[-1])

                if early_stopping is not None:
                    current_loss_for_es = intra_test_loss_log[-1] if len(intra_test_loss_log) > 0 else train_loss_log[-1]
                    if early_stopping(current_loss_for_es):
                        if config["verbose"]:
                            print(f"FT {pid_str[:-1]}: Early stopping reached after {epoch} epochs")
                        break

    elif config["alt_or_seq_MOE_user_emb_ft"].upper() == "ALTERNATING":
        #current_u_user = None  # placeholder for warm-start with the last PEFT solution
        current_u_user = support_stat_init
        for epoch in range(1, max_epochs + 1):
            #print(f"ALTERNATING; epoch number {epoch+1}")

            # ---- Inner PEFT (optimize u_user) ----
            if config["gate_requires_u_user"]:
                # Prefer warm-starting from previous 'u_user' if available; else from support statistic; else None
                u_init = current_u_user if current_u_user is not None else support_stat_init
                # TODO: This seems really redundant. Just as in the seq case
                finetuned_model = WithUserOverride(finetuned_model, u_init, multimodal=config['multimodal'])
                current_u_user, moe_logs = peft_user_emb_vec(
                    copy.deepcopy(finetuned_model), fine_tune_loader, config, u_init=u_init
                )
                # Inject u_user
                finetuned_model.set_u_user(current_u_user)
                finetuned_model.train()

            # Safety: ensure something is trainable for the model step
            #_check_trainable(finetuned_model, f"alternating-epoch{epoch}-before-train", config)

            # ---- One epoch of model training with u_user fixed ----
            train_metrics = train_MOE_one_epoch(finetuned_model, fine_tune_loader, optimizer, config)
            train_loss_log.append(train_metrics["loss"])
            train_acc_log.append(train_metrics["acc"])

            if epoch%3==0:
                if val_loader is not None:
                    finetuned_model.eval()
                    intra_test_metrics = evaluate_MOE_model(finetuned_model, val_loader, config)
                    intra_test_loss_log.append(intra_test_metrics["loss"])
                    intra_test_acc_log.append(intra_test_metrics["acc"])

                if scheduler is not None:
                    if len(intra_test_loss_log) > 0:
                        scheduler.step(intra_test_loss_log[-1])
                    else:
                        scheduler.step(train_loss_log[-1])

                if early_stopping is not None:
                    current_loss_for_es = intra_test_loss_log[-1] if len(intra_test_loss_log) > 0 else train_loss_log[-1]
                    if early_stopping(current_loss_for_es):
                        if config["verbose"]:
                            print(f"FT {pid_str[:-1]}: Early stopping reached after {epoch} epochs")
                        break
    else:
        raise ValueError(f"Value {config['alt_or_seq_MOE_user_emb_ft']} not recognized! Must be 'sequential' or 'alternating'")

    # --- Final logging ---
    # Prepare console/file log message
    last_epoch = len(train_loss_log)
    final_train_loss = train_loss_log[-1] if len(train_loss_log) else float("nan")
    final_val_loss = intra_test_loss_log[-1] if len(intra_test_loss_log) else float("nan")

    log_message = (
        f"Participant ID {pid_str[:-1]}, "
        f"Epoch {last_epoch}/{max_epochs}, "
        f"FT Train Loss: {final_train_loss:.4f}, "
        f"Novel Intra Subject Testing Loss: {final_val_loss:.4f}\n"
    )
    if config["verbose"]:
        print(log_message, end="")
    if log_fh is not None:
        log_fh.write(log_message)
        log_fh.close()

    # --- Final metrics for return dict ---
    if val_loader is not None:
        final_intra_test_acc = evaluate_MOE_model(finetuned_model, val_loader, config)["acc"]
    else:
        final_intra_test_acc = []

    return {
        "finetuned_model": finetuned_model,
        "train_accuracy": train_metrics["acc"],  #evaluate_MOE_model(finetuned_model, fine_tune_loader, config)["acc"],
        "intra_test_accuracy": final_intra_test_acc,
        "train_loss_log": train_loss_log,
        "train_acc_log": train_acc_log,
        "intra_test_loss_log": intra_test_loss_log,
        "intra_test_acc_log": intra_test_acc_log,
    }

# ideal entropy for E=6 is log(6) ≈ 1.792
def gate_stats(usage):  # usage: (E,) on CUDA
    u = usage.detach().float().cpu().clamp_min(1e-8)
    E = u.numel()
    ent = float(-(u * u.log()).sum())              # entropy
    kl  = float(F.kl_div(u.log(), torch.full_like(u, 1.0/E), reduction='sum'))  # KL(u || uniform)
    return ent, kl, u.tolist()

def accuracy(logits, y):
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()

def gate_balance_loss(avg_usage, eps=1e-8):
    # KL( avg_usage || uniform )
    E = avg_usage.numel()
    target = torch.full_like(avg_usage, 1.0 / E)
    return F.kl_div((avg_usage + eps).log(), target, reduction='batchmean')

def make_ce(label_smooth=0.05):
    return nn.CrossEntropyLoss(label_smoothing=label_smooth)


def MOE_pretrain(model, config, train_loader, val_loader=None):
    device = config["device"]
    model.to(device)
    
    opt = set_MOE_optimizer(
        model,
        lr=config["learning_rate"],
        use_weight_decay=config["weight_decay"] > 0,
        weight_decay=config["weight_decay"],
        optimizer_name=config["optimizer"],
    )

    best_val_acc, best_state = -1.0, None

    # Logs
    train_loss_log, train_acc_log = [], []
    val_loss_log, val_acc_log = [], []

    for ep in range(1, config["num_epochs"]+1):
        # TODO: Ought to put this somewhere better or add a toggle...
        #if ep%5==0:
        #    print(f"EPOCH {ep}: Train (loss, acc): ({train_metrics['loss']}, {train_metrics['acc']})")
        #    if val_loader is not None:
        #        print(f"Val (loss, acc): ({val_metrics['loss']}, {val_metrics['acc']})\n")

        train_metrics = train_MOE_one_epoch(model, train_loader, opt, config)
        train_loss_log.append(train_metrics['loss'])
        train_acc_log.append(train_metrics['acc'])

        if val_loader is not None:
            val_metrics = evaluate_MOE_model(model, val_loader, config)
            val_loss, val_acc = val_metrics['loss'], val_metrics['acc']
            val_loss_log.append(val_loss)
            val_acc_log.append(val_acc)

            #print(f"[epoch {ep:03d}] train: loss={tr_loss:.4f} acc={tr_acc:.3f} "
            #      f"| val: loss={val_loss:.4f} acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                # This makes it save the pretrained model every time, including during hyperparam tuning, which is unnecessary
                #torch.save(best_state, config["models_save_dir"]+"_best_state.pt")
        #else:
        #    print(f"[epoch {ep:03d}] train: loss={tr_loss:.4f} acc={tr_acc:.3f}")

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[pretrain] loaded best model (val acc={best_val_acc:.3f})")  # from {config['models_save_dir']+'_best_state.pt'}

    # Always return logs so you can inspect curves
    logs = {
        "train_loss": train_loss_log,
        "train_acc": train_acc_log,
        "val_loss": val_loss_log if val_loader is not None else None,
        "val_acc": val_acc_log if val_loader is not None else None,
        "model": model,
        "best_state": best_state,
        "best_val_acc": best_val_acc
    }

    return model, logs

@torch.no_grad()
def uservec_from_support_mean(model, support_loader, config):
    """
    unimodal=False (legacy): expect (x, _) and call model.backbone(x).
    multimodal=True: expect dict batches; call model with named args (via router)
                     and use aux['fused_h'] as features to average.
    """
    device = config["device"]
    multimodal = bool(config["multimodal"])
    model.eval()

    feats = []

    if multimodal:
        for batch in support_loader:
            outputs, _labels, _B = _model_forward_router(
                model, batch, device, multimodal=True
            )
            # Expect (logits, aux) where aux has 'fused_h'
            if not (isinstance(outputs, tuple) and len(outputs) >= 2):
                raise RuntimeError("Model must return (logits, aux) when multimodal=True.")
            _logits, aux = outputs
            if not isinstance(aux, dict) or "fused_h" not in aux:
                raise RuntimeError("aux dictionary missing 'fused_h'. Please return aux['fused_h']=<feature tensor>.")
            h = aux["fused_h"]  # shape (B, D)
            if h.dim() != 2:
                # If your fused_h has shape (B, D, ...) reduce over extra dims if needed
                h = h.flatten(start_dim=1)
            feats.append(h)
    else:
        for x, _ in support_loader:
            x = x.to(device)
            h = model.backbone(x)   # (B, D)
            feats.append(h)

    if not feats:
        raise ValueError("Support loader produced no samples.")
    H = torch.cat(feats, dim=0)  # (N, D)
    h_bar = H.mean(dim=0, keepdim=True)  # (1, D)

    # Project to user_emb_dim if needed using identity-padded linear map
    target_dim = int(config["user_emb_dim"])
    if h_bar.size(-1) != target_dim:
        proj = nn.Linear(h_bar.size(-1), target_dim, bias=False).to(device)
        with torch.no_grad():
            proj.weight.zero_()
            k = min(h_bar.size(-1), target_dim)
            proj.weight[:k, :k] = torch.eye(k, device=device)
        u0 = proj(h_bar)
    else:
        u0 = h_bar

    return F.layer_norm(u0, u0.shape[-1:])


def _set_rnns_train(model: nn.Module, enable: bool, zero_dropout: bool = True):
    for m in model.modules():
        if isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
            m.train(enable)  # required for cuDNN backward
            if zero_dropout and hasattr(m, "dropout"):
                # when enabling train, set internal LSTM dropout to 0 for determinism
                if enable:
                    m.dropout = 0.0
            try:
                if enable:
                    m.flatten_parameters()
            except Exception:
                pass

def peft_user_emb_vec(model, support_loader, config, u_init=None):
    steps = int(config["num_ft_epochs"])
    device = config["device"]
    multimodal = bool(config["multimodal"])

    if not isinstance(model, WithUserOverride):
        raise ValueError("peft_user_emb_vec expects a WithUserOverride-wrapped model.")

    # Prepare user param
    model.begin_user_training(u_init)
    u_param = [model.u_user_param]
    assert u_param[0].requires_grad, "User embedding is not trainable (requires_grad=False)."

    opt = torch.optim.AdamW(u_param, lr=config["ft_learning_rate"], weight_decay=config["ft_weight_decay"])
    ce  = make_ce(config["label_smooth"])

    # Dense warmup for gate if requested
    dense_warmup = config.get("gate_dense_before_topk", False)
    original_top_k = getattr(model.gate, "top_k", None)
    if dense_warmup:
        model.gate.top_k = None

    loss_log, acc_log = [], []

    # Keep global modules deterministic (BN/Dropout off)
    model.eval()

    # Enable only RNNs' train mode during updates (cuDNN backward requirement)
    _set_rnns_train(model, True, zero_dropout=True)

    try:
        torch.set_grad_enabled(True)

        for step in range(steps):
            # end dense warmup after ~1/3 steps
            if dense_warmup and step == max(1, steps // 3):
                model.gate.top_k = config["top_k"]

            epoch_loss, correct, n = 0.0, 0, 0

            for batch in support_loader:
                if multimodal:
                    outputs, y, B = _model_forward_router(model, batch, device, multimodal=True)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    batch_size = B
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    batch_size = x.size(0)

                loss = ce(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()    # works: RNNs are in train() even though model is eval()
                opt.step()

                epoch_loss += loss.item() * batch_size
                correct    += (logits.argmax(-1) == y).sum().item()
                n          += batch_size

            loss_log.append(epoch_loss / max(1, n))
            acc_log.append(correct / max(1, n))

    finally:
        # Restore RNN modes and gate.k
        _set_rnns_train(model, False, zero_dropout=False)
        model.gate.top_k = original_top_k
        model.eval()

    # Commit learned vector and return
    u_learned = model.u_user_param.detach().clone()
    model.end_user_training(commit=True)
    return u_learned, {"loss": loss_log, "acc": acc_log}


def _unpack_batch(batch, expect_user_id=True, device="cuda", num_classes=None, num_users_train=None):
    if isinstance(batch, dict):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        #user_ids = batch.get("user_id", None)  # TODO: Idk if this works...
        user_ids = batch["user_id"]
        if user_ids is not None:
            user_ids = user_ids.to(device)
    else:
        x = batch[0].to(device)
        y = batch[1].to(device)
        user_ids = batch[2].to(device) if expect_user_id and len(batch) > 2 else None

    if y.dtype != torch.long:
        y = y.long()

    # Range checks on CPU → clean Python errors
    with torch.no_grad():
        y_cpu = y.detach().cpu()
        if num_classes is not None:
            y_min, y_max = int(y_cpu.min()), int(y_cpu.max())
            if y_min < 0 or y_max >= num_classes:
                raise ValueError(f"[labels] found y in [{y_min}, {y_max}] but NUM_CLASSES={num_classes} "
                                 f"(expected 0..{num_classes-1}).")

        if user_ids is not None and num_users_train is not None:
            u_cpu = user_ids.detach().cpu()
            u_min, u_max = int(u_cpu.min()), int(u_cpu.max())
            if u_min < 0 or u_max >= num_users_train:
                raise ValueError(f"[user_id] found user_id in [{u_min}, {u_max}] but num_users_train={num_users_train} "
                                 f"(expected 0..{num_users_train-1}).")

    return x, y, user_ids

@torch.no_grad()
def debug_one_batch(model, loader, device="cuda", num_classes=None, num_users_train=None):
    model.eval()
    for batch in loader:
        x, y, user_ids = _unpack_batch(batch, expect_user_id=True, device=device,
                                       num_classes=num_classes, num_users_train=num_users_train)
        print(f"[debug] x {tuple(x.shape)} dtype={x.dtype}  y range [{int(y.min())}, {int(y.max())}]")
        if user_ids is not None:
            print(f"[debug] user_ids range [{int(user_ids.min())}, {int(user_ids.max())}] "
                  f"(num_users_train={num_users_train})")
        logits, aux = model(x, user_ids=user_ids, user_embed_override=None)#, return_aux=True)
        print(f"[debug] logits {tuple(logits.shape)}  gate_usage {tuple(aux['gate_usage'].shape)} "
              f"sum={aux['gate_usage'].sum().item():.3f}")
        break

@torch.no_grad()
def build_user_prototype(model, support_x):
    # support_x: (N_support, 16, 5)
    h = model.backbone(support_x)     # (N,64)
    return h.mean(dim=0, keepdim=True)  # (1,64)

def gate_from_prototype(model, h, p_user, temperature=0.5, top_k=2):
    # TODO: Should take top_k from config...

    # Cosine similarity between h and expert keys, modulated by p_user
    keys = model.expert_keys  # (E,64)
    # Simple affine modulation with user prototype
    mod_keys = keys + 0.25 * p_user  # broadcast (1,64) -> (E,64)
    sim = F.cosine_similarity(h.unsqueeze(1), mod_keys.unsqueeze(0), dim=-1)  # (B,E)
    w = F.softmax(sim / temperature, dim=-1)
    if top_k < w.size(-1):
        topk = torch.topk(w, top_k, dim=-1)
        mask = torch.zeros_like(w).scatter(-1, topk.indices, 1.0)
        w = w * mask
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
    return w  # (B,E)

def build_user_id_map(user_ids_list):
    """
    user_ids_list: list/array of all user IDs from *pretrain train + pretrain val*.
    Returns: id2idx (dict), idx2id (list)
    """
    uniq = np.unique(np.asarray(user_ids_list))
    id2idx = {int(u): i for i, u in enumerate(uniq)}
    idx2id = [int(u) for u in uniq]
    return id2idx, idx2id

def remap_user_ids(user_ids, id2idx):
    """
    user_ids: 1D np.array or torch tensor of original IDs (e.g., 106, 6, 127,...)
    id2idx: mapping dict from original ID -> contiguous index
    Returns torch.LongTensor of remapped IDs (0..N-1)
    """
    if torch.is_tensor(user_ids):
        user_ids_np = user_ids.detach().cpu().numpy()
    else:
        user_ids_np = np.asarray(user_ids)
    mapped = np.array([id2idx[int(u)] for u in user_ids_np], dtype=np.int64)
    return torch.from_numpy(mapped).long()

def filter_by_participant(data_dict, participant_id):
    """
    Filters the feature, labels, and participant_ids arrays in a data_dict
    to only include entries for the specified participant_id.
    """
    # Get the list of participant_ids
    all_ids = data_dict['participant_ids']
    
    # Find indices where participant_id matches
    matching_indices = [i for i, pid in enumerate(all_ids) if pid == participant_id]
    
    # Slice each array using the matching indices
    filtered_features = [data_dict['feature'][i] for i in matching_indices]
    filtered_labels = [data_dict['labels'][i] for i in matching_indices]
    filtered_ids = [all_ids[i] for i in matching_indices]  # Optional, but keeps structure consistent
    
    return {
        'feature': filtered_features,
        'labels': filtered_labels,
        'participant_ids': filtered_ids
    }
