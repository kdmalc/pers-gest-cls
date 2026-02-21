# This is a legacy file. I deleted most of the unused funcs.
## Honestly could probably replace a lot of this code or move it into a more relevant file...

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------
# Freeze/unfreeze helpers
# -----------------
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

def apply_dropout_flag(mod, enabled):
    for m in mod.modules():
        if isinstance(m, nn.Dropout):
            m.p = m.p if enabled else 0.0

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
