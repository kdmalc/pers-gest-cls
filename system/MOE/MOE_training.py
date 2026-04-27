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

def _model_forward_router(model, batch, device, multimodal: bool, config: dict = None):
    """
    Returns: (logits_or_tuple, labels_tensor, batch_size)
    - If `multimodal` is True we call model with named args.
      Otherwise we call model(features) like the old path.
    
    Args:
        config: if provided, used to validate that batch contents match
                use_imu / use_demographics flags. Pass it in — silent
                mismatches between config and batch are a common bug source.
    """
    if isinstance(batch, dict):
        labels = batch['labels']
        if labels is None:
            raise KeyError("Batch missing 'labels' key.")
        labels = _to_device(labels.long(), device)
        B = labels.size(0)

        if multimodal:
            emg = _to_device(batch["emg"], device)

            imu = batch.get("imu", None)
            if config is not None and config.get("use_imu", False):
                if imu is None:
                    raise ValueError(
                        "config has use_imu=True but batch['imu'] is None or missing. "
                        "Check your dataloader — IMU data is not being packed into the batch."
                    )
            imu = _to_device(imu, device) if imu is not None else None

            demo = batch.get("demo", None)
            if config is not None and config.get("use_demographics", False):
                if demo is None:
                    raise ValueError(
                        "config has use_demographics=True but batch['demo'] is None or missing. "
                        "Check your dataloader — demographics are not being packed into the batch."
                    )
            demo = _to_device(demo, device) if demo is not None else None

            outputs = model(
                x_emg=emg,
                x_imu=imu,
                demographics=demo,
            )
            return outputs, labels, B
        else:
            x = _to_device(batch["emg"], device)
            outputs = model(x)
            return outputs, labels, B

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
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience


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


def accuracy(logits, y):
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def gate_balance_loss(avg_usage, eps=1e-8):
    """KL( avg_usage || uniform ). Legacy helper kept for backwards compat."""
    E = avg_usage.numel()
    target = torch.full_like(avg_usage, 1.0 / E)
    return F.kl_div((avg_usage + eps).log(), target, reduction='batchmean')


def make_ce(label_smooth=0.05):
    return nn.CrossEntropyLoss(label_smoothing=label_smooth)


def compute_moe_aux_loss(
    routing_info: dict,
    config: dict,
) -> torch.Tensor:
    """
    Compute the combined MoE auxiliary loss from a routing_info dict.

    Handles both dense and top-k routing automatically:
      - Dense (MOE_top_k is None):
          KL(avg_gate_weights || uniform)
          No importance loss needed — soft weights ARE dispatch weights.
      - Top-k:
          Switch-style f_i * P_i loss   (penalises uneven dispatch counts)
        + Shazeer importance loss CV²   (penalises ordinal ranking dominance)

    The two losses are complementary and must be used together for top-k:
      - f_i * P_i can be satisfied even when one expert dominates the ranking
        if mean soft weights appear roughly uniform.
      - Importance loss catches this by penalising variance in the SUM of soft
        weights per expert, which captures consistent ranking dominance even
        when the mean looks flat.

    Args:
        routing_info : dict with keys:
                         "gate_weights"      (B, E) — hard weights (post top-k)
                         "gate_weights_soft" (B, E) — soft weights (pre top-k)
                       As returned by model.forward(return_routing=True).
        config       : training config dict.  Keys read:
                         "MOE_top_k"             int | None
                         "MOE_aux_coeff"         float  (Switch loss weight)
                         "MOE_importance_coeff"  float  (importance loss weight)

    Returns:
        Scalar aux loss tensor (on the same device as gate_weights).

    Raises:
        KeyError  : if routing_info is missing expected keys.
        ValueError: if routing_info values are None (model not in routing mode).
    """
    # NOTE: Importance loss turned off for now!

    w_hard = routing_info["gate_weights"]
    w_soft = routing_info["gate_weights_soft"]

    if w_hard is None or w_soft is None:
        raise ValueError(
            "routing_info contains None gate weights. "
            "Make sure model.forward() was called with return_routing=True "
            "before passing routing_info to compute_moe_aux_loss()."
        )

    top_k          = config.get("MOE_top_k", None)
    aux_coeff      = config["MOE_aux_coeff"]
    imp_coeff      = config.get("MOE_importance_coeff", aux_coeff)

    if top_k is None:
        # Dense routing: single KL load-balance loss on hard weights
        # (which equal soft weights for dense routing).
        from MOE_encoder import dense_MOE_aux_loss
        return dense_MOE_aux_loss(w_hard, coeff=aux_coeff)
    else:
        # Top-k routing: Switch loss + importance loss.
        from MOE_encoder import topk_MOE_aux_loss, importance_loss
        switch_loss = topk_MOE_aux_loss(w_soft, w_hard, coeff=aux_coeff)
        imp_loss    = importance_loss(w_soft, coeff=imp_coeff)
        return switch_loss + imp_loss