"""
pretrain_trainer.py
===================
Standard supervised pretraining loop for EMG gesture classification.

Features:
  - Cross-entropy loss with optional label smoothing
  - Cosine LR schedule with linear warmup (best default for transformers and RNNs)
  - Early stopping on val loss
  - Best-model checkpointing
  - Per-epoch metrics (loss, top-1 accuracy)
  - Gradient clipping (important for LSTMs)
  - Mixed precision (optional, via torch.amp)

Usage:
    from pretrain_trainer import pretrain
    model, history = pretrain(model, train_dl, val_dl, config)
"""
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 12, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float('inf')
        self.counter   = 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# ─────────────────────────────────────────────────────────────────────────────
# Cosine LR with warmup
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int, min_lr: float = 1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        cosine   = 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
        base_lr  = optimizer.param_groups[0]['initial_lr']
        scaled   = min_lr + (base_lr - min_lr) * cosine
        return scaled / base_lr  
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ─────────────────────────────────────────────────────────────────────────────
# Single epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, optimizer, criterion, device, config, scaler=None, is_train=True, epoch=1):
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    use_imu = config.get('use_imu', False)
    clip_val = float(config.get('grad_clip', 1.0))
    use_amp  = config.get('use_amp', False) and torch.cuda.is_available()

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch_idx, batch in enumerate(loader):
            emg    = batch['emg'].to(device)      # (B, C, T)
            labels = batch['labels'].to(device)   # (B,)
            imu    = batch['imu'].to(device) if (use_imu and batch['imu'] is not None) else None

            # --- DIAGNOSTICS: Print Data Stats on the very first batch of training ---
            if is_train and epoch == 1 and batch_idx == 0:
                print(f"\n{'='*40}")
                print(f" DIAGNOSTICS: EPOCH 1, BATCH 0")
                print(f"{'='*40}")
                print(f"EMG - Shape: {emg.shape}")
                # Print first 5 values of the first channel, first batch item
                print(f"EMG - First 5 vals: {emg[0, 0, :5].tolist()}")
                print(f"EMG - Mean: {emg.mean().item():.6f} | Std: {emg.std().item():.6f} | Norm: {torch.norm(emg).item():.6f}")
                
                if imu is not None:
                    print(f"IMU - Mean: {imu.mean().item():.6f} | Std: {imu.std().item():.6f}")
                
                print(f"Labels - First 5: {labels[:5].tolist()}")
                print(f"Labels - Min: {labels.min().item()} | Max: {labels.max().item()} (Ensure these are 0-indexed!)")
                print(f"{'='*40}\n")

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                logits = model(emg, imu)
                loss   = criterion(logits, labels)

            if is_train and batch_idx == 0 and epoch <= 3:
                with torch.no_grad():
                    feat_debug, _ = model.backbone(emg, imu)
                    print("------ LATENT FEATURES CHECK ------")
                    print(f"[Ep{epoch}] feat: mean={feat_debug.mean():.3f} std={feat_debug.std():.3f} "
                        f"logits: mean={logits.mean():.3f} std={logits.std():.3f} "
                        f"max_logit={logits.max():.3f}")

            if is_train:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer) # Unscale before clipping/checking gradients
                    
                    # --- DIAGNOSTICS: Check Gradient Norms (Once per epoch, on batch 0) ---
                    if batch_idx == 0:
                        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2).item()
                        print(f"[Epoch {epoch}] Initial Grad Norm: {total_norm:.4f}")

                    nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    
                    # --- DIAGNOSTICS: Check Gradient Norms (Once per epoch, on batch 0) ---
                    if batch_idx == 0:
                        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2).item()
                        print(f"[Epoch {epoch}] Initial Grad Norm: {total_norm:.4f}")
                    #if batch_idx == 0 and is_train:
                    #    print("\n--- PER-LAYER GRAD NORMS ---")
                    #    for name, p in model.named_parameters():
                    #        if p.grad is not None:
                    #            print(f"  {name:<50s}  grad_norm={p.grad.norm():.4f}  param_norm={p.data.norm():.4f}")
                    #        else:
                    #            print(f"  {name:<50s}  NO GRAD")

                    nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    optimizer.step()

                # --- DIAGNOSTICS: Check for model collapse (Are we predicting the same class?) ---
                if batch_idx == 0:
                    preds = logits.argmax(dim=1)
                    unique_preds = torch.unique(preds).tolist()
                    print(f"[Epoch {epoch}] Unique predictions in batch 0: {unique_preds}")

            B = emg.size(0)
            total_loss    += loss.item() * B
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += B

    return total_loss / total_samples, total_correct / total_samples

# ─────────────────────────────────────────────────────────────────────────────
# Main pretrain function
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(model, train_dl, val_dl, config: dict, save_path: str = None):
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    # ── Loss ────────────────────────────────────────────────────────────────
    label_smooth = float(config.get('label_smooth', 0.1))
    criterion    = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # ── Optimizer ───────────────────────────────────────────────────────────
    opt_name = config.get('optimizer', 'adamw').lower()
    lr       = float(config.get('learning_rate', 1e-3))
    wd       = float(config.get('weight_decay', 1e-4))

    if opt_name == 'adamw':
        decay_params     = [p for n, p in model.named_parameters() if 'bias' not in n and 'norm' not in n.lower() and p.requires_grad]
        no_decay_params  = [p for n, p in model.named_parameters() if ('bias' in n or 'norm' in n.lower()) and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': decay_params,    'weight_decay': wd},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=lr)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    for pg in optimizer.param_groups:
        pg.setdefault('initial_lr', lr)

    # ── Scheduler ───────────────────────────────────────────────────────────
    num_epochs    = int(config.get('num_epochs', 100))
    warmup_epochs = int(config.get('warmup_epochs', 5))
    use_scheduler = config.get('use_scheduler', True)
    scheduler     = build_scheduler(optimizer, num_epochs, warmup_epochs) if use_scheduler else None

    # ── Early stopping ───────────────────────────────────────────────────────
    use_es = bool(config.get('use_early_stopping', True))
    es     = EarlyStopping(
        patience  = int(config.get('es_patience', 12)),
        min_delta = float(config.get('es_min_delta', 1e-4)),
    ) if use_es else None

    # ── AMP ─────────────────────────────────────────────────────────────────
    use_amp = config.get('use_amp', False) and torch.cuda.is_available()
    scaler  = GradScaler() if use_amp else None

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_state    = None
    best_epoch    = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
    }

    print(f"\n{'='*60}")
    print(f"Pretraining: {config.get('model_type', 'Model')}")
    print(f"  Device={device} | Epochs={num_epochs} | LR={lr} | BS={config.get('batch_size',64)}")
    print(f"  Optimizer={opt_name} | WD={wd} | LabelSmooth={label_smooth}")
    print(f"  GradClip={config.get('grad_clip',1.0)} | AMP={use_amp}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Train (Note: passing the epoch number here now)
        tr_loss, tr_acc = _run_epoch(
            model, train_dl, optimizer, criterion, device, config,
            scaler=scaler, is_train=True, epoch=epoch
        )

        if scheduler:
            scheduler.step()

        # Val (Runs without optimizer to check test-time performance)
        va_loss, va_acc = _run_epoch(
            model, val_dl, None, criterion, device, config,
            is_train=False, epoch=epoch
        )

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]['lr']
        print(f"[Ep {epoch:3d}/{num_epochs}] "
              f"train loss={tr_loss:.4f} acc={tr_acc*100:.1f}% | "
              f"val loss={va_loss:.4f} acc={va_acc*100:.1f}% | "
              f"lr={cur_lr:.2e} | {elapsed:.1f}s")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = copy.deepcopy(model.state_dict())
            best_epoch    = epoch
            if save_path:
                torch.save(best_state, save_path)
                print(f"  ✓ Saved best model → {save_path}")

        if es is not None and es(va_loss):
            print(f"\n[EarlyStopping] Stopped at epoch {epoch} (patience={es.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n[Done] Loaded best model from epoch {best_epoch} | val_loss={best_val_loss:.4f}")

    history['best_epoch']    = best_epoch
    history['best_val_loss'] = best_val_loss
    history['best_val_acc']  = history['val_acc'][best_epoch - 1] if best_epoch > 0 else 0.0

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Default config reference
# ─────────────────────────────────────────────────────────────────────────────

PRETRAIN_DEFAULT_CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────
    "train_PIDs":   ["P102","P114"],        
            #["P102","P114","P119","P005","P107","P126","P132","P112",
            #"P103","P125","P127","P010","P128","P111","P118",
            #"P124","P110","P116","P108","P104","P122","P131","P106","P115"],
    "val_PIDs":     ["P102","P114"],        
            #["P011","P006","P105","P109"],
    "train_gesture_range":  list(range(1, 11)),   # gestures 1-10
    "valtest_gesture_range":list(range(1, 11)),
    "use_imu":              False,
    "batch_size":           64,
    "num_workers":          1,
    # ── Augmentation ──────────────────────────────────────────────────────
    "augment":              False,
    "noise_std":            0.05,  # TODO: Is this way too big?? What are our EMG magnitudes?
    "max_shift":            4,
    "ch_drop_prob":         0.05,
    # ── Training ──────────────────────────────────────────────────────────
    "num_epochs":           50,
    "learning_rate":        1e-3,
    "optimizer":            "adamw",
    "weight_decay":         1e-4,
    "label_smooth":         0.0,
    "grad_clip":            10.0,
    "warmup_epochs":        5,
    "use_scheduler":        True,
    "use_amp":              False,
    # ── Early stopping ────────────────────────────────────────────────────
    "use_early_stopping":   True,
    "es_patience":          12,
    "es_min_delta":         1e-4,
    # ── Device ────────────────────────────────────────────────────────────
    "device":               "cuda",
}