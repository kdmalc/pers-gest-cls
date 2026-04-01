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
  - MoE auxiliary load-balancing loss (when config["use_moe"]=True)
  - Periodic routing analysis dumps (when config["use_moe"]=True)

Usage:
    from pretrain_trainer import pretrain
    model, history = pretrain(model, train_dl, val_dl, config)

MoE-specific config keys:
    use_moe        : bool  — activates aux loss + routing logging
    moe_aux_coeff  : float — scale for Switch Transformer load-balancing loss (default 1e-2)
    moe_log_every  : int   — dump routing analysis every N epochs (default 5; 0 = never)
    moe_plot_dir   : str   — directory to save routing heatmaps (default None = no plots)
"""
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import sys
import os
# This finds the 'system' directory (one level up from 'pretraining') and adds it to the search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MOE.MOE_encoder import dense_MOE_aux_loss as _dense_MOE_aux_loss
from MOE.MOE_encoder import topk_MOE_aux_loss as _topk_MOE_aux_loss
# TODO: Need to define _moe_aux_loss throughout now
from MOE.MOE_analysis import RoutingCollector, RoutingAnalyzer

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

def _run_epoch(model, loader, optimizer, criterion, device, config, scaler=None, is_train=True, epoch=1,
               routing_collector=None):
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    total_aux_loss = 0.0                          # tracked separately for diagnostics
    use_imu     = config.get('use_imu', False)
    clip_val    = float(config.get('grad_clip', 1.0))
    use_amp     = config.get('use_amp', False) and torch.cuda.is_available()
    use_moe     = config.get('use_moe', False)
    aux_coeff   = float(config.get('moe_aux_coeff', 1e-2))

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch_idx, batch in enumerate(loader):
            emg    = batch['emg'].to(device)      # (B, C, T)
            labels = batch['labels'].to(device)   # (B,)
            imu    = batch['imu'].to(device) if (use_imu and batch.get('imu') is not None) else None

            # --- DIAGNOSTICS: Print Data Stats on the very first batch of training ---
            if is_train and epoch == 1 and batch_idx == 0:
                print(f"\n{'='*40}")
                print(f" DIAGNOSTICS: EPOCH 1, BATCH 0")
                print(f"{'='*40}")
                print(f"EMG - Shape: {emg.shape}")
                print(f"EMG - First 5 vals: {emg[0, 0, :5].tolist()}")
                print(f"EMG - Mean: {emg.mean().item():.6f} | Std: {emg.std().item():.6f} | Norm: {torch.norm(emg).item():.6f}")
                if imu is not None:
                    print(f"IMU - Mean: {imu.mean().item():.6f} | Std: {imu.std().item():.6f}")
                print(f"Labels - First 5: {labels[:5].tolist()}")
                print(f"Labels - Min: {labels.min().item()} | Max: {labels.max().item()} (Ensure these are 0-indexed!)")
                if use_moe:
                    print(f"MoE - aux_coeff={aux_coeff:.4f}  num_experts={config.get('num_experts', '?')}"
                          f"  placement={config.get('moe_placement', '?')}")
                print(f"{'='*40}\n")

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            # ── Forward pass ────────────────────────────────────────────────
            # MoE models can return (logits, routing_info) when return_routing=True.
            # We collect routing info even during training so the collector works.
            collect_routing = (routing_collector is not None) and use_moe
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                if use_moe:
                    out = model(emg, imu, return_routing=True)
                    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                        logits, routing_info = out
                    else:
                        # Model doesn't support return_routing (e.g. TST, non-MoE path)
                        logits = out[0] if isinstance(out, tuple) else out
                        routing_info = None
                else:
                    logits = model(emg, imu)
                    routing_info = None

                main_loss = criterion(logits, labels)

                # Auxiliary load-balancing loss (MoE only)
                if use_moe and routing_info is not None and aux_coeff > 0 and is_train:
                    gate_w = routing_info.get('gate_weights')
                    if gate_w is not None:
                        if config['top_k'] is None or config['top_k']==config['num_experts']:  # dense/soft routing
                            aux = _dense_MOE_aux_loss(gate_w, coeff=aux_coeff)
                        else:  # top_k
                            gate_w_soft = routing_info.get('gate_weights_soft', gate_w)
                            aux = _topk_MOE_aux_loss(gate_w_soft, gate_w, coeff=aux_coeff)
                        loss = main_loss + aux
                        total_aux_loss += aux.item() * emg.size(0)
                    else:
                        loss = main_loss
                else:
                    loss = main_loss

            # ── Routing collection (val or train) ───────────────────────────
            if collect_routing and routing_info is not None:
                gate_w = routing_info.get('gate_weights')
                if gate_w is not None:
                    # TODO: What? I thought batch had key user_ids, not pids... ....
                    pids  = batch.get('pid', batch.get('pids', ['?'] * emg.size(0)))
                    demo  = batch.get('demographics')
                    routing_collector.add(
                        gate_weights   = gate_w.detach().cpu(),
                        gesture_labels = labels.cpu(),
                        pids           = pids,
                        demographics   = demo.cpu() if demo is not None else None,
                    )

            # ── Latent feature diagnostics (first few epochs, first batch) ──
            if is_train and batch_idx == 0 and epoch <= 3:
                with torch.no_grad():
                    feat_debug, _ = model.backbone(emg, imu)
                    print("------ LATENT FEATURES CHECK ------")
                    print(f"[Ep{epoch}] feat: mean={feat_debug.mean():.3f} std={feat_debug.std():.3f} "
                          f"logits: mean={logits.mean():.3f} std={logits.std():.3f} "
                          f"max_logit={logits.max():.3f}")
                    if use_moe and routing_info is not None:
                        gw = routing_info.get('gate_weights')
                        if gw is not None:
                            print(f"[Ep{epoch}] Gate weights mean per expert: "
                                  f"{gw.mean(0).tolist()}")

            # ── Backward pass ───────────────────────────────────────────────
            if is_train:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    if batch_idx == 0:
                        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2).item()
                        print(f"[Epoch {epoch}] Initial Grad Norm: {total_norm:.4f}")

                    nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    if batch_idx == 0:
                        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2).item()
                        print(f"[Epoch {epoch}] Initial Grad Norm: {total_norm:.4f}")

                    nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    optimizer.step()

                if batch_idx == 0:
                    preds = logits.argmax(dim=1)
                    unique_preds = torch.unique(preds).tolist()
                    print(f"[Epoch {epoch}] Unique predictions in batch 0: {unique_preds}")

            B = emg.size(0)
            total_loss    += loss.item() * B
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += B

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    avg_aux  = total_aux_loss / max(total_samples, 1)
    return avg_loss, avg_acc, avg_aux

# ─────────────────────────────────────────────────────────────────────────────
# Main pretrain function
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(model, train_dl, val_dl, config: dict, save_path: str = None):
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    use_moe   = config.get('use_moe', False)
    moe_log_every = int(config.get('moe_log_every', 5))    # 0 = never
    moe_plot_dir  = config.get('moe_plot_dir', None)
    num_experts   = int(config.get('num_experts', 4))

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

    # ── MoE routing collectors (one for train, one for val) ──────────────────
    model_name = config.get('model_type', 'Model')
    train_routing_collector = (
        RoutingCollector(num_experts=num_experts, model_name=f"{model_name}_train")
        if use_moe else None
    )
    val_routing_collector = (
        RoutingCollector(num_experts=num_experts, model_name=f"{model_name}_val")
        if use_moe else None
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_state    = None
    best_epoch    = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'train_aux_loss': [], 'val_aux_loss': [],
        'routing_reports': [],   # one dict per moe_log_every epochs
    }

    print(f"\n{'='*60}")
    print(f"Pretraining: {model_name}")
    print(f"  Device={device} | Epochs={num_epochs} | LR={lr} | BS={config.get('batch_size',64)}")
    print(f"  Optimizer={opt_name} | WD={wd} | LabelSmooth={label_smooth}")
    print(f"  GradClip={config.get('grad_clip',1.0)} | AMP={use_amp}")
    if use_moe:
        print(f"  MoE: placement={config.get('moe_placement','?')} | "
              f"E={num_experts} | aux_coeff={config.get('moe_aux_coeff',1e-2):.4f} | "
              f"log_every={moe_log_every}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Reset routing collectors at the start of each epoch
        if train_routing_collector is not None:
            train_routing_collector.reset()
        if val_routing_collector is not None:
            val_routing_collector.reset()

        # Train
        tr_loss, tr_acc, tr_aux = _run_epoch(
            model, train_dl, optimizer, criterion, device, config,
            scaler=scaler, is_train=True, epoch=epoch,
            routing_collector=train_routing_collector,
        )

        if scheduler:
            scheduler.step()

        # Val
        va_loss, va_acc, va_aux = _run_epoch(
            model, val_dl, None, criterion, device, config,
            is_train=False, epoch=epoch,
            routing_collector=val_routing_collector,
        )

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        history['train_aux_loss'].append(tr_aux)
        history['val_aux_loss'].append(va_aux)

        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]['lr']
        aux_str = f" | aux={tr_aux:.4f}" if use_moe and tr_aux > 0 else ""
        print(f"[Ep {epoch:3d}/{num_epochs}] "
              f"train loss={tr_loss:.4f} acc={tr_acc*100:.1f}%{aux_str} | "
              f"val loss={va_loss:.4f} acc={va_acc*100:.1f}% | "
              f"lr={cur_lr:.2e} | {elapsed:.1f}s")

        # ── MoE routing analysis ─────────────────────────────────────────────
        if use_moe and moe_log_every > 0 and epoch % moe_log_every == 0:
            _run_routing_analysis_epoch(
                epoch, val_routing_collector, train_routing_collector,
                config, moe_plot_dir, history,
            )

        # ── Checkpoint ──────────────────────────────────────────────────────
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = copy.deepcopy(model.state_dict())
            best_epoch    = epoch
            if save_path:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "config":           config,
                        "epoch":            epoch,
                        "val_loss":         best_val_loss,
                        "checkpoint_type":  "best",
                    },
                    save_path,
                )
                print(f"  ✓ Saved best model → {save_path}")

        if es is not None and es(va_loss):
            print(f"\n[EarlyStopping] Stopped at epoch {epoch} (patience={es.patience})")
            break

    # ── Final routing analysis ────────────────────────────────────────────────
    if use_moe and moe_log_every > 0:
        print("\n[MoE] Running final routing analysis on val set...")
        _run_routing_analysis_epoch(
            epoch, val_routing_collector, train_routing_collector,
            config, moe_plot_dir, history,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n[Done] Loaded best model from epoch {best_epoch} | val_loss={best_val_loss:.4f}")

    history['best_epoch']    = best_epoch
    history['best_val_loss'] = best_val_loss
    history['best_val_acc']  = history['val_acc'][best_epoch - 1] if best_epoch > 0 else 0.0

    return model, history


def _run_routing_analysis_epoch(epoch, val_collector, train_collector, config, plot_dir, history):
    """Helper: finalize collectors, print report, optionally save plots."""
    demo_labels = config.get('demo_dim_labels', None)

    for collector, tag in [(val_collector, 'val'), (train_collector, 'train')]:
        if collector is None:
            continue
        try:
            record   = collector.finalize()
            analyzer = RoutingAnalyzer(record)
            report   = analyzer.full_report(print_report=(tag == 'val'), demo_dim_labels=demo_labels)
            report['epoch'] = epoch
            report['split'] = tag
            history['routing_reports'].append(report)

            if plot_dir is not None and tag == 'val':
                import os
                epoch_plot_dir = os.path.join(plot_dir, f"epoch_{epoch:03d}")
                analyzer.plot_all(save_dir=epoch_plot_dir)
        except Exception as e:
            print(f"[MoE routing analysis] Warning: {e}")