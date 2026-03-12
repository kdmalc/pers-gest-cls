"""
contrastive_train.py

Training loop for contrastive gesture recognition.

Usage:
    python contrastive_train.py --fold 0
    python contrastive_train.py --fold 0 --loss_mode siamese --arch_mode cnn_lstm

Architecture overview:
    Train : Flat SupCon/Siamese loss on balanced batches (no episodes)
    Val   : 1-shot prototype accuracy on held-out users (no retraining)
    Test  : Same as Val (run after selecting best checkpoint via val acc)
"""

import os
import json
import copy
import math
import time
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from contrastive_config import get_config, apply_fold_to_config
from contrastive_encoder import ContrastiveGestureEncoder
from contrastive_losses import build_loss
from contrastive_data_pipeline import get_contrastive_dataloaders


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# OPTIMIZER + SCHEDULER
# ============================================================

def build_optimizer(model: nn.Module, config: dict):
    opt_name = config.get('optimizer', 'adamw').lower()
    lr  = config['learning_rate']
    wd  = config['weight_decay']

    if opt_name == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'sgd':
        return SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, config: dict, num_epochs: int):
    sched_name = config.get('lr_scheduler', 'cosine')
    if sched_name == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - config.get('lr_warmup_epochs', 5),
            eta_min=config.get('lr_min', 1e-6),
        )
    elif sched_name == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=6,
            min_lr=config.get('lr_min', 1e-6),
        )
    else:
        return None


def warmup_lr(optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    """Linear LR warmup."""
    if epoch < warmup_epochs:
        scale = (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * scale


# ============================================================
# TRAINING STEP
# ============================================================

def train_one_epoch(model, loader, loss_fn, optimizer, config, epoch):
    model.train()
    device      = torch.device(config['device'])
    grad_clip   = config.get('grad_clip', 1.0)
    log_interval = config.get('log_interval', 50)
    use_hierarchy = config.get('label_hierarchy', False)

    total_loss  = 0.0
    n_steps     = 0

    for step, batch in enumerate(loader):
        emg      = batch['emg'].to(device)              # (B, C, T)
        labels   = batch['labels'].to(device)            # (B,)
        demo     = batch['demo'].to(device) if batch['demo'] is not None else None
        imu      = batch['imu'].to(device) if batch['imu'] is not None else None
        user_ids = batch['user_ids'].to(device)

        # Forward
        z = model(emg, imu=imu, demo=demo)              # (B, D) L2-normed

        # Loss
        loss = loss_fn(z, labels, user_ids=user_ids if use_hierarchy else None)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_steps    += 1

        if config.get('verbose', False) and (step + 1) % log_interval == 0:
            print(f"  [Epoch {epoch+1} | Step {step+1}] loss={loss.item():.4f}", flush=True)

    return total_loss / max(n_steps, 1)


# ============================================================
# VALIDATION (1-SHOT PROTOTYPING)
# ============================================================

@torch.no_grad()
def evaluate_prototyping(model, val_loader, config):
    """
    Evaluates 1-shot 10-way accuracy on validation episodes.
    For each episode:
      1. Build prototypes from support set (1 sample per class)
      2. Predict query labels via nearest cosine neighbor
      3. Compute accuracy
    Returns mean accuracy across all episodes.
    """
    model.eval()
    device  = torch.device(config['device'])
    all_acc = []

    for batch in val_loader:
        support = batch['support']
        query   = batch['query']

        s_emg    = support['emg'].to(device)
        s_labels = support['labels'].to(device)
        s_demo   = support['demo'].to(device) if support['demo'] is not None else None
        s_imu    = support['imu'].to(device) if support['imu'] is not None else None

        q_emg    = query['emg'].to(device)
        q_labels = query['labels'].to(device)
        q_demo   = query['demo'].to(device) if query['demo'] is not None else None
        q_imu    = query['imu'].to(device) if query['imu'] is not None else None

        # Build prototypes
        prototypes = model.get_prototypes(s_emg, s_labels, s_imu, s_demo)

        # Predict
        pred = model.predict(q_emg, prototypes, q_imu, q_demo)

        acc = (pred == q_labels).float().mean().item()
        all_acc.append(acc)

    return np.mean(all_acc) if all_acc else 0.0


# ============================================================
# FULL TRAINING LOOP
# ============================================================

def train(config: dict, fold_idx: int):
    device = torch.device(config['device'])
    set_seed(config['seed'])

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx} | arch={config['arch_mode']} | loss={config['loss_mode']}")
    print(f"  Train users: {config['train_PIDs']}")
    print(f"  Val users  : {config['val_PIDs']}")
    print(f"{'='*60}\n", flush=True)

    # ---- Model ----
    model   = ContrastiveGestureEncoder(config).to(device)
    loss_fn = build_loss(config)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}  |  embedding_dim={config['embedding_dim']}", flush=True)

    # ---- Data ----
    train_dl, val_dl = get_contrastive_dataloaders(config, config['tensor_dict_path'])

    # ---- Optimization ----
    optimizer = build_optimizer(model, config)
    num_epochs = config['num_epochs']
    scheduler  = build_scheduler(optimizer, config, num_epochs)

    base_lr        = config['learning_rate']
    warmup_epochs  = config.get('lr_warmup_epochs', 5)

    # ---- Training state ----
    best_val_acc  = -1.0
    best_state    = None
    no_improve    = 0
    es_patience   = config.get('earlystopping_patience', 12)
    es_min_delta  = config.get('earlystopping_min_delta', 0.002)

    train_loss_log = []
    val_acc_log    = []

    for epoch in range(num_epochs):
        ep_start = time.time()

        # LR warmup
        warmup_lr(optimizer, epoch, warmup_epochs, base_lr)

        # Train
        train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer, config, epoch)
        train_loss_log.append(train_loss)

        # Val
        val_acc = evaluate_prototyping(model, val_dl, config)
        val_acc_log.append(val_acc)

        # Scheduler step
        if scheduler is not None and epoch >= warmup_epochs:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        ep_time    = time.time() - ep_start

        print(f"[Fold {fold_idx} | Epoch {epoch+1:3d}/{num_epochs}] "
              f"loss={train_loss:.4f} | val_acc={val_acc*100:.2f}% | "
              f"lr={current_lr:.2e} | {ep_time:.1f}s", flush=True)

        # Best checkpoint
        if val_acc > best_val_acc + es_min_delta:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
            ckpt_path    = os.path.join(
                config['checkpoint_dir'],
                f"fold{fold_idx}_{timestamp}_best.pt"
            )
            torch.save({
                'fold_idx':      fold_idx,
                'epoch':         epoch,
                'model_state':   best_state,
                'val_acc':       best_val_acc,
                'config':        config,
                'train_loss_log': train_loss_log,
                'val_acc_log':   val_acc_log,
            }, ckpt_path)
            print(f"  ✓ New best val_acc={best_val_acc*100:.2f}% → saved {ckpt_path}", flush=True)
        else:
            no_improve += 1

        # Early stopping
        if config.get('use_earlystopping', True) and no_improve >= es_patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {es_patience} epochs).", flush=True)
            break

    print(f"\n[Fold {fold_idx}] Training complete. Best val acc = {best_val_acc*100:.2f}%")
    return best_val_acc, best_state, train_loss_log, val_acc_log


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',      type=int,   default=0,        help="Fold index (0-3)")
    parser.add_argument('--arch_mode', type=str,   default=None,     help="Override arch_mode")
    parser.add_argument('--loss_mode', type=str,   default=None,     help="Override loss_mode")
    args = parser.parse_args()

    config = get_config()

    # CLI overrides
    if args.arch_mode: config['arch_mode'] = args.arch_mode
    if args.loss_mode: config['loss_mode'] = args.loss_mode

    # Load user splits
    with open(config['user_split_json_filepath'], 'r') as f:
        all_splits = json.load(f)

    apply_fold_to_config(config, all_splits, args.fold)

    best_val_acc, best_state, train_loss_log, val_acc_log = train(config, args.fold)

    print(f"\nFinal summary:")
    print(f"  arch_mode  : {config['arch_mode']}")
    print(f"  loss_mode  : {config['loss_mode']}")
    print(f"  embedding_dim: {config['embedding_dim']}")
    print(f"  Best val acc : {best_val_acc*100:.2f}%")


if __name__ == '__main__':
    main()
