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

============================================================
METRIC GLOSSARY
============================================================
train_supcon_loss   [INHERENT — the actual training objective]
    The SupCon/Siamese loss computed during the forward/backward pass on
    training batches. This IS the signal driving weight updates. Lower = better
    separated embeddings on the training set. SupConLoss returns a single
    scalar — there is no inherent "accuracy" concept; it is a ranking/metric
    loss that shapes embedding geometry, not a classifier.

val_supcon_loss     [INHERENT — direct overfitting signal]
    The same SupCon loss computed on held-out val batches under torch.no_grad()
    (no backprop). Directly comparable to train_supcon_loss. If train loss
    keeps falling but val loss plateaus or rises, the encoder is overfitting.

train_1nn_acc       [PARAMETER-FREE — geometry readout, training set]
    For each sample in a training batch, find its nearest neighbor among all
    other samples in that batch by cosine similarity (no parameters, just a
    matmul). Check if the nearest neighbor shares the same gesture label.
    Measures whether same-class embeddings are locally clustering. Noisy
    because the "dataset" is just one batch, and high values can be achieved
    early even before good generalisation.

val_1nn_acc         [PARAMETER-FREE — geometry readout, val set]
    Episodic 1-shot evaluation: build one prototype per gesture class as the
    mean of 1 support embedding, then assign each query sample to its nearest
    prototype by cosine similarity. No parameters are learned — matching is
    pure nearest-centroid in the frozen embedding space. Measures whether the
    encoder generalises to unseen users. This is what the literature calls
    "few-shot transfer accuracy."

train_linprob_acc   [LIGHTWEIGHT FITTING — downstream task proxy, training set]
    Freeze the encoder backbone (before projection head), extract embeddings
    for all training samples, then fit a small nn.Linear classifier
    (backbone_dim -> num_classes) with cross-entropy for a fixed number of
    epochs. Report final train accuracy. Measures whether backbone features
    are linearly separable for the training distribution.

val_linprob_acc     [LIGHTWEIGHT FITTING — downstream task proxy, val set]
    Same linear classifier evaluated on val-set backbone embeddings.
    The gold-standard contrastive learning eval (used in SimCLR, SupCon papers).
    Unlike 1-NN, this does learn parameters (the linear layer weights), so it
    captures global structure rather than just local neighbourhood geometry.
    Key difference from 1-NN: 1-NN asks "are same-class embeddings close?";
    linear probe asks "is the full embedding space linearly organised by class?"
    Linear probe is more sensitive to class boundary quality; 1-NN is more
    sensitive to intra-class compactness. Both are complementary — a model can
    have high 1-NN acc (tight clusters) but lower linear probe acc (poor
    inter-class margin). Linear probe is run every config['epochs_between_linprob']
    epochs and at the final epoch (None is logged on skipped epochs).
============================================================
"""

import os
import json
import copy
import time
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

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
    lr = config['learning_rate']
    wd = config['weight_decay']
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
            optimizer, mode='max', factor=0.1, patience=6,
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
# TRAINING STEP  (returns supcon loss + in-batch 1-NN acc)
# ============================================================

def train_one_epoch(model, loader, loss_fn, optimizer, config, epoch):
    """
    One full pass over the training dataloader.
    Returns:
        avg_supcon_loss : mean SupCon/Siamese loss across all steps (the training objective)
        avg_1nn_acc     : mean in-batch nearest-neighbour accuracy (parameter-free geometry readout)
    """
    model.train()
    device    = torch.device(config['device'])
    grad_clip = config.get('grad_clip', 1.0)

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    n_steps       = 0

    for step, batch in enumerate(loader):
        emg      = batch['emg'].to(device)
        labels   = batch['labels'].to(device)
        demo     = batch['demo'].to(device) if batch['demo'] is not None else None
        imu      = batch['imu'].to(device)  if batch['imu']  is not None else None
        user_ids = batch['user_ids']

        # Forward through encoder + projection head -> L2-normed embeddings
        z = model(emg, imu=imu, demo=demo)  # (B, D)

        # --- In-batch 1-NN accuracy (parameter-free, no grad needed) ---
        with torch.no_grad():
            sim_matrix = torch.matmul(z, z.T)          # (B, B) cosine sims (z already normed)
            sim_matrix.fill_diagonal_(-1)               # mask self-similarity
            nn_idx  = sim_matrix.argmax(dim=1)          # nearest neighbour index per sample
            correct = (labels[nn_idx] == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

        # --- SupCon / Siamese loss + backprop ---
        loss = loss_fn(z, labels, user_ids=user_ids if config.get('label_hierarchy') else None)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_steps    += 1

    avg_loss = total_loss / max(n_steps, 1)
    avg_acc  = (total_correct / total_samples) * 100
    return avg_loss, avg_acc


# ============================================================
# VAL SUPCON LOSS  (same loss function, no backprop)
# ============================================================

@torch.no_grad()
def compute_val_supcon_loss(model, val_flat_loader, loss_fn, config):
    """
    Runs the SupCon/Siamese loss on held-out val batches without backprop.
    val_flat_loader yields flat balanced batches (same format as train_dl).
    This is the direct overfitting signal: if train_supcon_loss keeps falling
    but val_supcon_loss plateaus or rises, the encoder is overfitting to the
    training users' gesture geometry.

    Returns: mean val SupCon loss (scalar float)
    """
    model.eval()
    device     = torch.device(config['device'])
    total_loss = 0.0
    n_steps    = 0

    for batch in val_flat_loader:
        emg      = batch['emg'].to(device)
        labels   = batch['labels'].to(device)
        demo     = batch['demo'].to(device) if batch['demo'] is not None else None
        imu      = batch['imu'].to(device)  if batch['imu']  is not None else None
        user_ids = batch['user_ids']

        z    = model(emg, imu=imu, demo=demo)
        loss = loss_fn(z, labels, user_ids=user_ids if config.get('label_hierarchy') else None)
        total_loss += loss.item()
        n_steps    += 1

    return total_loss / max(n_steps, 1)


# ============================================================
# VAL 1-NN ACCURACY  (episodic prototyping)
# ============================================================

@torch.no_grad()
def evaluate_1nn_prototyping(model, val_episodic_loader, config):
    """
    Evaluates 1-shot 10-way accuracy on validation episodes.
    For each episode:
      1. Build class prototypes as mean of 1 support embedding per class
      2. Assign each query to its nearest prototype by cosine similarity
      3. Compute accuracy
    Returns mean accuracy across all episodes (0.0-1.0).

    Parameter-free: no weights are learned; matching is pure nearest-centroid
    in the frozen embedding space.
    """
    model.eval()
    device  = torch.device(config['device'])
    all_acc = []

    for batch in val_episodic_loader:
        support  = batch['support']
        query    = batch['query']

        s_emg    = support['emg'].to(device)
        s_labels = support['labels'].to(device)
        s_demo   = support['demo'].to(device) if support['demo'] is not None else None
        s_imu    = support['imu'].to(device)  if support['imu']  is not None else None

        q_emg    = query['emg'].to(device)
        q_labels = query['labels'].to(device)
        q_demo   = query['demo'].to(device) if query['demo'] is not None else None
        q_imu    = query['imu'].to(device)  if query['imu']  is not None else None

        ################################################################################
        ## DEBUGGING 0% VAL ACC: --> It is working now
        ## Check if query labels even exist in support labels
        #s_set = set(s_labels.cpu().numpy())
        #q_set = set(q_labels.cpu().numpy())
        #intersection = s_set.intersection(q_set)
        #if i == 0:
        #    print(f"DEBUG VAL: Support Classes: {s_set}")
        #    print(f"DEBUG VAL: Query Classes: {q_set}")
        #    print(f"DEBUG VAL: Intersection: {intersection}")
        #if not intersection:
        #    print("CRITICAL: Query classes are not present in Support classes.")
        ################################################################################

        prototypes = model.get_prototypes(s_emg, s_labels, s_imu, s_demo)
        pred       = model.predict(q_emg, prototypes, q_imu, q_demo)
        acc        = (pred == q_labels).float().mean().item()
        all_acc.append(acc)

    return float(np.mean(all_acc)) if all_acc else 0.0


# ============================================================
# LINEAR PROBE  (fits a frozen-backbone linear classifier)
# ============================================================

@torch.no_grad()
def _extract_backbone_embeddings(model, loader, config):
    """
    Extract backbone embeddings (before projection head) for all samples in loader.
    Uses model.encode() which returns un-projected backbone features — these
    transfer better to downstream tasks than the projection head output
    (well-established finding from SimCLR / SupCon literature).

    Handles both flat loaders (train batches with 'emg' key) and episodic
    loaders (val batches with 'support'/'query' keys) by checking batch format.

    Returns:
        embeddings : (N, backbone_dim) CPU float32 tensor
        labels     : (N,) CPU int64 tensor
    """
    model.eval()
    device = torch.device(config['device'])
    all_emb, all_lbl = [], []

    for batch in loader:
        if 'support' in batch:
            # Episodic val loader: pool support + query together for embedding extraction
            for split in ('support', 'query'):
                s    = batch[split]
                emg  = s['emg'].to(device)
                lbl  = s['labels']
                demo = s['demo'].to(device) if s['demo'] is not None else None
                imu  = s['imu'].to(device)  if s['imu']  is not None else None
                feat = model.encode(emg, imu=imu, demo=demo)  # (B, backbone_dim)
                all_emb.append(feat.cpu())
                all_lbl.append(lbl)
        else:
            # Flat train loader
            emg  = batch['emg'].to(device)
            lbl  = batch['labels']
            demo = batch['demo'].to(device) if batch['demo'] is not None else None
            imu  = batch['imu'].to(device)  if batch['imu']  is not None else None
            feat = model.encode(emg, imu=imu, demo=demo)
            all_emb.append(feat.cpu())
            all_lbl.append(lbl)

    embeddings = torch.cat(all_emb, dim=0).float()
    labels     = torch.cat(all_lbl, dim=0).long()
    return embeddings, labels


def run_linear_probe(model, train_loader, val_episodic_loader, config):
    """
    Fit a single nn.Linear(backbone_dim -> num_classes) on frozen backbone
    embeddings extracted from the training set, then evaluate on val embeddings.

    The encoder weights are completely frozen throughout — only the linear
    layer is optimised. This is the standard contrastive learning downstream
    evaluation protocol from the SimCLR / SupCon papers.

    Returns:
        train_acc : float, accuracy on training embeddings (0-100)
        val_acc   : float, accuracy on val embeddings (0-100)
    """
    device     = torch.device(config['device'])
    num_epochs = config.get('linprob_epochs', 50)
    lr         = config.get('linprob_lr', 1e-2)

    # Extract frozen embeddings once — single forward pass, no grad, cheap
    train_emb, train_lbl = _extract_backbone_embeddings(model, train_loader, config)
    val_emb,   val_lbl   = _extract_backbone_embeddings(model, val_episodic_loader, config)

    backbone_dim = train_emb.shape[1]
    num_classes  = config['num_classes']

    train_ds = torch.utils.data.TensorDataset(train_emb, train_lbl)
    train_probe_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=256, shuffle=True, drop_last=False
    )

    probe = nn.Linear(backbone_dim, num_classes).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    ce    = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(num_epochs):
        for xb, yb in train_probe_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            ce(probe(xb), yb).backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        train_preds = probe(train_emb.to(device)).argmax(dim=1).cpu()
        val_preds   = probe(val_emb.to(device)).argmax(dim=1).cpu()

    train_acc = (train_preds == train_lbl).float().mean().item() * 100
    val_acc   = (val_preds   == val_lbl  ).float().mean().item() * 100
    return train_acc, val_acc


# ============================================================
# FULL TRAINING LOOP
# ============================================================

def train(config: dict, fold_idx: int):
    device = torch.device(config['device'])
    set_seed(config['seed'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ---- Save dir: pretrained_outputs/ContrastiveNet/ (sibling of runs/) ----
    run_dir        = Path(config['run_dir'])
    pretrained_dir = run_dir.parent / "pretrained_outputs" / "ContrastiveNet"
    os.makedirs(pretrained_dir, exist_ok=True)

    # Name stem: {timestamp}_ContrastiveNet_{attn/lstm}_fold{N}
    arch_tag  = "lstm" if config.get('arch_mode', 'cnn_attn') == 'cnn_lstm' else "attn"
    save_stem = f"{timestamp}_ContrastiveNet_{arch_tag}_fold{fold_idx}"

    linprob_interval = config.get('epochs_between_linprob', 5)

    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx} | arch={config['arch_mode']} | loss={config['loss_mode']}")
    print(f"  Train users : {config['train_PIDs']}")
    print(f"  Val users   : {config['val_PIDs']}")
    print(f"  LinProbe every {linprob_interval} epochs")
    print(f"{'='*60}\n", flush=True)

    # ---- Model ----
    model   = ContrastiveGestureEncoder(config).to(device)
    loss_fn = build_loss(config)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}  |  backbone_dim={model.backbone_dim}  |  embedding_dim={config['embedding_dim']}", flush=True)

    # ---- Data ----
    # Expected API: get_contrastive_dataloaders returns
    #   train_dl          : flat balanced batches (training + val supcon loss source)
    #   val_episodic_dl   : episodic support/query batches (1-NN val acc + linprob val extraction)
    #   val_flat_dl       : flat balanced batches from val users (val supcon loss)
    # NOTE: You will need to add return_val_flat=True support to get_contrastive_dataloaders.
    train_dl, val_episodic_dl, val_flat_dl = get_contrastive_dataloaders(
        config, config['tensor_dict_path'], return_val_flat=True
    )

    # ---- Optimization ----
    optimizer  = build_optimizer(model, config)
    num_epochs = config['num_epochs']
    scheduler  = build_scheduler(optimizer, config, num_epochs)

    base_lr       = config['learning_rate']
    warmup_epochs = config.get('lr_warmup_epochs', 5)

    # ---- Training state ----
    best_val_1nn_acc = -1.0
    best_state       = None
    no_improve       = 0
    es_patience      = config.get('earlystopping_patience', 12)
    es_min_delta     = config.get('earlystopping_min_delta', 0.002)

    # History logs — one entry per epoch.
    # linprob entries are None on skipped epochs for easy epoch-index alignment.
    logs = {
        'train_supcon_loss': [],   # SupCon/Siamese loss, training batches (the training objective)
        'val_supcon_loss':   [],   # Same loss on val batches, no backprop (overfitting signal)
        'train_1nn_acc':     [],   # In-batch nearest-neighbour acc, train set (parameter-free)
        'val_1nn_acc':       [],   # Episodic 1-shot prototype acc, val users (parameter-free)
        'train_linprob_acc': [],   # Linear probe acc on train embeddings (None on skipped epochs)
        'val_linprob_acc':   [],   # Linear probe acc on val embeddings   (None on skipped epochs)
    }

    for epoch in range(num_epochs):
        ep_start = time.time()

        # --- LR warmup ---
        warmup_lr(optimizer, epoch, warmup_epochs, base_lr)

        # --- Train: supcon loss + in-batch 1-NN acc ---
        train_supcon_loss, train_1nn_acc = train_one_epoch(
            model, train_dl, loss_fn, optimizer, config, epoch
        )
        logs['train_supcon_loss'].append(train_supcon_loss)
        logs['train_1nn_acc'].append(train_1nn_acc)

        # --- Val SupCon loss (same objective, no backprop — overfitting signal) ---
        val_supcon_loss = compute_val_supcon_loss(model, val_flat_dl, loss_fn, config)
        logs['val_supcon_loss'].append(val_supcon_loss)

        # --- Val 1-NN episodic accuracy (parameter-free few-shot transfer) ---
        val_1nn_acc = evaluate_1nn_prototyping(model, val_episodic_dl, config)
        logs['val_1nn_acc'].append(val_1nn_acc)

        # --- Linear probe (every linprob_interval epochs and always on final epoch) ---
        is_last_epoch = (epoch == num_epochs - 1)
        run_linprob   = ((epoch + 1) % linprob_interval == 0) or is_last_epoch
        if run_linprob:
            train_lp_acc, val_lp_acc = run_linear_probe(
                model, train_dl, val_episodic_dl, config
            )
            logs['train_linprob_acc'].append(train_lp_acc)
            logs['val_linprob_acc'].append(val_lp_acc)
            linprob_str = f" | LinProbe: train={train_lp_acc:.2f}%  val={val_lp_acc:.2f}%"
        else:
            logs['train_linprob_acc'].append(None)
            logs['val_linprob_acc'].append(None)
            linprob_str = ""

        # --- LR scheduler step ---
        if scheduler is not None and epoch >= warmup_epochs:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_1nn_acc)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        ep_time    = time.time() - ep_start

        # --- Per-epoch print ---
        print(
            f"[Fold {fold_idx} | Epoch {epoch+1:3d}]  "
            f"SupCon: train={train_supcon_loss:.4f}  val={val_supcon_loss:.4f}  |  "
            f"1-NN: train={train_1nn_acc:.2f}%  val={val_1nn_acc*100:.2f}%"
            f"{linprob_str}  |  "
            f"lr={current_lr:.2e}  ({ep_time:.1f}s)",
            flush=True
        )

        # --- Best checkpoint (keyed on val_1nn_acc, consistent with prior behaviour) ---
        if val_1nn_acc > best_val_1nn_acc + es_min_delta:
            best_val_1nn_acc = val_1nn_acc
            best_state       = copy.deepcopy(model.state_dict())
            no_improve       = 0
            best_path        = pretrained_dir / f"{save_stem}_best.pt"
            torch.save({
                'fold_idx':    fold_idx,
                'epoch':       epoch,
                'model_state': best_state,
                'val_1nn_acc': best_val_1nn_acc,
                'config':      config,
                'logs':        logs,
            }, best_path)
            print(f"  ✓ New best val_1nn_acc={best_val_1nn_acc*100:.2f}% -> saved {best_path}", flush=True)
        else:
            no_improve += 1

        # --- Early stopping ---
        if config.get('use_earlystopping', True) and no_improve >= es_patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {es_patience} epochs).", flush=True)
            break

    # ---- Save last model ----
    last_path = pretrained_dir / f"{save_stem}_last.pt"
    torch.save({
        'fold_idx':    fold_idx,
        'epoch':       epoch,
        'model_state': copy.deepcopy(model.state_dict()),
        'val_1nn_acc': logs['val_1nn_acc'][-1] if logs['val_1nn_acc'] else 0.0,
        'config':      config,
        'logs':        logs,
    }, last_path)
    print(f"  ✓ Last model saved  -> {last_path}", flush=True)

    # ---- Save training history JSON ----
    history_path = pretrained_dir / f"{save_stem}_history.json"
    with open(history_path, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"  ✓ Training history  -> {history_path}", flush=True)

    # ---- Final summary ----
    lp_train_vals = [v for v in logs['train_linprob_acc'] if v is not None]
    lp_val_vals   = [v for v in logs['val_linprob_acc']   if v is not None]
    print(f"\n{'='*60}")
    print(f"  [Fold {fold_idx}] Training complete.")
    print(f"  Arch: {config['arch_mode']} ({arch_tag})  |  Loss: {config['loss_mode']}  |  Embed dim: {config['embedding_dim']}")
    print(f"")
    print(f"  METRIC SUMMARY")
    print(f"  SupCon loss  [inherent]         train: {min(logs['train_supcon_loss']):.4f}   val: {min(logs['val_supcon_loss']):.4f}")
    print(f"  1-NN acc     [parameter-free]   train: {max(logs['train_1nn_acc']):.2f}%   val: {max(logs['val_1nn_acc'])*100:.2f}%")
    if lp_val_vals:
        print(f"  Linear probe [lightweight fit]  train: {max(lp_train_vals):.2f}%   val: {max(lp_val_vals):.2f}%")
    print(f"")
    print(f"  KEY: SupCon loss = training objective (drives backprop)")
    print(f"       train/val SupCon loss gap = overfitting signal")
    print(f"       1-NN acc = parameter-free local cluster quality readout")
    print(f"       Linear probe = frozen backbone + fitted linear layer")
    print(f"       1-NN vs LinProbe gap = local compactness vs global separability")
    print(f"{'='*60}\n", flush=True)

    return best_val_1nn_acc, best_state, logs


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',      type=int, default=0,    help="Fold index (0-3)")
    parser.add_argument('--arch_mode', type=str, default=None, help="Override arch_mode")
    parser.add_argument('--loss_mode', type=str, default=None, help="Override loss_mode")
    args = parser.parse_args()

    config = get_config()

    if args.arch_mode: config['arch_mode'] = args.arch_mode
    if args.loss_mode: config['loss_mode'] = args.loss_mode

    with open(config['user_split_json_filepath'], 'r') as f:
        all_splits = json.load(f)

    if len(config["train_PIDs"]) == 0:
        print("Loading train/val/test split from json!")
        apply_fold_to_config(config, all_splits, args.fold)

    best_val_acc, best_state, logs = train(config, args.fold)

    print(f"\nFinal summary:")
    print(f"  arch_mode     : {config['arch_mode']}")
    print(f"  loss_mode     : {config['loss_mode']}")
    print(f"  embedding_dim : {config['embedding_dim']}")
    print(f"  Best val 1-NN : {best_val_acc*100:.2f}%")


if __name__ == '__main__':
    main()