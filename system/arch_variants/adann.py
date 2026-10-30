"""
adann.py
========
P3 (docs/specs/SPEC_architecture_exploration.md): supervised pretraining with a gradient-reversal
user-ID head, i.e. every training user is a domain and the encoder is trained to make users
indistinguishable (multi-source DANN, Ganin et al. 2016, as used for cross-user EMG by
Cote-Allard et al.). Then the usual supervised fine-tune on the support set (A2's evaluation path).

What this is NOT: Cote-Allard's full ADANN also keeps domain-specific normalisation layers and
pairs domains in the discriminator. This is the plain multi-domain DANN core, named "ADANN-style"
in the docs; say so in the paper if it is reported.

Hypothesis it tests: labels here are user-defined, so user identity is part of what a gesture
looks like. Removing it should HURT (prediction), which would itself be an informative result.

Interface: adann_pretrain(model, train_dl, val_dl, config) has pretrain_trainer.pretrain's
signature and history keys ('val_acc', 'val_loss', 'train_loss', ...), and returns the plain
classification model (the discriminator is discarded), so the existing fine-tune evaluation runs on
it unchanged. The train loader must carry "users" (get_pretrain_dataloaders adds them when
config has "adann_lambda").

Config keys: adann_lambda (max weight of the reversed user loss), adann_disc_hidden.
"""

from __future__ import annotations

import copy
import math
import time

import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam: float):
    return _GradReverse.apply(x, lam)


def dann_lambda(progress: float, lam_max: float) -> float:
    """Ganin et al. schedule: 0 at the start, -> lam_max, so the discriminator is useful before it bites."""
    return lam_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def _features_and_logits(model, emg, imu):
    out = model.backbone(emg, imu)
    feat = out[0]
    return feat, model.head(feat)


def adann_pretrain(model, train_dl, val_dl, config: dict, save_path: str = None):
    device = config["device"]
    model = model.to(device)
    n_users = len(config["train_PIDs"])
    disc = nn.Sequential(nn.Linear(model.feat_dim, config["adann_disc_hidden"]), nn.GELU(),
                         nn.Linear(config["adann_disc_hidden"], n_users)).to(device)
    params = list(model.parameters()) + list(disc.parameters())
    opt = torch.optim.Adam(params, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    cls_loss = nn.CrossEntropyLoss(label_smoothing=float(config["label_smooth"]))
    dom_loss = nn.CrossEntropyLoss()
    lam_max = float(config["adann_lambda"])
    clip = float(config["gradient_clip_max_norm"])
    num_epochs = int(config["num_epochs"])
    patience = int(config["earlystopping_patience"])
    min_delta = float(config["earlystopping_min_delta"])

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "train_dom_acc": [],
            "train_aux_loss": [], "val_aux_loss": [], "routing_reports": []}
    best_val, best_state, bad = float("inf"), None, 0
    n_steps = num_epochs * max(len(train_dl), 1)
    step = 0
    for ep in range(num_epochs):
        t0 = time.time()
        model.train(); disc.train()
        tl = tc = tn = dc = 0.0
        for batch in train_dl:
            if "users" not in batch:
                raise KeyError("ADANN needs per-sample user IDs: set config['adann_lambda'] before "
                               "building the dataloaders (get_pretrain_dataloaders)")
            emg, y, u = batch["emg"].to(device), batch["labels"].to(device), batch["users"].to(device)
            imu = batch["imu"].to(device) if batch["imu"] is not None else None
            feat, logits = _features_and_logits(model, emg, imu)
            lam = dann_lambda(step / max(n_steps, 1), lam_max)
            d_logits = disc(grad_reverse(feat, lam))
            loss = cls_loss(logits, y) + dom_loss(d_logits, u)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            step += 1
            tl += float(loss.item()) * y.numel()
            tc += float((logits.argmax(-1) == y).sum().item())
            dc += float((d_logits.argmax(-1) == u).sum().item())
            tn += y.numel()
        hist["train_loss"].append(tl / max(tn, 1))
        hist["train_acc"].append(tc / max(tn, 1))
        hist["train_dom_acc"].append(dc / max(tn, 1))

        model.eval()
        vl = vc = vn = 0.0
        with torch.no_grad():
            for batch in val_dl:
                emg, y = batch["emg"].to(device), batch["labels"].to(device)
                imu = batch["imu"].to(device) if batch["imu"] is not None else None
                _, logits = _features_and_logits(model, emg, imu)
                vl += float(cls_loss(logits, y).item()) * y.numel()
                vc += float((logits.argmax(-1) == y).sum().item())
                vn += y.numel()
        hist["val_loss"].append(vl / max(vn, 1))
        hist["val_acc"].append(vc / max(vn, 1))
        print(f"[ADANN] epoch {ep+1}/{num_epochs}: train loss {hist['train_loss'][-1]:.4f} "
              f"acc {hist['train_acc'][-1]*100:.1f}% | user-disc acc {hist['train_dom_acc'][-1]*100:.1f}% "
              f"(chance {100/n_users:.1f}%) | val loss {hist['val_loss'][-1]:.4f} "
              f"acc {hist['val_acc'][-1]*100:.1f}% | lambda {lam:.3f} ({time.time()-t0:.0f}s)")
        if hist["val_loss"][-1] < best_val - min_delta:
            best_val, best_state, bad = hist["val_loss"][-1], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[ADANN] early stopping at epoch {ep+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist
