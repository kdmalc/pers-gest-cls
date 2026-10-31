"""
metric_learners.py
==================
Episodic metric-based few-shot learners: trained on the same cached 10-way 1-shot episodes as
MAML, but with no inner loop. The support set is embedded and the query is classified against it
in one forward pass. docs/specs/SPEC_architecture_exploration.md section 2D (P1 and the
"for edification" alternatives).

  config["metric_kind"]
    "proto"     ProtoNet (Snell et al. 2017): negative squared Euclidean distance to class means.  P1.
    "matching"  Matching Network (Vinyals et al. 2016) without full-context embeddings: cosine
                attention over support items with a learned scale. At 1-shot it reduces to scaled
                cosine nearest-prototype; FCE (the LSTM over the support set) is not implemented.
    "relation"  Relation Network (Sung et al. 2018): an MLP scores [query, class mean] pairs. Trained
                with cross-entropy over the scores instead of the original MSE-to-one-hot (the
                common modern variant; it trains more stably).
    "ridge"     R2-D2 (Bertinetto et al. ICLR 2019): closed-form ridge regression on the support
                embeddings, differentiable, with a learned lambda, scale and bias. The cheapest
                "adapt a linear head in closed form" baseline, between ProtoNet and MAML.
    "feat"      FEAT (Ye et al. CVPR 2020): the class prototypes attend to each other through one
                transformer layer (set-to-set adaptation), then ProtoNet distance with a learned
                temperature. FEAT's auxiliary contrastive term is not implemented.
    "moemeta"   MoEMeta-TS (Wu & Yin, NeurIPS 2025, ported): MoE relation meta per class, local
                adaptation of the task vectors on the support set, translational distance, margin
                loss. Built by arch_variants.moemeta_ts (same episodic interface).  Q1 family.

  config["metric_backbone"]
    "cnnlstm"   DeepCNNLSTM (A2/A4 family; widths from cnn_base_filters / lstm_hidden)
    "moe"       DeepCNNLSTM_EncoderMOE (M0's expert bank, trained as a metric learner: separates
                "MoE features" from "MAML adaptation")
    "flexnet"   arch_variants.flexnet.FlexNet (any fusion / transformer variant)
    "spdnet"    arch_variants.spd.SPDNetEncoder (per-modality covariance -> BiMap/ReEig -> LogEig; P1_spd)

Training (metric_pretrain) mirrors mamlpp_pretrain's interface and history dict, so the HPO driver
and the epoch-level pruning callback treat both identically. Evaluation (metric_adapt_and_eval)
has mamlpp_adapt_and_eval's signature and return dict, so shared_maml.meta_evaluate works unchanged.
"""

from __future__ import annotations

import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

METRIC_KINDS = ("proto", "matching", "relation", "ridge", "feat")


def _build_encoder(config: dict) -> nn.Module:
    bb = config["metric_backbone"]
    cfg = dict(config)
    cfg["meta_learning"] = True        # the encoder's own head is unused; any n_way builds
    if bb == "cnnlstm":
        from pretraining.pretrain_models import build_model
        cfg["use_MOE"], cfg["model_type"] = False, "DeepCNNLSTM"
        return build_model(cfg)
    if bb == "moe":
        from MOE.MOE_encoder import build_MOE_model
        cfg["use_MOE"], cfg["model_type"], cfg["MOE_placement"] = True, "DeepCNNLSTM", "encoder"
        return build_MOE_model(cfg)
    if bb == "flexnet":
        from arch_variants.flexnet import FlexNet
        return FlexNet(cfg)
    if bb == "spdnet":
        from arch_variants.spd import SPDNetEncoder
        return SPDNetEncoder(cfg)
    raise ValueError(f"metric_backbone must be cnnlstm|moe|flexnet|spdnet, got {bb!r}")


class EpisodicMetricLearner(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        kind = config["metric_kind"]
        if kind not in METRIC_KINDS:
            raise ValueError(f"metric_kind must be one of {METRIC_KINDS}, got {kind!r}")
        self.kind = kind
        self.encoder = _build_encoder(config)
        D = self.encoder.feat_dim
        self.feat_dim = D
        if kind == "matching":
            self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        elif kind == "relation":
            h = config["metric_relation_hidden"]
            self.relation = nn.Sequential(nn.Linear(2 * D, h), nn.GELU(), nn.Linear(h, 1))
        elif kind == "ridge":
            self.log_lambda = nn.Parameter(torch.tensor(0.0))
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
        elif kind == "feat":
            from arch_variants.attention import MultiHeadAttention
            self.set_norm = nn.LayerNorm(D)
            self.set_attn = MultiHeadAttention(D, 1, dropout=config["dropout"])
            self.log_temp = nn.Parameter(torch.tensor(math.log(float(D))))

    # ── embedding ────────────────────────────────────────────────────────────
    def embed(self, x_emg, x_imu):
        """-> (features (B, D), routing dict or None)."""
        enc = self.encoder
        if hasattr(enc, "_backbone"):                                # FlexNet
            feat, _, routing = enc._backbone(x_emg, x_imu)
            return feat, routing
        if getattr(enc, "num_experts", None) is not None and hasattr(enc, "gate"):   # EncoderMOE
            feat, _, w_hard, w_soft = enc.backbone(x_emg, x_imu, return_routing=True)
            routing = {"gate_weights": w_hard, "gate_weights_soft": w_soft}
            if getattr(enc, "_last_router_logits", None) is not None:
                routing["router_logits"] = enc._last_router_logits
            enc._last_router_logits = None
            return feat, routing
        feat, _ = enc.backbone(x_emg, x_imu)
        return feat, None

    # ── episode logits ───────────────────────────────────────────────────────
    def episode_logits(self, support: dict, query: dict, device):
        """-> (query logits (Nq, n_way), query labels, routing dict or None)."""
        def _get(b, k):
            v = b.get(k)
            return v.to(device) if v is not None else None
        s_emg, s_imu, s_y = _get(support, "emg"), _get(support, "imu"), support["labels"].to(device).long()
        q_emg, q_imu, q_y = _get(query, "emg"), _get(query, "imu"), query["labels"].to(device).long()
        n_s = s_emg.shape[0]
        x_imu = torch.cat([s_imu, q_imu]) if s_imu is not None else None
        from MAML.mamlpp import task_support            # support-conditioned encoders (X4, X7T, R1)
        with task_support(self.encoder, support, device):
            z, routing = self.embed(torch.cat([s_emg, q_emg]), x_imu)
        zs, zq = z[:n_s], z[n_s:]
        n_way = int(s_y.max().item()) + 1
        onehot = F.one_hot(s_y, n_way).float()                        # (Ns, n)
        protos = (onehot.t() @ zs) / onehot.sum(dim=0, keepdim=True).t()   # (n, D) class means

        if self.kind == "proto":
            logits = -torch.cdist(zq, protos).pow(2)
        elif self.kind == "matching":
            sims = F.normalize(zq, dim=-1) @ F.normalize(zs, dim=-1).t() * self.log_scale.exp()  # (Nq, Ns)
            att = sims.softmax(dim=-1)
            logits = torch.log(att @ onehot + 1e-8)                  # log P(class) = log sum of attention
        elif self.kind == "relation":
            nq = zq.shape[0]
            pairs = torch.cat([zq.unsqueeze(1).expand(-1, n_way, -1),
                               protos.unsqueeze(0).expand(nq, -1, -1)], dim=-1)
            logits = self.relation(pairs).squeeze(-1)                 # (Nq, n)
        elif self.kind == "ridge":
            lam = self.log_lambda.exp()
            gram = zs @ zs.t() + lam * torch.eye(n_s, device=z.device, dtype=z.dtype)
            W = zs.t() @ torch.linalg.solve(gram, onehot)            # (D, n), Woodbury form
            logits = self.scale * (zq @ W) + self.bias
        else:  # feat
            p = protos.unsqueeze(0)                                  # (1, n, D): the set of prototypes
            p = (p + self.set_attn(self.set_norm(p))).squeeze(0)
            logits = -torch.cdist(zq, p).pow(2) / self.log_temp.exp()
        return logits, q_y, routing

    def forward(self, *args, **kwargs):
        raise RuntimeError("EpisodicMetricLearner is episodic: call episode_logits(support, query, device)")


def build_metric_learner(config: dict) -> EpisodicMetricLearner:
    if config["metric_kind"] == "moemeta":
        from arch_variants.moemeta_ts import MoEMetaTS
        return MoEMetaTS(config)
    return EpisodicMetricLearner(config)


def _episode_loss(model, episode, config, criterion):
    from MAML.mamlpp import _compute_aux_loss
    logits, y, routing = model.episode_logits(episode["support"], episode["query"], config["device"])
    # a learner with its own outer loss (MoEMeta-TS margin loss) overrides the cross-entropy criterion
    custom = getattr(model, "query_loss", None)
    loss = custom(logits, y) if custom is not None else criterion(logits, y)
    if routing is not None and config["use_MOE"]:
        loss = loss + _compute_aux_loss(routing, config, float(config["MOE_aux_coeff"]))
    return loss, logits, y


@torch.no_grad()
def metric_adapt_and_eval(model, config, support_batch, query_batch, debug=False):
    """Same signature / return dict as mamlpp.mamlpp_adapt_and_eval (used by shared_maml.meta_evaluate)."""
    from MAML.mamlpp import assert_support_query_rep_disjoint
    assert_support_query_rep_disjoint(support_batch, query_batch)
    was_training = model.training
    model.eval()
    logits, y, _ = model.episode_logits(support_batch, query_batch, config["device"])
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(dim=-1) == y).float().mean().item()
    if was_training:
        model.train()
    return {"loss": loss, "acc": acc, "pre_adapt_acc": None, "logits": logits.detach().float().cpu()}


def metric_pretrain(model, config, episodic_train_loader, episodic_val_loader=None, epoch_callback=None):
    """
    Episodic training without an inner loop. Same history keys as mamlpp_pretrain
    (val_acc_log, best_state, best_val_acc, best_val_epoch, train_loss_log, routing_reports).
    Outer optimiser, meta-batch accumulation, gradient clipping and early stopping follow the MAML path.
    """
    from MAML.shared_maml import meta_evaluate
    from MOE.MOE_training import SmoothedEarlyStopping

    device = config["device"]
    model.to(device)
    label_smooth = float(config["label_smooth"])
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth) if label_smooth > 0 else nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
                           weight_decay=float(config["weight_decay"]))
    meta_bs = int(config["meta_batchsize"])
    eps_per_epoch = int(config["episodes_per_epoch_train"])
    clip = float(config["gradient_clip_max_norm"])
    es = SmoothedEarlyStopping(patience=int(config["earlystopping_patience"]),
                               min_delta=float(config["earlystopping_min_delta"])) \
        if (episodic_val_loader is not None and bool(config["use_earlystopping"])) else None

    best_val_acc, best_state, best_epoch = -1.0, None, 0
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []
    for ep in range(1, int(config["num_epochs"]) + 1):
        t0 = time.time()
        model.train()
        n_ep = acc_n = 0
        loss_sum = correct = total = 0.0
        opt.zero_grad(set_to_none=True)
        done = False
        for step_item in episodic_train_loader:
            for episode in ([step_item] if isinstance(step_item, dict) else step_item):
                loss, logits, y = _episode_loss(model, episode, config, criterion)
                (loss / meta_bs).backward()
                loss_sum += float(loss.item())
                correct += float((logits.argmax(-1) == y).sum().item())
                total += y.numel()
                n_ep += 1
                acc_n += 1
                if acc_n == meta_bs:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    acc_n = 0
                if n_ep >= eps_per_epoch:
                    done = True
                    break
            if done:
                break
        if acc_n > 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad *= meta_bs / acc_n
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
        train_loss_log.append(loss_sum / max(n_ep, 1))
        train_acc_log.append(correct / max(total, 1))
        print(f"[metric:{config['metric_kind']}] epoch {ep}: train loss {train_loss_log[-1]:.4f}, "
              f"acc {train_acc_log[-1]*100:.2f}% ({time.time()-t0:.1f}s)")

        if episodic_val_loader is None:
            continue
        vm = meta_evaluate(model, episodic_val_loader, config, metric_adapt_and_eval)
        val_loss_log.append(vm["loss"])
        val_acc_log.append(vm["acc"])
        print(f"[metric:{config['metric_kind']}] epoch {ep}: val loss {vm['loss']:.4f}, "
              f"per-user-mean acc {vm['acc']*100:.2f}%")
        if epoch_callback is not None:
            epoch_callback(ep, vm["acc"])
        if vm["acc"] > best_val_acc:
            best_val_acc, best_epoch = vm["acc"], ep
            best_state = copy.deepcopy(model.state_dict())
        if es is not None and es(vm["loss"]):
            print(f"[metric] early stopping at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"train_loss_log": train_loss_log, "train_acc_log": train_acc_log,
                   "val_loss_log": val_loss_log, "val_acc_log": val_acc_log, "model": model,
                   "best_state": best_state, "best_val_acc": best_val_acc,
                   "best_val_epoch": best_epoch, "routing_reports": [], "moe_collapsed": False}
