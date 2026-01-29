import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def swap_expert_head_to_cosine(model, emb_dim, num_classes, init_tau):
    """THIS IS THE OLD VERSION. There is a newer version (a class method of the model) in MOE_mutlimodal_model_classes"""
    for i, exp in enumerate(model.experts):
        # exp: Expert with fc1, norm, drop, fc2
        exp.fc2 = CosineHead(emb_dim=emb_dim, num_classes=num_classes, init_tau=init_tau)

def make_gate(config):
    gate_type = config["gate_type"]
    emb_dim = config["emb_dim"]
    user_dim = config["user_emb_dim"]
    num_experts = config["num_experts"]
    top_k = config["top_k"]
    
    if gate_type == "user_aware":
        return UserAwareGate(emb_dim=emb_dim, user_dim=user_dim, num_experts=num_experts, top_k=top_k)
    elif gate_type == "feature_only":
        return FeatureOnlyGate(emb_dim=emb_dim, num_experts=num_experts, top_k=top_k)
    elif gate_type == "user_only":
        return UserOnlyGate(user_dim=user_dim, num_experts=num_experts, top_k=top_k)
    elif gate_type == "film":
        return FiLMGate(emb_dim=emb_dim, user_dim=user_dim, num_experts=num_experts, top_k=top_k)
    elif gate_type == "bilinear":
        
        return BilinearGate(emb_dim=emb_dim, user_dim=user_dim, num_experts=num_experts, rank=config["rank"], top_k=top_k)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")
    
# Override an existing model to take u_user. Easiest way to pass this in without a more in depth refactor
## Wait what is the purpose of this? Why can't I just pass it in every time? 
## I am doing this so that I don't have to edit the forward passes in the rest of my code I think?
class WithUserOverride(nn.Module):
    """
    Wraps a model and forces use of a specific user embedding vector.
    - Works with unimodal (x) or multimodal (x_emg/x_imu/demographics/user_ids) calls.
    - No *args/**kwargs to avoid duplicate-kw collisions.
    """

    def __init__(self, model: nn.Module, u_user: torch.Tensor, multimodal: bool = False):
        super().__init__()
        self.model = model
        self.multimodal = multimodal
        if u_user is None:
            raise ValueError("WithUserOverride requires a non-None user embedding tensor `u_user`.")
        if u_user.dim() == 1:
            u_user = u_user.unsqueeze(0)
        if u_user.dim() != 2:
            raise ValueError(f"`u_user` must be 2D (1, user_dim). Got shape {tuple(u_user.shape)}.")
        self.register_buffer("u_user", u_user.detach(), persistent=True)
        self.u_user_param = None

    # ---------- management ----------
    def set_u_user(self, new_u: torch.Tensor):
        if new_u is None:
            raise ValueError("`new_u` cannot be None.")
        if new_u.dim() == 1:
            new_u = new_u.unsqueeze(0)
        if new_u.dim() != 2 or new_u.size(0) != 1:
            raise ValueError(f"`new_u` must be shape (1, user_dim). Got {tuple(new_u.shape)}.")
        self.u_user.data = new_u.detach().to(self.u_user.device)

    def begin_user_training(self, u_init: torch.Tensor | None = None):
        tgt = u_init if u_init is not None else self.u_user
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)
        self.u_user_param = nn.Parameter(tgt.detach().to(self.u_user.device).clone(), requires_grad=True)

    def end_user_training(self, commit: bool = True):
        if commit and self.u_user_param is not None:
            self.set_u_user(self.u_user_param.data)
        self.u_user_param = None

    # ---------- forward ----------
    def forward(
        self,
        x=None, *,
        x_emg=None,
        x_imu=None,
        demographics=None,
        user_ids=None,
        return_aux=None,
        user_embed_override=None,
    ):
        if self.u_user is None and self.u_user_param is None and user_embed_override is None:
            raise RuntimeError("`u_user` is not set. Call set_u_user(...) or pass user_embed_override.")

        # ---------- multimodal ----------
        if self.multimodal:
            # Allow dict batch as `x`
            if isinstance(x, dict):
                x_emg        = x.get("emg",  x_emg)
                x_imu        = x.get("imu",  x_imu)
                demographics = x.get("demo", demographics)
                user_ids     = x.get("PIDs", user_ids)
            # Back-compat: if caller passed EMG positionally, accept it
            if x_emg is None and torch.is_tensor(x):
                x_emg = x

            if x_emg is None or not torch.is_tensor(x_emg):
                raise ValueError("Multimodal path requires `x_emg` tensor (or dict['emg']).")

            B = x_emg.size(0)
            device = x_emg.device

            # Build a (B, U) user embedding batch
            if user_embed_override is not None:
                u = user_embed_override.to(device)
            else:
                base_u = self.u_user_param if self.u_user_param is not None else self.u_user
                u = base_u.to(device)

            # Normalize shapes:
            # - (U,) -> (1, U)
            # - (1, U) -> expand to (B, U)
            # - (B, U) -> keep
            if u.dim() == 1:
                u = u.unsqueeze(0)
            if u.size(0) == 1 and B > 1:
                u = u.expand(B, -1)
            if u.size(0) != B:
                raise RuntimeError(f"User embedding batch size mismatch: got {u.size(0)}, expected {B}")

            # Optional: verify rows are identical (your original intent)
            # (comment out if you expect mixed-user batches)
            if not torch.allclose(u, u[0].expand_as(u)):
                raise RuntimeError("Expected a single user per batch; got differing user embeddings across rows.")

            # Default to True in multimodal unless explicitly set
            if return_aux is None:
                return_aux = True

            return self.model(
                x_emg=x_emg,
                x_imu=x_imu,
                demographics=demographics,
                user_ids=user_ids,
                user_embed_override=u,   # (B, U) — tiled, not compressed
                return_aux=return_aux,
            )

        # ---------- unimodal ----------
        # Accept either positional `x` or keyword `x_emg`
        if x is None:
            x = x_emg
        if x is None or not torch.is_tensor(x):
            raise ValueError("Unimodal path expects EMG tensor via `x` (positional) or `x_emg`.")
        B = x.size(0)
        device = x.device

        if user_embed_override is not None:
            u = user_embed_override.to(device)
        else:
            base_u = self.u_user_param if self.u_user_param is not None else self.u_user
            u = base_u.to(device)

        if u.dim() == 1:
            u = u.unsqueeze(0)
        if u.size(0) == 1 and B > 1:
            u = u.expand(B, -1)
        if u.size(0) != B:
            raise RuntimeError(f"User embedding batch size mismatch: got {u.size(0)}, expected {B}")

        if not torch.allclose(u, u[0].expand_as(u)):
            raise RuntimeError("Expected a single user per batch; got differing user embeddings across rows.")

        return self.model(
            x,                   # legacy models expect positional features
            user_ids=None,       # explicit; ignored by non-MoE models
            user_embed_override=u,  # (B, U)
        )


    # ---------- proxy ----------
    def __getattr__(self, name: str):
        if name in {"model", "u_user", "u_user_param"}:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def __dir__(self):
        return sorted(set(list(super().__dir__()) + dir(self.model)))
    

def compress_user_embed_batch(u_batch: torch.Tensor, *, name="user_embed_override") -> torch.Tensor:
    """
    If u_batch is (B,D), assert all B rows are identical and return shape (1,D).
    If already (1,D), return as-is.
    """
    if u_batch.dim() != 2:
        raise ValueError(f"{name} must be 2D (B,D). Got {tuple(u_batch.shape)}")
    B, D = u_batch.shape
    if B == 1:
        return u_batch

    first = u_batch[0]                     # (D,)
    # exact equality is fine here (they came from .expand()); if you want tolerance, add rtol/atol
    same = torch.equal(u_batch, first.unsqueeze(0).expand_as(u_batch))
    if not same:
        # helpful diagnostics
        max_abs_diff = (u_batch - first).abs().max().item()
        raise ValueError(
            f"{name} varies across batch (expected identical rows). "
            f"max|diff|={max_abs_diff:.3e}, shape={tuple(u_batch.shape)}"
        )
    return first.unsqueeze(0)              # (1,D)


class CosineHead(nn.Module):
    def __init__(self, emb_dim, num_classes, init_tau=10.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, emb_dim)*0.02)
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))
    def forward(self, h):
        h = F.normalize(h, dim=-1); W = F.normalize(self.W, dim=-1)
        return self.tau * (h @ W.t())

# ----- Backbone (MLP) -----
class TinyBackbone(nn.Module):
    def __init__(self, in_ch=16, seq_len=5, emb_dim=64):
        super().__init__()
        self.in_dim = in_ch * seq_len  # 80
        self.fc1 = nn.Linear(self.in_dim, 64)
        self.fc2 = nn.Linear(64, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, x):  # x: (B, 16, 5)
        B = x.size(0)
        x = x.view(B, -1)           # (B, 80)
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))
        return self.norm(h)         # (B, 64)

# ----- Expert head -----
class Expert(nn.Module):
    # TODO: Try swapping the last linear layer in Expert with CosineHead(emb_dim, NUM_CLASSES)

    def __init__(self, emb_dim=64, num_classes=10, pdrop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, num_classes)
        self.drop = nn.Dropout(pdrop)
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, h):
        z = F.gelu(self.fc1(h))
        z = self.norm(z)
        z = self.drop(z)
        return self.fc2(z)  # logits

# ----- Gating (feature + user embedding) -----
class UserAwareGate(nn.Module):
    def __init__(self, emb_dim=64, user_dim=16, num_experts=6, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.lin = nn.Linear(emb_dim + user_dim, num_experts)
    def forward(self, h, u):  # h:(B,64) u:(B,16)
        g = self.lin(torch.cat([h, u], dim=-1))  # (B,E)
        w = F.softmax(g, dim=-1)                 # dense probs
        # Top-k sparsification
        if self.top_k is not None and self.top_k < w.size(-1):
            topk = torch.topk(w, self.top_k, dim=-1)
            mask = torch.zeros_like(w).scatter(-1, topk.indices, 1.0)
            w = w * mask
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
        return w  # (B,E)
    
class FeatureOnlyGate(nn.Module):
    def __init__(self, emb_dim=64, num_experts=6, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.lin = nn.Linear(emb_dim, num_experts)
    def forward(self, h, u=None):
        g = self.lin(h)
        w = F.softmax(g, dim=-1)
        if self.top_k and self.top_k < w.size(-1):
            idx = torch.topk(w, self.top_k, dim=-1).indices
            mask = torch.zeros_like(w).scatter(-1, idx, 1.0)
            w = (w * mask) / (w * mask).sum(-1, keepdim=True).clamp_min(1e-9)
        return w

class UserOnlyGate(nn.Module):
    def __init__(self, user_dim=16, num_experts=6, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.lin = nn.Linear(user_dim, num_experts)
    def forward(self, h, u):
        w = F.softmax(self.lin(u), dim=-1)
        if self.top_k and self.top_k < w.size(-1):
            idx = torch.topk(w, self.top_k, dim=-1).indices
            mask = torch.zeros_like(w).scatter(-1, idx, 1.0)
            w = (w * mask) / (w * mask).sum(-1, keepdim=True).clamp_min(1e-9)
        return w

class FiLMGate(nn.Module):
    def __init__(self, emb_dim=64, user_dim=16, num_experts=6, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gamma = nn.Linear(user_dim, emb_dim)
        self.beta  = nn.Linear(user_dim, emb_dim)
        self.lin   = nn.Linear(emb_dim, num_experts)
    def forward(self, h, u):
        h_t = h * (1 + self.gamma(u)) + self.beta(u)  # affine mod
        w = F.softmax(self.lin(h_t), dim=-1)
        if self.top_k and self.top_k < w.size(-1):
            idx = torch.topk(w, self.top_k, dim=-1).indices
            mask = torch.zeros_like(w).scatter(-1, idx, 1.0)
            w = (w * mask) / (w * mask).sum(-1, keepdim=True).clamp_min(1e-9)
        return w

class BilinearGate(nn.Module):
    def __init__(self, emb_dim=64, user_dim=16, num_experts=6, rank=8, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.U = nn.Parameter(torch.randn(num_experts, emb_dim, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(num_experts, user_dim, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_experts))
    def forward(self, h, u):
        # g_e = sum_r (h @ U[e,:,r]) * (u @ V[e,:,r]) + b_e
        hU = torch.einsum('bd,erd->ber', h, self.U)  # (B,E,R)
        uV = torch.einsum('bd,erd->ber', u, self.V)  # (B,E,R)
        g = (hU * uV).sum(-1) + self.bias            # (B,E)
        w = F.softmax(g, dim=-1)
        if self.top_k and self.top_k < w.size(-1):
            idx = torch.topk(w, self.top_k, dim=-1).indices
            mask = torch.zeros_like(w).scatter(-1, idx, 1.0)
            w = (w * mask) / (w * mask).sum(-1, keepdim=True).clamp_min(1e-9)
        return w

# ----- MoE classifier -----
class MoEClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config["num_experts"]
        self.emb_dim = config["emb_dim"]
        self.user_emb_dim = config["user_emb_dim"]
        self.backbone = TinyBackbone(emb_dim=self.emb_dim)
        self.experts = nn.ModuleList([Expert(emb_dim=self.emb_dim, num_classes=config["num_classes"]) for _ in range(self.num_experts)])
        self.gate = make_gate(config)
        self.use_user_table = config["use_user_table"]
        if self.use_user_table:
            self.user_table = nn.Embedding(config["num_pretrain_users"], self.user_emb_dim)  # train-time users only
        else:
            self.user_table = None
        # Optional: expert keys for prototype routing
        self.expert_keys = nn.Parameter(torch.randn(self.num_experts, self.emb_dim) * 0.1)

    def forward(self, x, user_ids=None, user_embed_override=None, return_aux=False):
        """
        user_ids: (B,) LongTensor for known training users (pretrain), ignored if user_embed_override is provided.
        user_embed_override: (B, user_dim) for PEFT or prototype mode at test time (novel users).
        """
        h = self.backbone(x)  # (B,64)

        # Build user embedding
        if user_embed_override is not None:
            u = user_embed_override
        elif (self.user_table is not None) and (user_ids is not None):
            u = self.user_table(user_ids)
        else:
            # TODO: This is a bad case! Should only run for feature-only...
            ## Idk how to call this out in the least annoying way possible...
            # fallback: zeros (no user info)
            u = torch.zeros(h.size(0), self.user_emb_dim, device=h.device)

        # Gate
        w = self.gate(h, u)  # (B,E)

        # Expert logits + mixture
        ## Technically the principled approach would be to mix probabilities, not logits, but mixing logits is common in practice
        logits_per_exp = torch.stack([exp(h) for exp in self.experts], dim=1)  # (B,E,C)
        w_expanded = w.unsqueeze(-1)  # (B,E,1)
        logits = (w_expanded * logits_per_exp).sum(dim=1)  # (B,C)

        # Different mixture approaches I could try.
        # Probability version
        #elif mixture_mode == "probs":
        #    probs_per_exp = torch.softmax(logits_per_exp, dim=-1)              # (B,E,C)
        #    probs = (w.unsqueeze(-1) * probs_per_exp).sum(dim=1)               # (B,C)
        #    out = probs
        ## Log-sum-exp with prior weights (exact in log space)
        #elif mixture_mode == "logprobs":
        #    log_probs_per_exp = torch.log_softmax(logits_per_exp, dim=-1)      # (B,E,C)
        #    log_w = torch.log(w.clamp_min(1e-9)).unsqueeze(-1)                 # (B,E,1)
        #    log_mix = torch.logsumexp(log_w + log_probs_per_exp, dim=1)        # (B,C)
        #    out = log_mix

        aux = {}
        if return_aux:
            # Simple load-balancing proxy: mean gate usage per expert
            aux["gate_usage"] = w.mean(dim=0)  # (E,)
        return logits, aux