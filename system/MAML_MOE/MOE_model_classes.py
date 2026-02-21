import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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