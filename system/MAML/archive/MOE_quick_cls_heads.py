import torch
import torch.nn.functional as F
from collections import defaultdict

###################################################################################
# These do NOT update MOE at all, don't even use the experts/MOE part!
# Only take the model's backbone as a feature extractor

###################################################################################
# Then use cosine similarity to per-class prototypes on the one-shot support set to do the classification

@torch.no_grad()
def backbone_embed(model, loader, device):
    """Gets the latent space embeddings from the backbone model forward pass"""
    model.eval()
    H, Y = [], []
    for b in loader:
        xb = (b["x"] if isinstance(b, dict) else b[0]).to(device)
        yb = (b["y"] if isinstance(b, dict) else b[1]).to(device)
        hb = model.backbone(xb)  # (B, emb)
        H.append(hb); Y.append(yb)
    return torch.cat(H,0), torch.cat(Y,0)

@torch.no_grad()
def proto_eval_user(model, support_loader, query_loader, config, tau=0.5):
    device = config["device"]
    num_classes = config["num_classes"]

    # 1) prototypes from support
    Hs, Ys = backbone_embed(model, support_loader, device)
    cls2h = defaultdict(list)
    for h, y in zip(Hs, Ys):
        cls2h[int(y)] .append(h)
    protos = []
    for c in range(num_classes):
        if len(cls2h[c]) == 0:
            protos.append(torch.zeros(Hs.size(1), device=device))
        else:
            protos.append(torch.stack(cls2h[c],0).mean(0))
    P = torch.stack(protos,0)                    # (C, emb)
    P = F.normalize(P, dim=-1)

    # 2) cosine classification on query
    Hq, Yq = backbone_embed(model, query_loader, device)
    Hq = F.normalize(Hq, dim=-1)
    logits = (Hq @ P.t()) / tau
    acc = (logits.argmax(-1) == Yq).float().mean().item()
    return acc

###################################################################################
# Closed form ridge (L2) classification using the backbone embeddings
# Also does not update/use MOE

# Ought to add:
# cale/standardize features.
## L2-normalize X rows (or whiten) before fitting; this often stabilizes one-shot.
# Regularization strength.
## With one-shot per class, push reg up (e.g., grid over {0.1, 1, 10, 100}). A heuristic: start around reg ≈ D / max(N, C).

def ridge_fit(X, Y, num_classes, reg=1.0):
    # X:(N,D) Y:(N,), one-vs-rest ridge on one-hot
    N, D = X.shape
    T = X.new_zeros(N, num_classes); T[torch.arange(N), Y] = 1.0
    X_mean = X.mean(0, keepdim=True)
    T_mean = T.mean(0, keepdim=True)
    Xc = X - X_mean
    Tc = T - T_mean
    W = torch.linalg.solve(Xc.t() @ Xc + reg*torch.eye(D, device=X.device),
                           Xc.t() @ Tc)
    b = T_mean - X_mean @ W         # intercept (not regularized)
    return W, b

#Dual form for N ≪ D.
#When embeddings are high-dim and support is tiny, invert the N×N system instead:
def ridge_fit_dual(X, Y, num_classes, reg=1.0):
    N, D = X.shape
    T = X.new_zeros(N, num_classes); T[torch.arange(N), Y] = 1.0
    K = X @ X.t()                                   # (N,N)
    A = K + reg*torch.eye(N, device=X.device)
    W = X.t() @ torch.linalg.solve(A, T)            # (D,C)
    # (add intercept as above if desired)
    return W, T.mean(0, keepdim=True)

@torch.no_grad()
def ridge_eval_user(model, support_loader, query_loader, config, reg=1.0):
    device = config["device"]
    num_classes = config["num_classes"]
    
    Hs, Ys = backbone_embed(model, support_loader, device)
    W, b = ridge_fit(Hs, Ys, num_classes, reg=reg)
    Hq, Yq = backbone_embed(model, query_loader, device)
    logits = Hq @ W + b
    acc = (logits.argmax(-1) == Yq).float().mean().item()
    return acc
