"""
Z_zeroshot_crossuser.py
=======================
Zero-shot cross-user experiment (docs/specs/SPEC_zeroshot.md; framing thresholds fixed there in advance).

Per kfold7 fold k: train on the fold's 20 train users with GLOBAL labels (gesture id 0..9), select on its
4 val users, then classify every sample (10 gestures x 10 reps) of its 4 test users with NO support set
and no adaptation. Chance is 10%.

  Z1  supervised CNN-LSTM, A2 architecture and parameter count, 10-way softmax head. Hyperparameters:
      A2's HPO result (HPO_CONFIG=tuned), i.e. from the dev users only. Checkpoint selection: zero-shot
      LOSS on the whole val users (pretrain_trainer.pretrain selects on val loss; the val loader holds
      all 10 reps of the val users, so it is a zero-shot criterion).
  Z2  nearest class centroid (cosine) in Z1's penultimate embedding; centroids = mean embedding of each
      gesture over the 20 train users. Representation-level transfer, not head-confounded.
  Z3  the same centroid rule in the M0 encoder embedding of the SAME fold's M0 checkpoint (--checkpoint).
  Z4  Riemannian tangent space of per-modality channel covariance (EMG and IMU blocks separately, OAS
      shrinkage, reference mean over the train users) + logistic regression.

Outputs (results JSON): per model, per-user accuracy over its 100 samples, per-gesture accuracy pooled
over users, a 10x10 confusion matrix, per-user binomial tests against 0.10 (count significant after Holm),
and per_sample rows (user, gesture, rep, pred). Not aggregated with the episodic methods (different task).

    python Z_zeroshot_crossuser.py --fold-idx 0 --models Z1 Z2 Z4
    python Z_zeroshot_crossuser.py --fold-idx 0 --models Z3 --checkpoint <fold-0 M0 *_best.pt>
kfold_launcher.sh tokens: Z_all (Z1+Z2+Z4), Z3 (after the fold's M0 job).
"""

import os, sys, copy, argparse, pickle
import numpy as np
import torch

from pathlib import Path
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
for _p in ("", "system", "system/MAML", "system/MOE", "system/pretraining"):
    sys.path.insert(0, str(CODE_DIR / _p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

N_CLASSES = 10
CHANCE = 1.0 / N_CLASSES
ALL_REPS = list(range(1, 11))


# ─────────────────────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────────────────────
def collect(tensor_dict: dict, pids: list):
    """Every (gesture, rep) of the users, from a REORIENTED dict ((trials, C, T)).
    -> dict of arrays: emg (n, Ce, T), imu (n, Ci, T), y (n,), user (n,), rep (n,) (rep 1-indexed)."""
    emg, imu, y, user, rep = [], [], [], [], []
    for pid in pids:
        if pid not in tensor_dict:
            raise KeyError(f"{pid} not in the tensor dict")
        for g in range(N_CLASSES):
            slot = tensor_dict[pid][g]
            n = slot["emg"].shape[0]
            if n != len(ALL_REPS):
                raise ValueError(f"{pid}/{g}: {n} reps, expected {len(ALL_REPS)}")
            emg.append(slot["emg"].float().numpy())
            imu.append(slot["imu"].float().numpy())
            y += [g] * n
            user += [pid] * n
            rep += ALL_REPS
    return {"emg": np.concatenate(emg), "imu": np.concatenate(imu), "y": np.asarray(y),
            "user": np.asarray(user), "rep": np.asarray(rep)}


# ─────────────────────────────────────────────────────────────────────────────
# classifiers
# ─────────────────────────────────────────────────────────────────────────────
def nearest_centroid_cosine(train_feat, train_y, test_feat) -> np.ndarray:
    """Centroid per class (mean over all train samples of that class), cosine assignment."""
    classes = np.unique(train_y)
    cent = np.stack([train_feat[train_y == c].mean(axis=0) for c in classes])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    q = test_feat / (np.linalg.norm(test_feat, axis=1, keepdims=True) + 1e-12)
    return classes[(q @ cent.T).argmax(axis=1)]


def z4_riemann(train: dict, test: dict, C: float = 1.0) -> np.ndarray:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ftr, fte = [], []
    for key in ("emg", "imu"):
        cov = Covariances(estimator="oas")
        ctr = cov.fit_transform(train[key].astype(np.float64))
        cte = cov.transform(test[key].astype(np.float64))
        ts = TangentSpace(metric="riemann").fit(ctr)          # reference mean over the TRAIN users
        ftr.append(ts.transform(ctr))
        fte.append(ts.transform(cte))
    xtr, xte = np.concatenate(ftr, axis=1), np.concatenate(fte, axis=1)
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(xtr), train["y"])
    return clf.predict(sc.transform(xte))


@torch.no_grad()
def embed(model, data: dict, device, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(data["y"]), batch):
        e = torch.from_numpy(data["emg"][i:i + batch]).to(device)
        m = torch.from_numpy(data["imu"][i:i + batch]).to(device)
        out.append(model.backbone(e, m)[0].float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_head(model, data: dict, device, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(data["y"]), batch):
        e = torch.from_numpy(data["emg"][i:i + batch]).to(device)
        m = torch.from_numpy(data["imu"][i:i + batch]).to(device)
        logits = model(e, m)
        logits = logits[0] if isinstance(logits, tuple) else logits
        out.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(out)


# ─────────────────────────────────────────────────────────────────────────────
# reporting
# ─────────────────────────────────────────────────────────────────────────────
def summarise(pred, data: dict) -> dict:
    from scipy.stats import binomtest
    from analysis.stats_utils import holm
    y, users = data["y"], data["user"]
    correct = pred == y
    per_user = {u: float(correct[users == u].mean()) for u in sorted(set(users))}
    pvals = [binomtest(int(correct[users == u].sum()), int((users == u).sum()), CHANCE,
                       alternative="greater").pvalue for u in per_user]
    adj = holm(pvals)
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    return {
        "per_user_acc": per_user,
        "mean_acc": float(np.mean(list(per_user.values()))),
        "std_acc": float(np.std(list(per_user.values()))),
        "per_gesture_acc": {int(g): float(correct[y == g].mean()) for g in range(N_CLASSES)},
        "confusion": conf.tolist(),
        "binomial_vs_chance": {"p": dict(zip(per_user, pvals)), "p_holm": dict(zip(per_user, adj)),
                               "n_users_above_chance_holm": int(sum(a < 0.05 for a in adj))},
        "per_sample": [{"user": str(u), "gesture": int(g), "rep": int(r), "pred": int(p)}
                       for u, g, r, p in zip(users, y, data["rep"], pred)],
        "chance": CHANCE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# models
# ─────────────────────────────────────────────────────────────────────────────
def build_z1_config(seed: int, smoke: bool):
    from A2_no_maml_no_moe import build_config as a2_config
    from ablation_config import apply_smoke
    config = a2_config()                        # ablation_id "A2" so HPO_CONFIG=tuned picks A2's result
    config["train_reps"] = list(ALL_REPS)        # zero-shot: all reps of the train users ...
    config["val_reps"] = list(ALL_REPS)          # ... and the WHOLE val users as the selection set
    config["seed"] = seed
    return apply_smoke(config, smoke)


def train_z1(config: dict, tensor_dict: dict):
    from ablation_config import build_supervised_no_moe_model, count_parameters
    from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
    from pretraining.pretrain_trainer import pretrain
    model = build_supervised_no_moe_model(config).to(config["device"])
    train_dl, val_dl, n_classes = get_pretrain_dataloaders(config, tensor_dict)
    if n_classes != N_CLASSES:
        raise ValueError(f"expected {N_CLASSES} classes, got {n_classes}")
    model, hist = pretrain(model, train_dl, val_dl, config)
    return model, hist, count_parameters(model)


def load_encoder(checkpoint: str, device):
    """The model saved in a fold checkpoint (M0_full_model.py / arch_variant_eval.py format)."""
    from arch_variants.registry import arch_defaults, build_model_for_config
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    if "arch_family" not in cfg:
        # checkpoints written before the exploration keys existed: their models ARE the defaults
        cfg = {**arch_defaults({}), **cfg}
    cfg["device"] = device
    model = build_model_for_config(cfg)
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device), cfg


def main():
    from ablation_config import (apply_kfold_fold, add_fold_arg, add_seed_arg, add_smoke_arg, save_results,
                                 set_seeds, tensor_dict_file, kfold_tag)
    from MAML.maml_data_pipeline import reorient_tensor_dict
    ap = argparse.ArgumentParser(description="Zero-shot cross-user Z1-Z4 (SPEC_zeroshot.md)")
    add_fold_arg(ap, required=True)
    add_seed_arg(ap)
    add_smoke_arg(ap)
    ap.add_argument("--models", nargs="+", choices=["Z1", "Z2", "Z3", "Z4"], required=True)
    ap.add_argument("--checkpoint", default=None, help="the fold's M0 checkpoint (needed for Z3)")
    args = ap.parse_args()
    if "Z3" in args.models and not args.checkpoint:
        ap.error("Z3 needs --checkpoint (the same fold's M0 *_best.pt)")
    if "Z2" in args.models and "Z1" not in args.models:
        ap.error("Z2 uses Z1's embedding: run them together (--models Z1 Z2 ...)")

    set_seeds(args.seed)
    config = apply_kfold_fold(build_z1_config(args.seed, args.smoke), args.fold_idx)
    if args.checkpoint:
        _, ck_cfg = load_encoder(args.checkpoint, "cpu")
        if ck_cfg.get("fold_idx") != args.fold_idx or ck_cfg["emg_repr"] != config["emg_repr"]:
            raise ValueError(f"checkpoint is fold {ck_cfg.get('fold_idx')} / {ck_cfg['emg_repr']}, this run is "
                             f"fold {args.fold_idx} / {config['emg_repr']}")
    with open(tensor_dict_file(config), "rb") as f:
        tensor_dict = reorient_tensor_dict(pickle.load(f), config)
    train = collect(tensor_dict, config["train_PIDs"])
    test = collect(tensor_dict, config["test_PIDs"])
    print(f"[Z fold {args.fold_idx}] train {len(train['y'])} samples ({len(config['train_PIDs'])} users), "
          f"test {len(test['y'])} samples ({config['test_PIDs']})")

    results, extra = {}, {}
    if "Z1" in args.models:
        model, hist, n_params = train_z1(config, tensor_dict)
        results["Z1"] = summarise(predict_head(model, test, config["device"]), test)
        extra["Z1"] = {"n_params": n_params, "best_val_acc": float(max(hist["val_acc"])) if hist["val_acc"] else None}
        if "Z2" in args.models:
            results["Z2"] = summarise(nearest_centroid_cosine(
                embed(model, train, config["device"]), train["y"], embed(model, test, config["device"])), test)
    if "Z3" in args.models:
        enc, _ = load_encoder(args.checkpoint, config["device"])
        results["Z3"] = summarise(nearest_centroid_cosine(
            embed(enc, train, config["device"]), train["y"], embed(enc, test, config["device"])), test)
        extra["Z3"] = {"checkpoint": args.checkpoint}
    if "Z4" in args.models:
        results["Z4"] = summarise(z4_riemann(train, test), test)

    for m, r in results.items():
        print(f"[Z fold {args.fold_idx}] {m}: {r['mean_acc']*100:.1f}% (chance 10%), "
              f"{r['binomial_vs_chance']['n_users_above_chance_holm']}/{len(r['per_user_acc'])} users above "
              f"chance after Holm")
    config["ablation_id"] = "Z_zeroshot"
    save_results({"ablation_id": "Z_zeroshot", "fold_idx": args.fold_idx, "seed": args.seed,
                  "test_procedure": "kfold7", "task": "zero-shot, global labels, 10 classes, no support",
                  "models": results, "model_info": extra, "test_PIDs": config["test_PIDs"],
                  "config_snapshot": {k: str(v) for k, v in config.items()}},
                 config, tag=kfold_tag(args.fold_idx, args.seed) + "_" + "".join(sorted(args.models)))


if __name__ == "__main__":
    main()
