"""
A14_protonet_encoder.py
=======================
ProtoNet / kNN readout on the FROZEN meta-trained backbone of the same fold, evaluated on the shared
cached test episodes (docs/specs/SPEC_episode_cache.md). Answers "is the meta-learned adaptation better
than a nearest-prototype readout on the same features?" (reviewer R1 W1 / R4 W2).

    python A14_protonet_encoder.py --fold-idx 0 --checkpoint <M0 or A4 fold checkpoint .pt> \
        --model {M0,A4} [--cells all]

Replaces the old A14_protonet_baseline.py, which used its own sampler and the 4kfcv/val users.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

CODE_DIR = Path(os.environ["CODE_DIR"]).resolve()
for rel in ("", "system", "system/MAML", "system/MOE", "system/pretraining"):
    sys.path.insert(0, str(CODE_DIR / rel) if rel else str(CODE_DIR))

from ablation_config import (  # noqa: E402
    tensor_dict_file,
    _print_eval, _summarise_eval, add_fold_arg, apply_kfold_fold, build_maml_moe_model,
    build_maml_no_moe_model, count_parameters,
    make_base_config, save_results,
)
from MAML.maml_data_pipeline import maml_mm_collate, reorient_tensor_dict  # noqa: E402
from episodes import episode_cache as ec  # noqa: E402
from episodes.episode_torch import CachedEpisodeDataset, outcomes_to_per_episode  # noqa: E402


@torch.no_grad()
def features(model, emg, imu, device, moe):
    """Pooled final-layer feature the classification head consumes (backbone returns (l3, [..]))."""
    args = (emg.to(device), None if imu is None else imu.to(device), None)
    out = model.backbone(*args, return_routing=False) if moe else model.backbone(*args)
    return out[0] if isinstance(out, tuple) else out


def predict_readouts(model, batch, device, moe):
    """-> {readout name: predicted local labels (np array)} for one collated episode."""
    fs = features(model, batch["support"]["emg"], batch["support"]["imu"], device, moe)
    fq = features(model, batch["query"]["emg"], batch["query"]["imu"], device, moe)
    ys = batch["support"]["labels"].to(device)
    n = int(ys.max()) + 1
    protos = torch.stack([fs[ys == c].mean(dim=0) for c in range(n)])
    d_proto = torch.cdist(fq, protos)                       # Euclidean to class means
    d_knn = torch.cdist(fq, fs)                             # 1-NN on support features
    return {"proto": d_proto.argmin(1).cpu().numpy(), "knn": ys[d_knn.argmin(1)].cpu().numpy()}


def evaluate(model, tensor_dict, es, cells, modality_mask, device, moe):
    """-> {readout: {(n, k): per-episode rows}}"""
    model.eval()
    rows = {"proto": {}, "knn": {}}
    for n, k in cells:
        eps = es.select(n, k)
        dl = DataLoader(CachedEpisodeDataset(tensor_dict, eps, modality_mask), batch_size=1,
                        shuffle=False, num_workers=0, collate_fn=maml_mm_collate)
        correct = {"proto": [], "knn": []}
        for batch in dl:
            yq = batch["query"]["labels"].numpy()
            for name, pred in predict_readouts(model, batch, device, moe).items():
                correct[name].append(int((pred == yq).sum()))
        for name in rows:
            rows[name][(n, k)] = outcomes_to_per_episode(eps, correct[name])
    return rows


def main():
    ap = argparse.ArgumentParser(description="A14: ProtoNet/kNN readout on the frozen backbone")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", required=True, choices=["M0", "A4"])
    ap.add_argument("--cells", nargs="+", default=["10x1"])
    add_fold_arg(ap, required=True)
    args = ap.parse_args()
    cells = list(ec.CELLS) if args.cells == ["all"] else [tuple(map(int, c.lower().split("x")))
                                                          for c in args.cells]

    config = apply_kfold_fold(make_base_config(f"A14_{args.model}"), args.fold_idx)
    ckpt = torch.load(args.checkpoint, map_location=config["device"], weights_only=False)
    if ckpt["config"]["fold_idx"] != args.fold_idx:
        raise SystemExit(f"checkpoint is for fold {ckpt['config']['fold_idx']}, not {args.fold_idx}")
    # Rebuild the model from the config the checkpoint was TRAINED with (width, experts, ...).
    build = build_maml_moe_model if ckpt["config"]["use_MOE"] else build_maml_no_moe_model
    if (args.model == "M0") != bool(ckpt["config"]["use_MOE"]):
        raise SystemExit(f"--model {args.model} does not match checkpoint use_MOE")
    model = build(dict(ckpt["config"], device=config["device"]))
    model.load_state_dict(ckpt["model_state_dict"])        # strict: any mismatch raises
    device = torch.device(config["device"])
    model.to(device)
    print(f"[A14] {args.model} fold {args.fold_idx}: {count_parameters(model):,} params from "
          f"{args.checkpoint}")

    with open(tensor_dict_file(config), "rb") as f:
        tensor_dict = reorient_tensor_dict(pickle.load(f), config)
    es = ec.load("test", args.fold_idx)
    assert set(config["test_PIDs"]) == set(es.users)

    rows = evaluate(model, tensor_dict, es, cells, config["modality_mask"], device,
                    moe=bool(ckpt["config"]["use_MOE"]))
    summary = {"ablation_id": f"A14_{args.model}", "test_procedure": "kfold7",
               "fold_idxs": [args.fold_idx], "checkpoint": args.checkpoint}
    for name in rows:
        res = _summarise_eval(es, rows[name])
        print(f"--- readout {name} ---")
        _print_eval(res)
        summary[f"test_{name}"] = res
    save_results(summary, config, tag=f"fold{args.fold_idx:02d}_summary")


if __name__ == "__main__":
    main()
