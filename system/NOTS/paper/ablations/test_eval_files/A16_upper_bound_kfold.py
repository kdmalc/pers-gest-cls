"""
A16_upper_bound_kfold.py
========================
Reviewer ARaC's upper bound (Q2, R8), on the kfold7 protocol: take the fold's already-trained
parameter-matched model and fine-tune it on (nearly) ALL of a test user's data, then test on data it never
trained on. This replaces the L2SO-era A16_upper_bound.py, whose design it keeps:

  * no retraining: the SAME fold's checkpoint (--checkpoint; that fold never saw its test users);
  * leave-one-rep-out: for each of the 10 reps, the other 9 reps x 10 gestures (90 samples, global
    labels, fresh 10-way head) are the adaptation set and the held-out rep x 10 gestures is the test set;
    per-user accuracy = mean over the 10 held-out reps;
  * a properly tuned Adam fine-tune (full network), with ft_lr x ft_steps chosen by grid on the fold's
    VAL users only (--tune-reps held-out reps each), then frozen for the test users.

Bases: --base A2 (supervised CNN-LSTM, what ARaC asked about) or M0 (EncoderMoE plain-fine-tuned: how close
is 1-shot adaptation to this model's own ceiling).

    python A16_upper_bound_kfold.py --base M0 --fold-idx 0 --checkpoint <fold-0 M0 *_best.pt>
kfold_launcher.sh tokens: A16_A2, A16_M0 (after that fold's A2 / M0 job).
"""

import os, sys, copy, argparse, pickle, itertools
import numpy as np
import torch

from pathlib import Path
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
for _p in ("", "system", "system/MAML", "system/MOE", "system/pretraining"):
    sys.path.insert(0, str(CODE_DIR / _p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Z_zeroshot_crossuser import collect, load_encoder, ALL_REPS  # noqa: E402

FT_LR_GRID = (1e-4, 3e-4, 1e-3, 3e-3)
FT_STEPS_GRID = (50, 100, 200)


def lo1o_user(model, cfg: dict, data: dict, user: str, held_out_reps) -> float:
    """Mean accuracy over held-out reps for one user (fine-tune on the other 9 reps each time)."""
    from pretraining.pretrain_finetune import finetune_and_eval_user
    m = data["user"] == user
    emg, imu, y, rep = (torch.from_numpy(data["emg"][m]), torch.from_numpy(data["imu"][m]),
                        torch.from_numpy(data["y"][m]).long(), data["rep"][m])
    accs = []
    for r in held_out_reps:
        q = rep == r
        if q.sum() != 10 or (~q).sum() != 90:
            raise ValueError(f"{user} rep {r}: {int(q.sum())} query / {int((~q).sum())} support samples")
        out = finetune_and_eval_user(model, cfg,
                                     support_emg=emg[~q], support_imu=imu[~q], support_labels=y[~q],
                                     query_emg=emg[q], query_imu=imu[q], query_labels=y[q], mode="full")
        accs.append(out["acc"])
    return float(np.mean(accs))


def ft_config(cfg: dict, lr: float, steps: int) -> dict:
    c = copy.deepcopy(cfg)
    c.update(ft_lr=lr, ft_steps=steps, ft_optimizer="adam", ft_weight_decay=0.0, ft_label_smooth=0.0,
             n_way=10)
    return c


def main():
    from ablation_config import (apply_kfold_fold, add_fold_arg, add_seed_arg, add_smoke_arg, make_base_config,
                                 apply_smoke, save_results, set_seeds, tensor_dict_file, kfold_tag)
    from MAML.maml_data_pipeline import reorient_tensor_dict
    ap = argparse.ArgumentParser(description="A16 upper bound, kfold7, leave-one-rep-out")
    ap.add_argument("--base", choices=["A2", "M0"], required=True)
    ap.add_argument("--checkpoint", required=True, help="the SAME fold's checkpoint of --base")
    ap.add_argument("--tune-reps", type=int, default=3, help="held-out reps per val user in the grid")
    add_fold_arg(ap, required=True)
    add_seed_arg(ap)
    add_smoke_arg(ap)
    args = ap.parse_args()
    set_seeds(args.seed)

    config = apply_kfold_fold(apply_smoke(make_base_config(ablation_id=f"A16_{args.base}"), args.smoke),
                              args.fold_idx)
    model, ck_cfg = load_encoder(args.checkpoint, config["device"])
    if ck_cfg.get("fold_idx") != args.fold_idx or ck_cfg["emg_repr"] != config["emg_repr"]:
        raise ValueError(f"checkpoint is fold {ck_cfg.get('fold_idx')} / {ck_cfg['emg_repr']}; this run is fold "
                         f"{args.fold_idx} / {config['emg_repr']}")
    if set(ck_cfg["test_PIDs"]) != set(config["test_PIDs"]):
        raise ValueError("checkpoint's test users differ from this fold's")
    ck_cfg["device"] = config["device"]

    with open(tensor_dict_file(config), "rb") as f:
        tensor_dict = reorient_tensor_dict(pickle.load(f), config)
    val = collect(tensor_dict, config["val_PIDs"])
    test = collect(tensor_dict, config["test_PIDs"])

    lr_grid, steps_grid = (FT_LR_GRID[:2], FT_STEPS_GRID[:1]) if args.smoke else (FT_LR_GRID, FT_STEPS_GRID)
    tune_reps = ALL_REPS[: args.tune_reps]
    grid = {}
    for lr, steps in itertools.product(lr_grid, steps_grid):
        c = ft_config(ck_cfg, lr, steps)
        grid[f"lr{lr:g}_steps{steps}"] = float(np.mean(
            [lo1o_user(model, c, val, u, tune_reps) for u in config["val_PIDs"]]))
        print(f"[A16 {args.base} f{args.fold_idx}] val lr={lr:g} steps={steps}: "
              f"{grid[f'lr{lr:g}_steps{steps}']*100:.2f}%")
    best = max(grid, key=grid.get)
    lr, steps = float(best.split("_")[0][2:]), int(best.split("steps")[1])
    c = ft_config(ck_cfg, lr, steps)
    per_user = {u: lo1o_user(model, c, test, u, ALL_REPS) for u in config["test_PIDs"]}
    mean = float(np.mean(list(per_user.values())))
    print(f"[A16 {args.base} f{args.fold_idx}] TEST (lr={lr:g}, steps={steps}, 9 reps adaptation): "
          f"{mean*100:.2f}% over users {per_user}")
    save_results({"ablation_id": f"A16_{args.base}", "base": args.base, "fold_idx": args.fold_idx,
                  "seed": args.seed, "test_procedure": "kfold7",
                  "task": "10-way, leave-one-rep-out: fine-tune on 9 reps x 10 gestures, test the 10th rep",
                  "checkpoint": args.checkpoint, "val_grid": grid, "chosen": {"ft_lr": lr, "ft_steps": steps},
                  "per_user_acc": per_user, "mean_acc": mean,
                  "std_acc": float(np.std(list(per_user.values())))},
                 config, tag=kfold_tag(args.fold_idx, args.seed))


if __name__ == "__main__":
    main()
