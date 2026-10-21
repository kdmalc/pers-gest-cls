"""
Peak GPU memory of one M0 forward+backward: dense stack (old) vs running sum (new).

    python system/MOE/debug_peak_memory.py [--T 64 512 4300] [--batch 100] [--experts 22]

Uses the real M0 config (22 experts, top-9, 88 input channels) and random inputs; needs a CUDA
GPU. Prints one line per (T, implementation) and the ratio. Cluster: run on the debug partition.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CODE_DIR", str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("RUN_DIR", str(Path(os.environ.get("TEMP", "/tmp")) / "pgc-peakmem"))
for rel in ("", "system", "system/MAML", "system/MOE", "system/pretraining",
            "system/NOTS/paper/ablations/test_eval_files"):
    sys.path.insert(0, str(Path(os.environ["CODE_DIR"]) / rel))

import torch  # noqa: E402

import MOE.MOE_encoder as me  # noqa: E402
from ablation_config import build_maml_moe_model, make_base_config  # noqa: E402


def dense_stack(calls, w_hard):
    outs = [c() for c in calls]
    stacked = torch.stack(outs, dim=1)
    w = w_hard.reshape(w_hard.shape[0], w_hard.shape[1], *([1] * (outs[0].dim() - 1)))
    return (stacked * w).sum(dim=1)


def peak_mb(model, B, C, T, impl):
    me._weighted_expert_sum = impl
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(B, C, T, device="cuda")
    out = model(x[:, :16], x[:, 16:])
    (out[0] if isinstance(out, tuple) else out).sum().backward()
    torch.cuda.synchronize()
    m = torch.cuda.max_memory_allocated() / 2**20
    model.zero_grad(set_to_none=True)
    del out, x
    return m


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[64, 512, 2048])
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--experts", type=int, default=None)
    args = ap.parse_args()

    cfg = make_base_config("peakmem")
    cfg["device"] = "cuda"
    if args.experts is not None:
        cfg["num_experts"] = args.experts
    model = build_maml_moe_model(cfg).cuda().train()
    print(f"GPU {torch.cuda.get_device_name(0)}  E={cfg['num_experts']} top_k={cfg['MOE_top_k']} "
          f"B={args.batch}")
    new_impl = me._weighted_expert_sum
    for T in args.T:
        row = {}
        for name, impl in (("dense_stack", dense_stack), ("running_sum", new_impl)):
            try:
                row[name] = peak_mb(model, args.batch, 88, T, impl)
            except torch.cuda.OutOfMemoryError:
                row[name] = float("inf")
                torch.cuda.empty_cache()
        ratio = row["dense_stack"] / row["running_sum"] if row["running_sum"] != float("inf") else 0
        print(f"T={T:5d}  dense_stack={row['dense_stack']:9.1f} MB   "
              f"running_sum={row['running_sum']:9.1f} MB   ratio={ratio:.2f}x")


if __name__ == "__main__":
    main()
