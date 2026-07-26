"""
V7_checkpoint_param_count.py
============================
Verification V7: confirm WHICH Kaifosh & Reardon [11] checkpoint we loaded.

Why this gates everything else
------------------------------
Their discrete-gesture decoder (Fig. 2f, largest model) is ~6.5M parameters.
The ~60M figure is their HANDWRITING conformer (Fig. 2g). If we loaded the
handwriting model, the A10/A11 rows are measuring the wrong network and the
whole Kaifosh comparison is void.

Two independent checks, because they can disagree
-------------------------------------------------
  CHECK 1 (decisive): sum numel() over the checkpoint's own state_dict.
      This reads the FILE. It does not depend on what architecture we
      instantiate, so it cannot be fooled by a mis-specified constructor.

  CHECK 2: instantiate DiscreteGesturesArchitecture with the arguments used
      in A10_A11_A12_meta_pretrained.py and sum over model.parameters().
      This tells us what OUR code builds.

If CHECK 1 and CHECK 2 disagree, the constructor args in
A10_A11_A12_meta_pretrained.py do not describe the released checkpoint, and
strict=True loading would already have failed — so a disagreement here means
something changed upstream.

Expected, from the constructor args currently in the repo
  (input_channels=16, conv_output_channels=512, kernel_width=21, stride=10,
   lstm_hidden_size=512, lstm_num_layers=3, output_channels=9):

    conv_layer     Conv1d(16, 512, k=21)   16*512*21 + 512  =   172,544
    post_conv_LN   LayerNorm(512)                            =     1,024
    lstm           3 x LSTM(512 -> 512)    3 * 4*(2*512^2+2*512) = 6,303,744
    post_lstm_LN   LayerNorm(512)                            =     1,024
    projection     Linear(512, 9)          512*9 + 9         =     4,617
                                                            -----------
                                                  TOTAL   ~=  6,482,953

So ~6.48M is the PASS condition and ~60M is the FAIL condition.

Run on NOTS (the checkpoint lives there, not locally):
    python V7_checkpoint_param_count.py
    python V7_checkpoint_param_count.py --checkpoint /path/to/other.ckpt
"""

import os
import sys
import argparse
from pathlib import Path
from collections import OrderedDict

import torch

CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "NOTS" / "paper" / "ablations" / "test_eval_files"))

# Keep in sync with A10_A11_A12_meta_pretrained.py:META_CHECKPOINT_PATH
DEFAULT_CHECKPOINT = "/rhf/allocations/my13/emg_models/discrete_gestures/model_checkpoint.ckpt"

# Constructor args must mirror A10_A11_A12_meta_pretrained.py:MetaEMGWrapper.
ARCH_KWARGS = dict(
    input_channels=16,
    conv_output_channels=512,
    kernel_width=21,
    stride=10,
    lstm_hidden_size=512,
    lstm_num_layers=3,
    output_channels=9,
)

GESTURE_DECODER_PARAMS   = 6_500_000     # Fig. 2f, largest discrete-gesture model
HANDWRITING_CONFORMER    = 60_000_000    # Fig. 2g -- the WRONG model
TOLERANCE_FRAC           = 0.15          # +/-15% around the expected count


def human(n: int) -> str:
    return f"{n:,} ({n / 1e6:.3f}M)"


def check_1_state_dict(checkpoint_path: Path) -> int:
    """Sum numel() over the checkpoint file's own tensors. Architecture-independent."""
    print("=" * 72)
    print("CHECK 1 -- parameter count read directly from the checkpoint file")
    print("=" * 72)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in ckpt:
        print(f"  WARNING: no 'state_dict' key. Top-level keys: {list(ckpt.keys())}")
        sd = ckpt if isinstance(ckpt, (dict, OrderedDict)) else {}
    else:
        sd = ckpt["state_dict"]

    total = 0
    by_group: dict = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        n = v.numel()
        total += n
        # group by the module path minus the final .weight/.bias
        group = ".".join(k.split(".")[:-1]) or k
        by_group[group] = by_group.get(group, 0) + n

    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Tensors    : {sum(1 for v in sd.values() if torch.is_tensor(v))}")
    print(f"  TOTAL      : {human(total)}")
    print()
    print("  Breakdown by module (descending):")
    for group, n in sorted(by_group.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / total if total else 0.0
        print(f"    {n:>12,}  ({pct:5.1f}%)  {group}")

    # Report anything that hints at the handwriting model rather than the
    # gesture decoder. Their conformer has attention/conformer blocks; the
    # gesture decoder is conv + LSTM only.
    suspicious = [k for k in sd
                  if any(t in k.lower() for t in
                         ("conformer", "attention", "attn", "self_attn", "transformer"))]
    if suspicious:
        print()
        print("  !! Conformer/attention-style keys present -- this looks like the")
        print("     HANDWRITING model, not the discrete-gesture decoder:")
        for k in suspicious[:12]:
            print(f"       {k}")

    has_lstm = any("lstm" in k.lower() for k in sd)
    print()
    print(f"  Contains LSTM keys : {has_lstm}   (gesture decoder: expected True)")
    return total


def check_2_instantiated() -> int:
    """Sum numel() over the model our code actually builds."""
    print()
    print("=" * 72)
    print("CHECK 2 -- parameter count of the architecture OUR code instantiates")
    print("=" * 72)

    try:
        from generic_neuromotor_interface.networks import DiscreteGesturesArchitecture
    except ImportError as e:
        print(f"  SKIPPED -- could not import DiscreteGesturesArchitecture: {e}")
        print("  Check NEUROMOTOR_REPO / that the Meta repo is installed in this env.")
        return -1

    model = DiscreteGesturesArchitecture(**ARCH_KWARGS)
    total = sum(p.numel() for p in model.parameters())

    print(f"  Constructor args: {ARCH_KWARGS}")
    print(f"  TOTAL           : {human(total)}")
    print()
    print("  Breakdown by top-level child:")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n:
            print(f"    {n:>12,}  {name}  ({child.__class__.__name__})")
    return total


def verdict(ckpt_total: int, arch_total: int) -> int:
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)

    lo = GESTURE_DECODER_PARAMS * (1 - TOLERANCE_FRAC)
    hi = GESTURE_DECODER_PARAMS * (1 + TOLERANCE_FRAC)

    print(f"  Expected (gesture decoder, Fig. 2f) : ~{human(GESTURE_DECODER_PARAMS)}")
    print(f"  Wrong model (handwriting, Fig. 2g)  : ~{human(HANDWRITING_CONFORMER)}")
    print(f"  Accept window                       : {lo:,.0f} -- {hi:,.0f}")
    print()

    if arch_total > 0 and abs(arch_total - ckpt_total) > 0.01 * max(arch_total, 1):
        print(f"  !! MISMATCH between checkpoint ({human(ckpt_total)}) and instantiated")
        print(f"     architecture ({human(arch_total)}). The constructor args in")
        print("     A10_A11_A12_meta_pretrained.py do not describe this checkpoint.")
        print("     Note that strict=True loading should already have failed --")
        print("     investigate before trusting any A10/A11 number.")
        return 2

    if lo <= ckpt_total <= hi:
        print("  PASS -- this is the discrete-gesture decoder.")
        print("         A10/A11 are evaluating the intended network.")
        print("         'Scale alone is insufficient' keeps its premise (pretraining")
        print("         corpus N=4,800, largest model, Fig. 2f).")
        print("         Correct the paper's parameter count from 60M to ~6.5M.")
        return 0

    if ckpt_total > 0.5 * HANDWRITING_CONFORMER:
        print("  FAIL -- parameter count is in the handwriting-conformer range.")
        print("          We loaded the WRONG decoder. A10/A11 are void.")
        print("          Stop and re-download: ")
        print("            python -m generic_neuromotor_interface.scripts.download_models \\")
        print("                   --task discrete_gestures --output-dir ~/emg_models")
        return 1

    print("  UNEXPECTED -- count matches neither published model. Inspect the")
    print("               breakdown above before proceeding.")
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description="V7: verify which Kaifosh checkpoint is loaded.")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        print("This script must run on NOTS -- the checkpoint is not stored in the repo.")
        return 4

    ckpt_total = check_1_state_dict(ckpt_path)
    arch_total = check_2_instantiated()
    return verdict(ckpt_total, arch_total)


if __name__ == "__main__":
    sys.exit(main())
