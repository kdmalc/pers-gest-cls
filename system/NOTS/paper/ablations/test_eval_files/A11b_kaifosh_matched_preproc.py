"""
A11b_kaifosh_matched_preproc.py
===============================
The mandatory Kaifosh re-run: evaluate their pretrained discrete-gesture
decoder with THEIR input pipeline applied, and sweep the one free parameter
that pipeline has.

WHY THIS RUN EXISTS
-------------------
The published 62.1% / 56.0% are out-of-distribution evaluations of their model.
Reading `A10_A11_A12_meta_pretrained.py` and `build_2khz_tensor_dict.ipynb`
together:

  - Their pipeline is: rescale so NOISE s.d. = 1  ->  40 Hz 4th-order
    Butterworth high-pass  ->  Reinhard squash x/(32+|x|)  ->  learned conv.
  - The squash is INSIDE their released module (`net.compression`), so it always
    runs. The rescale and the 40 Hz high-pass are ours to apply, and NEITHER
    was applied.
  - Our 2 kHz tensor is band-passed 20-450 Hz and then divided by a single
    per-trial scalar giving whole-trial SIGNAL s.d. = 1.0.

Their model expects noise s.d. = 1 (so gesture-active samples land well above
1, which is why mu=32 is a sensible outlier knob). We supply signal s.d. = 1,
so gesture-active samples land near 1 and x/(32+|x|) ~= x/32 compresses the
entire recording into roughly +/-0.03. The network is evaluated in the
near-linear corner of a nonlinearity it was trained to use.

NOTE this corrects the earlier hypothesis. There is no literal multiply by
2.46e-6 anywhere in the code, so "dead sigmoid from a literal multiply" is not
what happened. The real defect is systematic UNDER-SCALING by roughly the
signal-to-noise ratio. It biases against their model in the same direction,
but the fix is a gain, and because our data is already s.d.-normalised, a
single global gain g is exactly the hypothesis "noise s.d. = 1/g". Hence a
sweep, not a two-way direction test.

WHAT IT REPORTS
---------------
For each gain condition: linear-probe and full-fine-tune accuracy, plus
pre/post-squash activation statistics so the operating regime is SHOWN rather
than asserted. A condition whose post-squash output is numerically zero, or
whose samples never approach mu, is not a fair measurement and is labelled as
such in the output.

Expect their numbers to come up. If they move materially, soften the sentence
that leans on the size of the gap rather than defending a gap that shrank.
The premise of "scale alone is insufficient" is independent of this -- it rests
on the pretraining corpus (N=4,800, largest model, Fig. 2f), which V7 confirms.

STILL UNCONTROLLED AFTER THIS RUN, and worth stating first
----------------------------------------------------------
  - Electrode topology. Their filters learned a circumferential wrist array
    (20 mm within-pair, 10.6-15 mm between channels); ours are spread across
    the upper body. Their gesture architecture -- unlike their wrist and
    handwriting decoders -- has NO rotational-invariance module, so their
    filters cannot be realigned to our montage even in principle.
  - Analog band. Their front end is 20 Hz HP / 850 Hz LP; our upstream band-pass
    is 20-450 Hz. "Matched preprocessing" is matched DIGITALLY only, and our
    40 Hz stage cascades onto an existing 20 Hz HP rather than replacing it.
  - Head and metric. Their readout is a 9-way multilabel sigmoid detector
    scored by CLER over Needleman-Wunsch-matched events after debouncing and
    state-machine filtering. We replace the head and drop that stack, so this
    measures "their pretrained trunk plus our head", never "their method".

Usage (NOTS):
    python A11b_kaifosh_matched_preproc.py                       # full sweep
    python A11b_kaifosh_matched_preproc.py --gains none 32 noise_floor
    python A11b_kaifosh_matched_preproc.py --ft-modes head_only
    python A11b_kaifosh_matched_preproc.py --no-highpass         # isolate the gain
"""

import os
import sys
import copy
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))
sys.path.insert(0, str(CODE_DIR / "system" / "NOTS" / "paper" / "ablations" / "test_eval_files"))

from ablation_config import (
    set_seeds, FIXED_SEED, run_supervised_test_eval, save_results,
    save_model_checkpoint, count_parameters, replace_head_for_eval,
)
from A10_A11_A12_meta_pretrained import (
    MetaEMGWrapper, build_config_meta, META_CHECKPOINT_PATH, EMG_2KHZ_PKL_PATH,
)
from kaifosh_preprocessing import (
    apply_kaifosh_preprocessing, CompressionActivationLogger, classify_regime,
    KAIFOSH_RESCALE_CONSTANT, DEFAULT_FS,
)

# Log-spaced gains bracketing plausible surface-EMG SNRs, plus the two
# named conditions. "none" reproduces the published number and must stay in
# the sweep so 62.1%/56.0% remain locatable.
DEFAULT_GAINS = ["none", "literal", "noise_floor", 3.0, 10.0, 32.0, 100.0, 316.0]

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class KaifoshPreprocessedWrapper(MetaEMGWrapper):
    """
    MetaEMGWrapper with their steps 1-2 applied inside forward().

    Subclassing rather than wrapping keeps `.head` pointing at
    `network.projection`, so `replace_head_for_eval()` and the existing
    fine-tuning path work unchanged.

    Step 3 (the mu=32 squash) is deliberately NOT applied here -- it lives in
    `network.compression` and applying it twice would compress the range again.
    """

    def __init__(self, checkpoint_path, gain_mode, fs=DEFAULT_FS,
                 do_highpass=True, zero_phase=False, freeze_backbone=True):
        super().__init__(checkpoint_path, freeze_backbone=freeze_backbone)
        self.gain_mode = gain_mode
        self.fs = fs
        self.do_highpass = do_highpass
        self.zero_phase = zero_phase

    def forward(self, x_emg, x_imu=None, demographics=None):
        x_emg = apply_kaifosh_preprocessing(
            x_emg,
            gain_mode=self.gain_mode,
            fs=self.fs,
            do_highpass=self.do_highpass,
            zero_phase=self.zero_phase,
        )
        return super().forward(x_emg, x_imu, demographics)


def gain_label(g) -> str:
    if isinstance(g, str):
        return g
    return f"gain{g:g}"


def run_one_condition(gain, ft_mode, args) -> dict:
    label = gain_label(gain)
    print(f"\n{'#'*70}")
    print(f"# A11b  gain={label}  ft_mode={ft_mode}  highpass={not args.no_highpass}")
    print(f"{'#'*70}")

    config = build_config_meta(ablation_id=f"A11b_{label}_{ft_mode}")
    config["test_procedure"] = "hpo_test_split"
    set_seeds(FIXED_SEED)

    model = KaifoshPreprocessedWrapper(
        META_CHECKPOINT_PATH,
        gain_mode=gain,
        fs=args.fs,
        do_highpass=not args.no_highpass,
        zero_phase=args.zero_phase,
        freeze_backbone=(ft_mode == "head_only"),
    )
    model = replace_head_for_eval(model, config)
    model.to(config["device"])

    n_params = count_parameters(model)
    print(f"[A11b] Parameters: {n_params:,}")
    if n_params > 20_000_000:
        print("[A11b] WARNING: parameter count is far above the ~6.5M expected for")
        print("       their discrete-gesture decoder. Run V7_checkpoint_param_count.py")
        print("       before trusting this row.")

    # Attach the activation logger to their internal squash.
    logger = CompressionActivationLogger(model.network.compression)

    try:
        test_results = run_supervised_test_eval(
            model, config,
            tensor_dict_path=EMG_2KHZ_PKL_PATH,
            test_pids=config["test_PIDs"],
            ft_mode=ft_mode,
        )
        activations = logger.report(label=f"gain={label}, ft={ft_mode}")
    finally:
        logger.remove()

    # Flag conditions that are not fair measurements, so a number from a dead,
    # under-scaled or saturated network never gets quoted as their performance.
    # Uses the shared classifier in kaifosh_preprocessing so the printed verdict
    # and the saved JSON cannot disagree.
    is_fair, invalid_reason = classify_regime(activations)

    result = {
        "ablation_id":     f"A11b_{label}_{ft_mode}",
        "description":     ("Kaifosh discrete-gesture decoder with their input pipeline "
                            "applied (gain + 40 Hz Butterworth HP); their mu=32 squash "
                            "is internal to the module."),
        "gain_mode":       label,
        "gain_value":      (None if isinstance(gain, str) else float(gain)),
        "literal_constant_used": (KAIFOSH_RESCALE_CONSTANT if gain == "literal" else None),
        "highpass_applied": not args.no_highpass,
        "highpass_hz":     40.0,
        "highpass_order":  4,
        "highpass_zero_phase": args.zero_phase,
        "ft_mode":         ft_mode,
        "n_params":        n_params,
        "test_results":    test_results,
        "test_acc":        test_results["mean_acc"],
        "activations":     activations,
        "valid_measurement": is_fair,
        "invalid_reason":  invalid_reason,
        "uncontrolled_confounds": [
            "Electrode topology: their filters learned a circumferential wrist "
            "array; ours spans the upper body. Their gesture architecture has no "
            "rotational-invariance module, so realignment is impossible in principle.",
            "Analog band: their front end is 20 Hz HP / 850 Hz LP; our data is "
            "band-passed 20-450 Hz upstream, so the 40 Hz stage cascades rather "
            "than replaces. Matched digitally only.",
            "Head and metric: their 9-way multilabel sigmoid detector, scored by "
            "CLER over Needleman-Wunsch-matched events after debouncing and "
            "state-machine filtering, is replaced by our N-way softmax. This row "
            "measures their pretrained trunk plus our head.",
        ],
        "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(result, config, tag=f"A11b_{label}_{ft_mode}")

    flag = "" if is_fair else f"   [NOT A FAIR MEASUREMENT: {invalid_reason}]"
    print(f"[A11b] {label} / {ft_mode}: {test_results['mean_acc']*100:.2f}% "
          f"± {test_results['std_acc']*100:.2f}%{flag}")
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Kaifosh baseline re-run with their preprocessing + gain sweep.")
    ap.add_argument("--gains", nargs="+", default=None,
                    help=f"Gain conditions. Default: {DEFAULT_GAINS}")
    ap.add_argument("--ft-modes", nargs="+", default=["head_only", "full"],
                    choices=["head_only", "full"])
    ap.add_argument("--fs", type=float, default=DEFAULT_FS)
    ap.add_argument("--no-highpass", action="store_true",
                    help="Skip the 40 Hz HP to isolate the effect of the gain alone.")
    ap.add_argument("--zero-phase", action="store_true",
                    help="Use sosfiltfilt instead of causal sosfilt. Their model is "
                         "streaming, so causal (the default) is the faithful choice.")
    args = ap.parse_args()

    gains = []
    for g in (args.gains if args.gains is not None else DEFAULT_GAINS):
        if isinstance(g, str) and g in ("none", "literal", "noise_floor"):
            gains.append(g)
        else:
            gains.append(float(g))

    all_results = []
    for gain in gains:
        for ft_mode in args.ft_modes:
            try:
                all_results.append(run_one_condition(gain, ft_mode, args))
            except Exception as e:
                print(f"[A11b] FAILED gain={gain_label(gain)} ft={ft_mode}: "
                      f"{type(e).__name__}: {e}")
                all_results.append({
                    "gain_mode": gain_label(gain), "ft_mode": ft_mode,
                    "error": f"{type(e).__name__}: {e}",
                })

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*88}")
    print("A11b SUMMARY -- Kaifosh decoder under their preprocessing")
    print(f"{'='*88}")
    print(f"{'gain':>14}  {'ft_mode':>10}  {'acc':>9}  {'post|x|p99':>11}  "
          f"{'frac>mu/10':>11}  valid")
    print("-" * 88)
    valid = []
    for r in all_results:
        if "error" in r:
            print(f"{r['gain_mode']:>14}  {r['ft_mode']:>10}  {'ERROR':>9}  "
                  f"{'-':>11}  {'-':>11}  -   {r['error'][:28]}")
            continue
        post = r.get("activations", {}).get("post", {})
        pre = r.get("activations", {}).get("pre", {})
        print(f"{r['gain_mode']:>14}  {r['ft_mode']:>10}  "
              f"{r['test_acc']*100:8.2f}%  "
              f"{post.get('abs_p99', float('nan')):11.4g}  "
              f"{pre.get('frac_above_mu_10', float('nan')):11.4g}  "
              f"{'yes' if r['valid_measurement'] else 'NO'}")
        if r["valid_measurement"]:
            valid.append(r)

    print("-" * 88)
    if valid:
        best = max(valid, key=lambda r: r["test_acc"])
        print(f"\nBest VALID condition: gain={best['gain_mode']}, "
              f"ft={best['ft_mode']}, acc={best['test_acc']*100:.2f}%")
        print("This is the number to quote for their model, not the 'none' row.")
        print("If it is materially above the published 62.1%/56.0%, soften any")
        print("sentence whose force depends on the SIZE of the gap.")
    else:
        print("\nNo condition produced a fair measurement. Do not quote any number")
        print("from this run; investigate the pipeline before reporting.")

    print("\nReminder: matched preprocessing does not remove the electrode-topology,")
    print("analog-band, or head/metric confounds. State those first, then reframe")
    print("the row as a transfer study.")


if __name__ == "__main__":
    main()
