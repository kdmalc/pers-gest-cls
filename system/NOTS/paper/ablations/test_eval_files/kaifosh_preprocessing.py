"""
kaifosh_preprocessing.py
========================
Kaifosh & Reardon [11] discrete-gesture input pipeline, and the diagnostics
needed to tell whether our data reaches their model in the regime it was
trained for.

WHAT WENT WRONG BEFORE (read this first)
----------------------------------------
The rebuttal plan hypothesised a "dead sigmoid from a literal multiply by
2.46e-6". Reading the code, that is NOT what happened -- there is no multiply
anywhere. The actual situation is different and worth stating precisely.

Their pipeline, from their Methods:
    1. rescale so the NOISE s.d. is 1.0   (their constant 2.46e-6 is calibrated
       to their own 2.46 uVrms input-referred noise floor)
    2. 40 Hz 4th-order Butterworth high-pass
    3. Reinhard squash  f(x) = x / (mu + |x|),  mu = 32
    4. learned strided conv (stride 10, 2 kHz -> 200 Hz) -- part of the model

Step 3 lives INSIDE their released module (`net.compression`), so it is applied
whether or not we ask for it. Steps 1 and 2 are ours to apply, and neither was.

What our 2 kHz tensor actually contains, from
`system/universal_preprocessing/EMG_preprocessing/build_2khz_tensor_dict.ipynb`:
    - band-pass 20-450 Hz + mean subtraction  (not their 40 Hz HP; and 450 Hz
      vs the 850 Hz analog low-pass on their front end)
    - `normalize_gestures_by_std_any_channels`, which divides each trial by a
      SINGLE scalar computed over all 16 channels flattened, giving
      whole-trial SIGNAL s.d. = 1.0

So the mismatch is a units mismatch, and it is one-directional:

    they expect   NOISE  s.d. = 1  ->  active-gesture samples land well above 1,
                                       which is exactly why mu = 32 is a
                                       sensible outlier knob
    we supply     SIGNAL s.d. = 1  ->  active-gesture samples land near 1,
                                       so x/(32+|x|) ~= x/32 and the whole
                                       recording is compressed into roughly
                                       +/-0.03

The network is therefore evaluated in the near-linear, near-zero corner of a
nonlinearity whose operating range it was trained on. That still biases against
their model, as the plan suspected -- but via systematic UNDER-SCALING by
roughly the signal-to-noise ratio, not via a literal multiply. The fix is a
gain, and because our data is already s.d.-normalised, a single global gain g
is exactly equivalent to asserting "noise s.d. = 1/g". That makes the honest
experiment a GAIN SWEEP rather than a two-way direction test.

GAIN MODES
----------
  "none"        : current behaviour. The baseline being corrected. Keep it in
                  the sweep so the published 62.1%/56.0% remain locatable.
  "literal"     : multiply by 2.46e-6 verbatim -- the "as published, taken
                  literally" reading. Expected to zero the input. Included
                  because a reviewer may ask what the literal reading gives.
  "noise_floor" : divide by a per-channel noise-floor estimate (see
                  `estimate_noise_floor`). Closest to their stated INTENT.
  float         : a fixed global gain, e.g. 32.0. Use for the sweep.

Nothing here mutates its input.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from scipy.signal import butter, sosfilt, sosfiltfilt
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


KAIFOSH_RESCALE_CONSTANT = 2.46e-6   # their published constant
KAIFOSH_HP_CUTOFF_HZ     = 40.0
KAIFOSH_HP_ORDER         = 4
KAIFOSH_SQUASH_MU        = 32.0      # applied INSIDE their module, not here
DEFAULT_FS               = 2000.0

# Post-squash operating-range thresholds used to decide whether a condition is
# a fair measurement of their model. Judgement calls, stated explicitly so they
# can be argued with rather than buried:
#   below UNDERSCALED_P99 -> squash is a near-constant 1/mu scale (the defect)
#   above SATURATED_P99   -> squash is discarding amplitude information
UNDERSCALED_P99 = 0.15
SATURATED_P99   = 0.90


# ---------------------------------------------------------------------------
# Noise-floor estimation
# ---------------------------------------------------------------------------

def estimate_noise_floor(x: torch.Tensor,
                         fs: float = DEFAULT_FS,
                         win_ms: float = 100.0,
                         hop_ms: float = 50.0,
                         percentile: float = 10.0) -> torch.Tensor:
    """
    Per-channel noise-floor estimate for (B, C, T) input.

    Their constant normalises a measured QUIESCENT noise floor to 1.0. Our
    trials are segmented around gestures, so we have no dedicated rest
    recording. We use the quietest part of each trial as a proxy: slide a
    window, take per-channel RMS in each, then take a low percentile across
    windows.

    ASSUMPTION, and it is load-bearing: every trial contains at least one
    ~100 ms window that is close to quiescent. For segmented gesture trials
    with pre/post padding this is usually true, but it is an assumption and
    should be stated wherever a noise_floor number is reported. If gestures
    fill the entire trial the estimate is biased HIGH, which under-gains and
    therefore still under-states their model -- i.e. it fails safe in the
    same direction as the existing bug, so do not treat it as conservative.

    Returns (B, C, 1), broadcastable against x.
    """
    assert x.dim() == 3, f"expected (B, C, T), got {tuple(x.shape)}"
    B, C, T = x.shape

    win = max(1, int(round(win_ms * fs / 1000.0)))
    hop = max(1, int(round(hop_ms * fs / 1000.0)))
    if T < win:
        # Trial shorter than one window: fall back to whole-trial RMS.
        return x.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)

    # (B, C, n_win, win)
    frames = x.float().unfold(dimension=-1, size=win, step=hop)
    rms = frames.pow(2).mean(dim=-1).sqrt()          # (B, C, n_win)

    q = float(percentile) / 100.0
    floor = torch.quantile(rms, q, dim=-1, keepdim=True)   # (B, C, 1)
    return floor.clamp_min(1e-12)


# ---------------------------------------------------------------------------
# High-pass filter
# ---------------------------------------------------------------------------

def butter_highpass_sos(cutoff_hz: float = KAIFOSH_HP_CUTOFF_HZ,
                        fs: float = DEFAULT_FS,
                        order: int = KAIFOSH_HP_ORDER):
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required for the Butterworth high-pass.")
    return butter(order, cutoff_hz / (0.5 * fs), btype="highpass", output="sos")


def apply_highpass(x: torch.Tensor,
                   cutoff_hz: float = KAIFOSH_HP_CUTOFF_HZ,
                   fs: float = DEFAULT_FS,
                   order: int = KAIFOSH_HP_ORDER,
                   zero_phase: bool = False) -> torch.Tensor:
    """
    Butterworth high-pass along the time axis of (B, C, T).

    zero_phase=False (default) uses causal `sosfilt`, which is what their
    STREAMING model saw at training time. `sosfiltfilt` is zero-phase and
    non-causal; it produces slightly cleaner signals but is not what their
    network was trained on, so it is not the default. Report which was used.

    Note our data has already been band-passed 20-450 Hz upstream, so this is
    a second-stage filter, not the only one. Cascading a 20 Hz HP with a 40 Hz
    HP is not identical to a single 40 Hz HP -- state that rather than claiming
    exactly matched preprocessing.
    """
    sos = butter_highpass_sos(cutoff_hz, fs, order)
    arr = x.detach().cpu().numpy().astype(np.float64)
    filt = sosfiltfilt(sos, arr, axis=-1) if zero_phase else sosfilt(sos, arr, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(filt)).to(device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------------------
# Gain
# ---------------------------------------------------------------------------

def apply_gain(x: torch.Tensor, gain_mode, fs: float = DEFAULT_FS,
               return_gain: bool = False):
    """
    Apply step 1 of their pipeline. See module docstring for the mode meanings.
    """
    if gain_mode is None or gain_mode == "none":
        g = torch.ones(1, device=x.device, dtype=x.dtype)
        out = x
    elif gain_mode == "literal":
        g = torch.full((1,), KAIFOSH_RESCALE_CONSTANT, device=x.device, dtype=x.dtype)
        out = x * KAIFOSH_RESCALE_CONSTANT
    elif gain_mode == "noise_floor":
        floor = estimate_noise_floor(x, fs=fs).to(x.dtype)   # (B, C, 1)
        g = 1.0 / floor
        out = x * g
    else:
        try:
            gf = float(gain_mode)
        except (TypeError, ValueError):
            raise ValueError(
                f"gain_mode must be 'none' | 'literal' | 'noise_floor' | float, "
                f"got {gain_mode!r}"
            )
        g = torch.full((1,), gf, device=x.device, dtype=x.dtype)
        out = x * gf

    return (out, g) if return_gain else out


def apply_kaifosh_preprocessing(x: torch.Tensor,
                                gain_mode="noise_floor",
                                fs: float = DEFAULT_FS,
                                do_highpass: bool = True,
                                zero_phase: bool = False) -> torch.Tensor:
    """
    Steps 1-2 of their pipeline on (B, C, T). Step 3 (the mu=32 Reinhard
    squash) is inside their module and must NOT be applied here -- applying it
    twice silently halves the dynamic range again.
    """
    out = apply_gain(x, gain_mode, fs=fs)
    if do_highpass:
        out = apply_highpass(out, fs=fs, zero_phase=zero_phase)
    return out


# ---------------------------------------------------------------------------
# Activation logging around their internal squash
# ---------------------------------------------------------------------------

def reinhard(x: torch.Tensor, mu: float = KAIFOSH_SQUASH_MU) -> torch.Tensor:
    """Their compression, reimplemented for diagnostics only."""
    return x / (mu + x.abs())


def summarise(t: torch.Tensor, mu: float = KAIFOSH_SQUASH_MU) -> dict:
    """Scalar summary of a tensor's operating range."""
    f = t.detach().float()
    a = f.abs()
    return {
        "mean":            float(f.mean()),
        "std":             float(f.std()),
        "abs_mean":        float(a.mean()),
        "abs_p50":         float(a.median()),
        "abs_p99":         float(torch.quantile(a.flatten().float(), 0.99)),
        "abs_max":         float(a.max()),
        # Fraction of samples large enough for the squash to be meaningfully
        # nonlinear. If this is ~0 the network is operating in its linear
        # corner and the comparison is invalid.
        "frac_above_mu":   float((a > mu).float().mean()),
        "frac_above_mu_10": float((a > mu / 10.0).float().mean()),
    }


# ---------------------------------------------------------------------------
# Module-level activation accumulator
# ---------------------------------------------------------------------------
#
# WHY NOT A FORWARD HOOK.  The first version of this file registered a forward
# hook on the model's `compression` module. It silently recorded nothing, and
# every A11b row came back with "activations": {} and
# valid_measurement: false.
#
# Cause: pretrain_finetune.finetune_and_eval_user does
#     ft_model = copy.deepcopy(model)
# once per episode. copy.deepcopy copies a module's _forward_hooks AND deep-copies
# the hook callable's closure, so each episode got its own private copy of the
# logger and appended into that copy's list. The original logger the runner held a
# reference to was never touched.
#
# A module-level list is immune: module globals are not copied by deepcopy, so
# every cloned model appends to this one list.
#
# Recording happens in KaifoshPreprocessedWrapper.forward rather than in a hook,
# which is equivalent -- `pre` is exactly the tensor handed to their compression,
# and `post` is reinhard(pre) -- and does not depend on hook survival.

_ACT_RECORDS: list = []
_ACT_MAX_RECORDS = 32          # cap: quantiles over (B,16,4300) are not free
_ACT_SUBSAMPLE = 200_000       # elements sampled per tensor for the stats


def reset_activations() -> None:
    """Call before each condition, or stats bleed across gain settings."""
    _ACT_RECORDS.clear()


def record_activations(pre: torch.Tensor, mu: float = KAIFOSH_SQUASH_MU) -> None:
    """Record pre/post-squash stats for one batch. Cheap and bounded."""
    if len(_ACT_RECORDS) >= _ACT_MAX_RECORDS:
        return
    with torch.no_grad():
        flat = pre.detach().float().flatten()
        if flat.numel() > _ACT_SUBSAMPLE:
            idx = torch.randint(0, flat.numel(), (_ACT_SUBSAMPLE,), device=flat.device)
            flat = flat[idx]
        _ACT_RECORDS.append({
            "pre":  summarise(flat, mu=mu),
            "post": summarise(reinhard(flat, mu=mu), mu=mu),
        })


def get_activation_summary() -> dict:
    """Aggregate to the same shape CompressionActivationLogger.aggregate() returned."""
    if not _ACT_RECORDS:
        return {}
    out = {}
    for side in ("pre", "post"):
        keys = _ACT_RECORDS[0][side].keys()
        out[side] = {k: float(np.mean([r[side][k] for r in _ACT_RECORDS]))
                     for k in keys}
    out["n_batches"] = len(_ACT_RECORDS)
    return out


def report_activations(label: str = "") -> dict:
    """Print the verdict for the accumulated stats. Returns the aggregate."""
    agg = get_activation_summary()
    if not agg:
        print(f"[activations{' ' + label if label else ''}] NO BATCHES RECORDED -- "
              f"the wrapper never called record_activations(). Check that "
              f"KaifoshPreprocessedWrapper.forward is the forward actually used.")
        return agg
    pre, post = agg["pre"], agg["post"]
    print(f"\n  --- compression activations {label} ({agg['n_batches']} batches) ---")
    print(f"    PRE-squash   abs_mean={pre['abs_mean']:.4g}  "
          f"abs_p99={pre['abs_p99']:.4g}  abs_max={pre['abs_max']:.4g}")
    print(f"                 frac |x| > mu(32)  = {pre['frac_above_mu']:.4f}")
    print(f"    POST-squash  abs_mean={post['abs_mean']:.4g}  "
          f"abs_p99={post['abs_p99']:.4g}  abs_max={post['abs_max']:.4g}")
    fair, why = classify_regime(agg)
    print(f"    VERDICT: {'FAIR MEASUREMENT' if fair else 'NOT FAIR -- ' + str(why)}")
    return agg


class CompressionActivationLogger:
    """
    Forward hook on their `compression` module, recording input and output
    statistics so we can SHOW which regime the network is in rather than
    asserting it.

    Usage:
        logger = CompressionActivationLogger(wrapper.network.compression)
        ... run eval ...
        logger.report("gain=32")
        logger.remove()
    """

    def __init__(self, compression_module, max_batches: int = 64):
        self.records: list = []
        self.max_batches = max_batches
        self._handle = compression_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if len(self.records) >= self.max_batches:
            return
        x_in = inputs[0] if isinstance(inputs, tuple) else inputs
        self.records.append({
            "pre":  summarise(x_in),
            "post": summarise(output),
        })

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def aggregate(self) -> dict:
        if not self.records:
            return {}
        out = {}
        for side in ("pre", "post"):
            keys = self.records[0][side].keys()
            out[side] = {k: float(np.mean([r[side][k] for r in self.records]))
                         for k in keys}
        out["n_batches"] = len(self.records)
        return out

    def report(self, label: str = "") -> dict:
        agg = self.aggregate()
        if not agg:
            print(f"[activations{' ' + label if label else ''}] no batches recorded.")
            return agg

        pre, post = agg["pre"], agg["post"]
        print(f"\n  --- compression activations {label} "
              f"({agg['n_batches']} batches) ---")
        print(f"    PRE-squash   abs_mean={pre['abs_mean']:.4g}  "
              f"abs_p99={pre['abs_p99']:.4g}  abs_max={pre['abs_max']:.4g}")
        print(f"                 frac |x| > mu(32)   = {pre['frac_above_mu']:.4f}")
        print(f"                 frac |x| > mu/10    = {pre['frac_above_mu_10']:.4f}")
        print(f"    POST-squash  abs_mean={post['abs_mean']:.4g}  "
              f"abs_p99={post['abs_p99']:.4g}  abs_max={post['abs_max']:.4g}")

        # Interpretation, so the number is not left to be misread.
        #
        # The discriminating quantity is the POST-squash operating range, not
        # the fraction of pre-squash samples above some multiple of mu. Their
        # network was trained with noise s.d. = 1, which puts gesture-active
        # samples well above mu and produces post-squash values spanning a good
        # fraction of the available +/-1. If our post-squash p99 sits far below
        # that, the squash is acting as a near-constant 1/mu scale and the
        # network is off-distribution regardless of how many samples technically
        # exceed mu/10.
        #
        # Thresholds are judgement calls, not theory. They are set so that the
        # uncorrected pipeline (whole-trial s.d. = 1, no gain) is flagged: it
        # lands near p99 ~ 0.1, roughly 5x below a noise-floor-normalised input.
        p99 = post["abs_p99"]
        if post["abs_max"] < 1e-6:
            print("    VERDICT: DEAD. Post-squash output is numerically zero; this")
            print("             network cannot classify anything. Any accuracy here")
            print("             is chance plus head bias. Do not report it.")
        elif p99 < UNDERSCALED_P99:
            print(f"    VERDICT: UNDER-SCALED (post-squash p99 = {p99:.3g} < "
                  f"{UNDERSCALED_P99}).")
            print("             The squash is acting as a near-constant 1/mu scale, so")
            print("             the model is operating in the linear corner of a")
            print("             nonlinearity it was trained to use. This is the defect")
            print("             the gain sweep exists to correct -- the resulting")
            print("             accuracy is a floor, not a fair measurement.")
        elif p99 > SATURATED_P99:
            print(f"    VERDICT: SATURATED (post-squash p99 = {p99:.3g} > "
                  f"{SATURATED_P99}). Gain is too high and the squash is")
            print("             discarding amplitude information.")
        else:
            print(f"    VERDICT: PLAUSIBLE operating range (post-squash p99 = "
                  f"{p99:.3g}) -- the squash is doing")
            print("             non-trivial compression without saturating.")
        return agg


def classify_regime(activations: dict) -> tuple:
    """
    (is_fair_measurement, reason) for an aggregate from
    CompressionActivationLogger.aggregate(). Single source of truth so the
    logger's printed verdict and the runner's saved JSON cannot disagree.
    """
    post = activations.get("post", {})
    if not post:
        return False, "no activations recorded"
    if post.get("abs_max", 1.0) < 1e-6:
        return False, "dead: post-squash output is numerically zero"
    p99 = post.get("abs_p99", 0.0)
    if p99 < UNDERSCALED_P99:
        return False, (f"under-scaled: post-squash p99 = {p99:.3g}, below "
                       f"{UNDERSCALED_P99}; squash is in its linear corner")
    if p99 > SATURATED_P99:
        return False, (f"saturated: post-squash p99 = {p99:.3g}, above "
                       f"{SATURATED_P99}; amplitude information is being discarded")
    return True, None
