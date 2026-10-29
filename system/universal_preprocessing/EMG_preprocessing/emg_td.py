"""
EMG-TD64: classical Hudgins time-domain features as a 64-step windowed sequence, for the classical
LDA / kNN baselines reviewers ask for (docs/SPEC_group_analysis.md sibling: docs/SPEC_2khz_emg.md §4).

Same pipeline, windows and step alignment as EMG-FS64 (emg_fs64.py); only the features differ:
per channel per window (F = 4): log MAV, log WL (mean absolute first difference), ZC rate and SSC rate
(counts per sample, so they do not depend on the window length, which varies with trial length).

ZC / SSC thresholds are in units of the trial's block-normalised std (std over all channels == 1) and
are FIXED a priori (not tuned): ZC counts sign changes with |x[i] - x[i+1]| >= ZC_THR; SSC counts
slope sign changes with (x[i] - x[i-1]) * (x[i] - x[i+1]) >= SSC_THR ** 2.
"""
import numpy as np

import emg_fs64 as F

N_FEATURES = 4
FEATURE_NAMES = ("log_mav", "log_wl", "zc_rate", "ssc_rate")
ZC_THR = 0.1
SSC_THR = 0.1


def td_features(win: np.ndarray, zc_thr: float = ZC_THR, ssc_thr: float = SSC_THR) -> np.ndarray:
    """win: (C, n_steps, W) -> (n_steps, C * N_FEATURES), channel-major."""
    W = win.shape[-1]
    a, b = win[..., :-1], win[..., 1:]
    mav = np.abs(win).mean(-1)
    wl = np.abs(b - a).mean(-1)
    zc = (((a * b) < 0) & (np.abs(b - a) >= zc_thr)).sum(-1) / W
    left, right = win[..., 1:-1] - win[..., :-2], win[..., 1:-1] - win[..., 2:]
    ssc = ((left * right) >= ssc_thr ** 2).sum(-1) / W
    feats = np.stack([F.safe_log(mav), F.safe_log(wl), zc, ssc], axis=-1)       # (C, n_steps, F)
    return np.transpose(feats, (1, 0, 2)).reshape(win.shape[1], -1)


def trial_to_td64(emg_lc: np.ndarray, fs: float = F.FS_EMG, notch_hz=F.NOTCH_HZ) -> np.ndarray:
    """Raw trial (L, 16) -> (64, 16 * 4) float32."""
    x = F.preprocess(emg_lc, fs, notch_hz)
    starts, W = F.window_layout(x.shape[0], fs)
    out = td_features(F.extract_windows(x, starts, W))
    if not np.isfinite(out).all():
        raise ValueError("non-finite features")
    return out.astype(np.float32)
