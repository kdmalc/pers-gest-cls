"""
EMG-FS64: a 64-step windowed spectral-feature sequence from native-rate sEMG (docs/SPEC_2khz_emg.md).

For each trial (L samples x 16 channels at FS_EMG = 2048 Hz) we produce (64, 16*F) features whose 64
steps sit at the same NORMALISED times as the 64 IMU steps of the fused tensor dict
(step i is centred at (i + 0.5) / 64 of the trial), so EMG-FS64 concatenates with the existing IMU
tensor along channels exactly like the 20 Hz envelope did.

Pipeline per trial: demean -> 20-450 Hz band-pass (zero phase) -> [optional mains notch] ->
`$B` block normalisation (one scalar std over all 16 channels, which preserves cross-channel amplitude
ratios) -> windowed features.

Features per channel per window (F = 7): log m0, log(m2/m0^2), log(m4/m0^4), log sparsity,
log irregularity-factor/WL (the repository's Khushaba TD-PSD definitions,
EMG_preprocessing/shared_processing.py), plus log WL and log MAV.
"""
import numpy as np
from scipy import signal

FS_EMG = 2048          # Hz (recording rate per the lab; the older notebooks wrongly used 2000)
BAND_HZ = (20.0, 450.0)
MAINS_HZ = (60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0)
# The raw data show NO mains line (diagnose_2khz.py: 59-61 Hz power ~1.3x its neighbourhood), so no
# notch is applied by default; pass notch_hz=MAINS_HZ to preprocess() if a dataset needs it.
NOTCH_HZ = ()
NOTCH_Q = 30.0
N_STEPS = 64
MIN_WIN_S = 0.150
N_FEATURES = 7
FEATURE_NAMES = ("log_m0", "log_m2n", "log_m4n", "log_sparsity", "log_irreg_over_wl",
                 "log_wl", "log_mav")
EPS = 1e-10


def safe_log(x):
    return np.log(np.maximum(x, EPS))


def preprocess(emg_lc: np.ndarray, fs: float = FS_EMG, notch_hz=NOTCH_HZ) -> np.ndarray:
    """emg_lc: (L, C) raw -> (L, C) filtered and block-normalised (std over everything == 1)."""
    x = np.asarray(emg_lc, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sos = signal.butter(4, BAND_HZ, btype="bandpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=0)
    for f0 in notch_hz:
        b, a = signal.iirnotch(f0, NOTCH_Q, fs=fs)
        x = signal.filtfilt(b, a, x, axis=0)
    sd = x.std()
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("degenerate trial: zero or non-finite std after filtering")
    return x / sd


def window_layout(length: int, fs: float = FS_EMG, n_steps: int = N_STEPS,
                  min_win_s: float = MIN_WIN_S):
    """-> (starts, W): window start indices (may be negative: reflect-padded) and window length."""
    W = max(int(round(min_win_s * fs)), int(round(2.0 * length / n_steps)))
    centers = (np.arange(n_steps) + 0.5) * length / n_steps
    starts = np.round(centers - W / 2.0).astype(int)
    return starts, W


def extract_windows(x_lc: np.ndarray, starts: np.ndarray, W: int) -> np.ndarray:
    """(L, C) -> (C, n_steps, W) with reflect padding at the trial edges."""
    L = x_lc.shape[0]
    pad = int(max(0, -starts.min(), starts.max() + W - L)) + 1
    if pad >= L:
        raise ValueError(f"trial of {L} samples is too short for windows of {W}")
    xp = np.pad(x_lc, ((pad, pad), (0, 0)), mode="reflect")
    idx = (starts[:, None] + pad) + np.arange(W)[None, :]              # (n_steps, W)
    return np.transpose(xp[idx], (2, 0, 1))                            # (C, n_steps, W)


def td_features(win: np.ndarray, fs: float = FS_EMG) -> np.ndarray:
    """win: (C, n_steps, W) -> (n_steps, C * N_FEATURES), channel-major (channel c feature f at c*F+f)."""
    d1 = np.gradient(win, axis=-1, edge_order=2)
    d2 = np.gradient(d1, axis=-1, edge_order=2)
    inv_fs = 1.0 / fs
    m0 = (win ** 2).sum(-1) * inv_fs
    m2 = (d1 ** 2).sum(-1) * inv_fs
    m4 = (d2 ** 2).sum(-1) * inv_fs
    wl = np.abs(d1).sum(-1) * inv_fs
    sparsity = np.sqrt(np.maximum(m0 * m4, 0.0)) / (m2 + EPS)
    irreg = np.sqrt(m2 ** 2 / (m0 * m4 + EPS))
    feats = np.stack([
        safe_log(m0),
        safe_log(m2 / (m0 ** 2 + EPS)),
        safe_log(m4 / (m0 ** 4 + EPS)),
        safe_log(sparsity),
        safe_log(irreg / (wl + EPS)),
        safe_log(wl),
        safe_log(np.abs(win).mean(-1)),
    ], axis=-1)                                                         # (C, n_steps, F)
    return np.transpose(feats, (1, 0, 2)).reshape(win.shape[1], -1)


def trial_to_fs64(emg_lc: np.ndarray, fs: float = FS_EMG, notch_hz=NOTCH_HZ) -> np.ndarray:
    """Raw trial (L, 16) -> (64, 16 * 7) float32."""
    x = preprocess(emg_lc, fs, notch_hz)
    starts, W = window_layout(x.shape[0], fs)
    out = td_features(extract_windows(x, starts, W), fs)
    if not np.isfinite(out).all():
        raise ValueError("non-finite features")
    return out.astype(np.float32)
