"""EMG-FS64 feature pipeline: shapes, alignment to normalised time, scale behaviour, errors."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system" / "universal_preprocessing"
                       / "EMG_preprocessing"))
import emg_fs64 as F  # noqa: E402


def _noise(L=6000, C=16, seed=0):
    return np.random.default_rng(seed).normal(size=(L, C)) * 1e-4


def test_output_shape_dtype_and_finite():
    out = F.trial_to_fs64(_noise())
    assert out.shape == (64, 16 * F.N_FEATURES) and out.dtype == np.float32
    assert np.isfinite(out).all()


@pytest.mark.parametrize("L", [1580, 3000, 6381, 24999])
def test_window_layout_covers_trial_and_min_window(L):
    starts, W = F.window_layout(L)
    assert W >= int(round(0.150 * F.FS_EMG))
    centers = starts + W / 2.0
    assert np.all(np.diff(centers) > 0)
    assert centers[0] == pytest.approx(0.5 * L / 64, abs=1.0)
    assert centers[-1] == pytest.approx(63.5 * L / 64, abs=1.0)


def test_burst_appears_at_its_normalised_time_for_any_trial_length():
    # a burst at 70% of the trial must show up near step 0.7*64 regardless of the length
    for L in (2500, 7000):
        x = _noise(L) * 0.1
        a, b = int(0.68 * L), int(0.72 * L)
        x[a:b] += np.random.default_rng(1).normal(size=(b - a, 16)) * 5e-3
        out = F.trial_to_fs64(x)
        log_mav = out[:, F.FEATURE_NAMES.index("log_mav")::F.N_FEATURES].mean(axis=1)
        assert abs(int(np.argmax(log_mav)) - int(0.70 * 64)) <= 3, (L, int(np.argmax(log_mav)))


def test_amplitude_scale_is_removed_by_block_normalisation():
    x = _noise()
    a = F.trial_to_fs64(x)
    b = F.trial_to_fs64(x * 37.0)
    assert np.allclose(a, b, atol=1e-3)


def test_cross_channel_amplitude_ratio_is_preserved():
    x = _noise()
    x[:, 3] *= 4.0                                          # channel 3 four times larger
    out = F.trial_to_fs64(x)
    lm = out[:, F.FEATURE_NAMES.index("log_mav")::F.N_FEATURES].mean(axis=0)
    others = np.delete(lm, 3).mean()
    assert lm[3] - others == pytest.approx(np.log(4.0), abs=0.3)


def test_spectral_shape_features_separate_low_and_high_frequency_content():
    rng = np.random.default_rng(2)
    white = rng.normal(size=(6000, 16))
    lp = signal_lp(white)
    f_white, f_lp = F.trial_to_fs64(white), F.trial_to_fs64(lp)
    j = F.FEATURE_NAMES.index("log_m4n")
    assert f_white[:, j::F.N_FEATURES].mean() > f_lp[:, j::F.N_FEATURES].mean() + 0.5


def signal_lp(x):
    from scipy import signal
    sos = signal.butter(4, 80.0, btype="low", fs=F.FS_EMG, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def test_60hz_line_is_attenuated_at_least_10x():
    t = np.arange(6000) / F.FS_EMG
    x = _noise() * 0.2 + 5e-3 * np.sin(2 * np.pi * 60 * t)[:, None]
    freqs = np.fft.rfftfreq(6000, 1 / F.FS_EMG)
    k60 = np.argmin(np.abs(freqs - 60))

    def line_to_floor(y):
        spec = np.abs(np.fft.rfft(y - y.mean(0), axis=0)).mean(axis=1)
        floor = np.r_[spec[k60 - 40:k60 - 10], spec[k60 + 10:k60 + 40]].mean()
        return spec[k60] / floor

    assert line_to_floor(x) / line_to_floor(F.preprocess(x, notch_hz=F.MAINS_HZ)) > 10   # a line 100x the noise floor


def test_degenerate_and_short_trials_raise():
    with pytest.raises(ValueError, match="degenerate"):
        F.trial_to_fs64(np.zeros((3000, 16)))
    with pytest.raises(ValueError, match="too short"):
        F.extract_windows(np.ones((100, 2)), np.array([0, 50]), 400)
