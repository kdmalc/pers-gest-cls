"""
E1 diagnostics for the native-rate EMG (docs/SPEC_2khz_emg.md). Runs on the raw CSVs (laptop or cluster).

    python diagnose_2khz.py --root "<...>/segmented_raw_data" [--pids P102 P103 ...] [--out diag.json]

Reports, over all trials read:
  * trial length (samples and seconds at 2048 Hz),
  * active-segment statistics: activity fraction and lead/trail rest margins (smoothed 20-450 Hz
    envelope above 20% of the trial's 95th-percentile envelope),
  * the SAMPLING-RATE check: the mains line is 60 Hz, so if the true rate is 2048 Hz its spectral
    peak sits at 60.0 Hz on a 2048 Hz axis; at a true 2000 Hz it would sit at 61.44 Hz,
  * 60 Hz line strength relative to its neighbourhood, and the fraction of power above 450 Hz.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emg_fs64 as F  # noqa: E402
from raw_emg_io import list_emg_files, read_emg_csv  # noqa: E402

ASSUMED_FS = 2048.0


def mains_peak_hz(x_lc: np.ndarray) -> float:
    """Peak frequency (Hz, on an ASSUMED_FS axis) of the strongest line in 55-66 Hz, refined by a
    zero-padded FFT of the channel-averaged power spectrum."""
    x = x_lc - x_lc.mean(axis=0, keepdims=True)
    n = 1 << 17
    spec = (np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], n=n, axis=0)) ** 2).mean(axis=1)
    freqs = np.fft.rfftfreq(n, 1 / ASSUMED_FS)
    band = (freqs >= 55) & (freqs <= 66)
    return float(freqs[band][np.argmax(spec[band])])


def activity(x_lc: np.ndarray, fs: float):
    """-> (active fraction, lead rest s, trail rest s) from the smoothed rectified envelope."""
    sos = signal.butter(4, F.BAND_HZ, btype="bandpass", fs=fs, output="sos")
    env = np.abs(signal.sosfiltfilt(sos, x_lc - x_lc.mean(0), axis=0)).mean(axis=1)
    k = max(1, int(0.100 * fs))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    act = env > 0.2 * np.percentile(env, 95)
    idx = np.flatnonzero(act)
    if idx.size == 0:
        return 0.0, len(env) / fs, 0.0
    return float(act.mean()), idx[0] / fs, (len(env) - 1 - idx[-1]) / fs


def line_ratio_and_hf(x_lc: np.ndarray, fs: float):
    f, p = signal.welch(x_lc - x_lc.mean(0), fs=fs, nperseg=min(4096, len(x_lc)), axis=0)
    p = p.mean(axis=1)
    near = (f >= 59) & (f <= 61)
    around = ((f >= 50) & (f < 57)) | ((f > 63) & (f <= 70))
    hf = p[f > 450].sum() / p[(f >= 20)].sum()
    return float(p[near].max() / p[around].mean()), float(hf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pids", nargs="+", default=None)
    ap.add_argument("--max-trials-per-user", type=int, default=None, help="subsample for a quick look")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    pids = args.pids or sorted(p.name for p in root.iterdir()
                               if p.name.startswith("P") and p.name[1:].isdigit() and len(p.name) == 4)
    lens, act, lead, trail, peaks, ratios, hfs = [], [], [], [], [], [], []
    for pid in pids:
        try:
            files = list_emg_files(root, pid)
        except ValueError as e:                       # participants without all 10 gestures x 10 reps
            print(f"skip {pid}: {e}")
            continue
        trials = [p for items in files.values() for _, p, _ in items]
        if args.max_trials_per_user:
            trials = trials[:: max(1, len(trials) // args.max_trials_per_user)]
        for path in trials:
            x = read_emg_csv(path)
            lens.append(len(x))
            a, l_, t_ = activity(x, ASSUMED_FS)
            act.append(a); lead.append(l_); trail.append(t_)
            peaks.append(mains_peak_hz(x))
            r, hf = line_ratio_and_hf(x, ASSUMED_FS)
            ratios.append(r); hfs.append(hf)
        print(f"{pid}: {len(trials)} trials read")

    def q(v, ps=(0, 10, 50, 90, 100)):
        return {f"p{p}": float(np.percentile(v, p)) for p in ps}
    lens = np.asarray(lens)
    out = {
        "n_trials": int(lens.size),
        "length_samples": q(lens), "length_s_at_2048": q(lens / 2048.0),
        "length_s_at_2000": q(lens / 2000.0),
        "active_fraction": q(act), "lead_rest_s": q(lead), "trail_rest_s": q(trail),
        "mains_peak_hz_on_2048_axis": q(peaks, (5, 25, 50, 75, 95)),
        "line60_to_neighbourhood_power": q(ratios, (10, 50, 90)),
        "power_fraction_above_450hz": q(hfs, (10, 50, 90)),
    }
    med = out["mains_peak_hz_on_2048_axis"]["p50"]
    if out["line60_to_neighbourhood_power"]["p50"] < 3.0:
        out["rate_verdict"] = ("INCONCLUSIVE: no mains line in the data (median 59-61 Hz power is "
                               "%.1fx its neighbourhood), so the recording rate cannot be read from "
                               "the signal; rely on the acquisition-system specification"
                               % out["line60_to_neighbourhood_power"]["p50"])
    else:
        out["rate_verdict"] = ("2048 Hz (mains at %.2f Hz on a 2048 Hz axis)" % med
                               if abs(med - 60.0) < abs(med - 61.44) else
                               "2000 Hz (mains at %.2f Hz on a 2048 Hz axis)" % med)
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
