"""build_fs64_tensor_dict on a tiny synthetic raw directory + fused dict (no real data)."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system" / "universal_preprocessing"
                       / "EMG_preprocessing"))
import build_fs64_tensor_dict as B  # noqa: E402
import raw_emg_io as R  # noqa: E402

PIDS = ["P004", "P005"]          # P005 is a dev user


def _make_raw(root, drop=None):
    rng = np.random.default_rng(0)
    for pid in PIDS:
        (root / pid).mkdir(parents=True)
        t = 31130000
        for g in R.GESTURE_TO_CLASS:
            for rep in range(1, 11):
                if drop == (pid, g, rep):
                    continue
                L = 2500 + 100 * rep
                x = rng.normal(size=(L, 16)) * (1 + R.GESTURE_TO_CLASS[g]) * 1e-4
                df = pd.DataFrame(x, columns=[f"EMG{i}" for i in range(1, 17)])
                t += 5
                df.to_csv(root / pid / f"2300{t}_{pid}_experimenter-defined_EMG_{g}_{rep}.csv")


def _make_fused(path):
    names = list(R.GESTURE_TO_CLASS)
    data = {pid: {c: {"emg": torch.zeros(10, 64, 16), "imu": torch.randn(10, 64, 72),
                      "demo": torch.zeros(12), "gesture_id": names[c], "enc_gesture_id": c,
                      "enc_pid": i, "rep_indices": list(range(1, 11))} for c in range(10)}
            for i, pid in enumerate(PIDS)}
    with open(path, "wb") as f:
        pickle.dump({"data": data, "gesture_label_encoder": None, "pid_encoder": None}, f)


def test_build_replaces_emg_keeps_everything_else(tmp_path):
    _make_raw(tmp_path / "raw")
    _make_fused(tmp_path / "fused.pkl")
    out = B.build(tmp_path / "raw", tmp_path / "fused.pkl", tmp_path / "out.pkl",
                  dev_pids=["P005"], log=lambda *_: None)
    with open(tmp_path / "fused.pkl", "rb") as f:
        old = pickle.load(f)["data"]
    for pid in PIDS:
        for c in range(10):
            e = out["data"][pid][c]
            assert e["emg"].shape == (10, 64, 112) and torch.isfinite(e["emg"]).all()
            assert torch.equal(e["imu"], old[pid][c]["imu"]) and e["rep_indices"] == list(range(1, 11))
    assert out["emg_repr"] == "fs64" and out["feature_norm"]["computed_from_dev_pids"] == ["P005"]
    # standardised on the dev user only: per-feature-type mean ~0, std ~1 there
    dev = torch.cat([out["data"]["P005"][c]["emg"].reshape(-1, 16, 7) for c in range(10)])
    assert dev.mean(dim=(0, 1)).abs().max() < 1e-3
    assert (dev.std(dim=(0, 1)) - 1).abs().max() < 1e-3
    assert out["feature_norm"]["mean"] and len(out["feature_norm"]["mean"]) == 7


def test_missing_rep_or_wrong_label_order_raises(tmp_path):
    _make_raw(tmp_path / "raw", drop=("P004", "pan", 3))
    _make_fused(tmp_path / "fused.pkl")
    with pytest.raises(ValueError, match="reps"):
        B.build(tmp_path / "raw", tmp_path / "fused.pkl", tmp_path / "o.pkl",
                dev_pids=["P005"], log=lambda *_: None)

    (tmp_path / "raw2").mkdir()
    _make_raw(tmp_path / "raw2" / "x")
    with open(tmp_path / "fused.pkl", "rb") as f:
        fused = pickle.load(f)
    fused["data"]["P004"][0]["gesture_id"] = "delete"
    with open(tmp_path / "fused_bad.pkl", "wb") as f:
        pickle.dump(fused, f)
    with pytest.raises(ValueError, match="expected 'close'"):
        B.build(tmp_path / "raw2" / "x", tmp_path / "fused_bad.pkl", tmp_path / "o2.pkl",
                dev_pids=["P005"], log=lambda *_: None)


def test_timestamp_seconds_orders_reps():
    assert R.timestamp_seconds("230031130021") - R.timestamp_seconds("230031130016") == 5


def test_pid_mismatch_raises_except_for_the_known_p005_p105_case(tmp_path):
    d = tmp_path / "P004"
    d.mkdir()
    (d / "230104110125_P105_experimenter-defined_EMG_pan_1.csv").write_text(",EMG1\n0,1\n")
    with pytest.raises(ValueError, match="folder PID"):
        R.list_emg_files(tmp_path, "P004")             # reaches the PID check before the rep check
    d5 = tmp_path / "P005"
    d5.mkdir()
    (d5 / "230104110125_P105_experimenter-defined_EMG_pan_1.csv").write_text(",EMG1\n0,1\n")
    with pytest.raises(ValueError, match="reps"):      # PID accepted, then the missing reps are caught
        R.list_emg_files(tmp_path, "P005")
