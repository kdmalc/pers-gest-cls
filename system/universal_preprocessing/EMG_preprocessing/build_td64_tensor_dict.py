"""
Build segfilt_td64_tensor_dict.pkl: the fused tensor dict with EMG replaced by classical Hudgins
time-domain features (MAV, WL, ZC, SSC) in 64 windows (emg_td.py). Everything else (IMU, labels, rep
ids, dev-user-only feature standardisation, all checks) is shared with build_fs64_tensor_dict.build.

    python build_td64_tensor_dict.py --root <segmented_raw_data> \
        --fused-dict dataset/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl \
        --out dataset/meta-learning-sup-que-ds/segfilt_td64_tensor_dict.pkl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_fs64_tensor_dict as B  # noqa: E402
import emg_td as T  # noqa: E402


def build(root, fused_path, out_path, **kw):
    return B.build(root, fused_path, out_path, trial_fn=T.trial_to_td64, n_features=T.N_FEATURES,
                   feature_names=T.FEATURE_NAMES, repr_name="td64", **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fused-dict", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.root, a.fused_dict, a.out)
