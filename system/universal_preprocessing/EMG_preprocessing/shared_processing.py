"""
shared_processing.py
====================
Consolidated library of all preprocessing and feature engineering functions
for the Meta-Gesture EMG/IMU dataset.

Data structure convention (nested_dict):
    { PID -> { gesture_name -> { gesture_num -> { modality -> List[List[float]] } } } }
    where the inner list is shape [n_channels x n_timepoints]

Feature DataFrame convention:
    Each row = one gesture trial
    Columns: ['Participant', 'Gesture_ID', 'Gesture_Num', <feature_cols...>]
"""

import numpy as np
import pandas as pd
import os
from scipy.signal import butter, sosfilt, iirnotch
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ===========================================================================
# SAFE MATH UTILITIES
# ===========================================================================

SAFE_LOG_MIN = -10.0
SAFE_LOG_MAX = 10.0

def safe_log(x, eps=1e-10):
    """Numerically stable log: clamps input away from zero and clips output."""
    return float(np.clip(np.log(np.maximum(x, eps)), SAFE_LOG_MIN, SAFE_LOG_MAX))


# ===========================================================================
# SIGNAL FILTERING  (from preprocessing.py)
# ===========================================================================

def apply_bpf_and_notch(data, lowcut=20, highcut=450, fs=2000,
                         notch_freq=60, quality_factor=30,
                         use_notch=False, return_as_list=True):
    """
    Bandpass filter (Butterworth order-4) and optional notch filter.
    Operates on a 2D list/array of shape [n_channels x n_timepoints].
    """
    sos = butter(N=4, Wn=[lowcut, highcut], btype='band', fs=fs, output='sos')
    filtered_data = sosfilt(sos, data)

    if use_notch:
        nyquist = 0.5 * fs
        notch_b, notch_a = iirnotch(notch_freq / nyquist, quality_factor)
        filtered_data = sosfilt(
            [[notch_b[0], notch_b[1], notch_b[2], 1, notch_a[1], notch_a[2]]],
            filtered_data
        )

    return filtered_data.tolist() if return_as_list else filtered_data


def apply_filter_to_nested_dict(nested_dict, normalization_method=None,
                                 zero_threshold=1e-7, already_BPFd=False):
    """
    Apply BPF and/or normalisation to all signals in a nested dict.

    normalization_method: 'MEANSUBTRACTION' | 'MINMAXSCALER' |
                          'STANDARDSCALER' | '$B' | None
    already_BPFd: if True, skips the bandpass filter step entirely.
    """

    def preprocess_data(data_list, method):
        if method is None:
            return data_list
        data_array = np.array(data_list)
        if method == 'MEANSUBTRACTION':
            return (data_array - np.mean(data_array, axis=1, keepdims=True)).tolist()
        elif method == 'MINMAXSCALER':
            scaler = MinMaxScaler()
            return scaler.fit_transform(data_array.T).T.tolist()
        elif method == 'STANDARDSCALER':
            scaler = StandardScaler()
            return scaler.fit_transform(data_array.T).T.tolist()
        elif method == '$B':
            std_dev = np.std(data_array)
            return (data_array / std_dev).tolist() if std_dev != 0 else data_array.tolist()
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

    preprocessed_dict = {}
    for participant, gestures in nested_dict.items():
        preprocessed_dict[participant] = {}
        for gesture_id, trials in gestures.items():
            preprocessed_dict[participant][gesture_id] = {}
            for gesture_num, modalities in trials.items():
                preprocessed_dict[participant][gesture_id][gesture_num] = {}
                for modality, data_list in modalities.items():
                    filtered_data = data_list if already_BPFd else apply_bpf_and_notch(data_list)
                    preprocessed_dict[participant][gesture_id][gesture_num][modality] = \
                        preprocess_data(filtered_data, normalization_method)
    return preprocessed_dict


def normalize_gestures_by_std(nested_dict, num_channels=16):
    """
    Per-gesture normalisation: flatten all channels → compute std → divide.
    Reshapes back to [num_channels x time] after normalising.
    """
    processed_dict = {}
    for participant, gestures in nested_dict.items():
        processed_dict[participant] = {}
        for gesture_id, trials in gestures.items():
            processed_dict[participant][gesture_id] = {}
            for gesture_num, modalities in trials.items():
                processed_dict[participant][gesture_id][gesture_num] = {}
                for modality, data_list in modalities.items():
                    data_array = np.array(data_list)
                    long_vector = data_array.flatten()
                    std_dev = np.std(long_vector)
                    normalized_vector = long_vector / std_dev if std_dev > 0 else long_vector
                    reshaped_data = normalized_vector.reshape(num_channels, -1)
                    processed_dict[participant][gesture_id][gesture_num][modality] = \
                        reshaped_data.tolist()
    return processed_dict


def normalize_gestures_by_std_any_channels(nested_dict):
    """
    Same as normalize_gestures_by_std but infers num_channels from the data.
    Use this for IMU (72 channels) or any modality where channel count varies.
    """
    processed_dict = {}
    for participant, gestures in nested_dict.items():
        processed_dict[participant] = {}
        for gesture_id, trials in gestures.items():
            processed_dict[participant][gesture_id] = {}
            for gesture_num, modalities in trials.items():
                processed_dict[participant][gesture_id][gesture_num] = {}
                for modality, data_list in modalities.items():
                    data_array = np.array(data_list)       # [n_channels x n_timepoints]
                    num_channels = data_array.shape[0]
                    long_vector = data_array.flatten()
                    std_dev = np.std(long_vector)
                    normalized_vector = long_vector / std_dev if std_dev > 0 else long_vector
                    reshaped_data = normalized_vector.reshape(num_channels, -1)
                    processed_dict[participant][gesture_id][gesture_num][modality] = \
                        reshaped_data.tolist()
    return processed_dict


# ===========================================================================
# FEATURE-MATRIX NORMALISATION  (from preprocessing.py)
# ===========================================================================

def normalize_whole_dataset_features(df,
                                      metadata_cols=('Participant', 'Gesture_ID', 'Gesture_Num')):
    """
    Global normalisation of the feature matrix: divides every feature by the
    overall standard deviation across all features and all samples (std → 1).
    Metadata columns are left untouched.
    """
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    feature_matrix = df[feature_cols].values.astype(float)
    global_std = np.std(feature_matrix, ddof=1)
    if global_std <= 0:
        raise ValueError("Global std is zero — cannot normalise feature matrix.")
    feature_matrix /= global_std
    normalized_df = df.copy()
    normalized_df[feature_cols] = feature_matrix
    return normalized_df


# ===========================================================================
# KHUSHABA SPECTRAL MOMENTS  (from segraw_featureengr.py — with safe_log)
# ===========================================================================

def zero_order(channel_data, fs=2000):
    """Zero-order spectral moment: log and raw total signal power."""
    signal_squared = np.abs(channel_data) ** 2
    total_power = np.sum(signal_squared) * (1 / fs)
    return safe_log(total_power), total_power


def second_order(channel_data, zero_order_raw, fs=2000):
    """Second-order spectral moment (energy of first derivative)."""
    first_deriv = np.gradient(channel_data, edge_order=2)
    signal_squared = np.abs(first_deriv) ** 2
    total_power = np.sum(signal_squared) * (1 / fs)
    return safe_log(total_power / (zero_order_raw ** 2 + 1e-10)), total_power


def fourth_order(channel_data, zero_order_raw, fs=2000):
    """Fourth-order spectral moment (energy of second derivative)."""
    first_deriv = np.gradient(channel_data, edge_order=2)
    second_deriv = np.gradient(first_deriv, edge_order=2)
    signal_squared = np.abs(second_deriv) ** 2
    total_power = np.sum(signal_squared) * (1 / fs)
    return safe_log(total_power / (zero_order_raw ** 4 + 1e-10)), total_power


def sparsity(zero_order_raw, second_order_raw, fourth_order_raw):
    """
    Sparsity: sqrt(m0 * m4) / m2.
    Measures whether spectral energy is concentrated or spread out.
    Uses safe_log to handle degenerate (near-zero) cases.
    """
    numerator = np.sqrt(np.maximum(zero_order_raw * fourth_order_raw, 0.0))
    denominator = second_order_raw + 1e-10
    return safe_log(numerator / denominator)


def irregularity_factor(channel_data, zero_order_raw, second_order_raw,
                         fourth_order_raw, fs=2000):
    """
    Irregularity factor: m2^2 / (m0 * m4), normalised by waveform length.
    Uses safe_log to handle degenerate (near-zero WL or IF) cases.
    """
    first_deriv = np.gradient(channel_data, edge_order=2)
    WL = np.sum(np.abs(first_deriv)) * (1 / fs)

    m0, m2, m4 = zero_order_raw, second_order_raw, fourth_order_raw
    denominator = m0 * m4
    IF = np.sqrt((m2 ** 2) / (denominator + 1e-10))
    return safe_log(IF / (WL + 1e-10))


def create_khushaba_spectralmomentsFE_vectors(nested_dict, fs=2000):
    """
    Compute Khushaba spectral moment features for every gesture trial in
    nested_dict. Works on any modality (EMG or IMU) — channel count is inferred
    automatically from the data.

    Returns a DataFrame where each row is one gesture trial:
        ['Participant', 'Gesture_ID', 'Gesture_Num',
         '<modality><ch_idx>_zero_order', ..._second_order, ..._fourth_order,
         ..._sparsity, ..._irregularity_factor, ...]
    """
    result_data = []

    for pid, gestures in nested_dict.items():
        for gesture_name, gesture_data in gestures.items():
            for gesture_num, modality_data in gesture_data.items():
                result_dict = {
                    'Participant': pid,
                    'Gesture_ID': gesture_name,
                    'Gesture_Num': gesture_num,
                }
                for modality_str, single_gesture_data in modality_data.items():
                    for channel_idx, channel_data in enumerate(single_gesture_data):
                        ch_data = np.array(channel_data, dtype=np.float64)
                        zero_log, m0 = zero_order(ch_data, fs)
                        second_log, m2 = second_order(ch_data, m0, fs)
                        fourth_log, m4 = fourth_order(ch_data, m0, fs)
                        spar = sparsity(m0, m2, m4)
                        irreg = irregularity_factor(ch_data, m0, m2, m4, fs)

                        col_prefix = f"{modality_str}{channel_idx}"
                        result_dict[f"{col_prefix}_zero_order"] = zero_log
                        result_dict[f"{col_prefix}_second_order"] = second_log
                        result_dict[f"{col_prefix}_fourth_order"] = fourth_log
                        result_dict[f"{col_prefix}_sparsity"] = spar
                        result_dict[f"{col_prefix}_irregularity_factor"] = irreg

                result_data.append(result_dict)

    return pd.DataFrame(result_data)


# ===========================================================================
# DATA LOADING — segraw format  (from segraw_aggregating_unified_df.ipynb)
# ===========================================================================

def load_segraw_data(pIDs, data_dir_path, modalities=("E",),
                     expt_types=("experimenter-defined",),
                     num_emg_channels=16, num_imu_sensors=12):
    """
    Load segmented raw CSV files into a nested dict.

    NOTE: num_imu_sensors is not called but thats fine bc we manually hardcode/correct the IMU channels. Simple range(num_imu_sensors) would be incorrect

    modalities: tuple/list containing 'E' (EMG), 'I' (IMU), or both.
    Returns: { PID -> { gesture_id -> { gesture_num -> { modality -> [ch x time] } } } }
    """
    nested_dict = {}

    for expt_type in expt_types:
        for pid in pIDs:
            print(pid)
            pid_path = os.path.join(data_dir_path, pid)
            if not os.path.isdir(pid_path):
                print(f"  Path not found: {pid_path}")
                continue

            for file in os.listdir(pid_path):
                if expt_type not in file:
                    continue

                load_emg = "E" in modalities and "EMG" in file
                load_imu = "I" in modalities and "IMU" in file
                if not load_emg and not load_imu:
                    continue

                parts = file.split("_")
                if len(parts) < 6:
                    print(f"  Unexpected filename: {file}")
                    continue

                modality   = parts[3]                       # "EMG" or "IMU"
                gestureID  = parts[4]
                gestureNum = parts[5].rsplit(".", 1)[0]

                if modality == "EMG":
                    headers = [f"EMG{i}" for i in range(1, num_emg_channels + 1)]
                else:
                    headers = [
                        f"IMU{j}_{ax}"
                        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15]
                        for ax in ("ax", "ay", "az", "vx", "vy", "vz")
                    ]

                file_path = os.path.join(pid_path, file)
                try:
                    df = pd.read_csv(file_path, header=0)
                    if df.columns[0] == "":
                        df = df.iloc[:, 1:].reset_index(drop=True)
                    df = df[headers]
                except FileNotFoundError:
                    print(f"  File not found: {file_path}")
                    continue
                except pd.errors.EmptyDataError:
                    print(f"  Empty file: {file_path}")
                    continue
                except KeyError:
                    print(f"  Missing expected columns in: {file}")
                    continue

                nested_dict.setdefault(pid, {}) \
                           .setdefault(gestureID, {}) \
                           .setdefault(gestureNum, {})

                nested_dict[pid][gestureID][gestureNum][modality] = df.T.values.tolist()

    return nested_dict
