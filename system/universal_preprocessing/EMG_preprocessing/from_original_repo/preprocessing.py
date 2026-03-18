
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, iirnotch


def normalize_whole_dataset_features(df, metadata_cols=['Participant', 'Gesture_ID', 'Gesture_Num']):
    """
    Normalizes the feature columns (excluding metadata) such that the overall 
    standard deviation across all features is set to 1.

    Args:
        df (pd.DataFrame): Input dataframe with 3 metadata columns ('PID', 'gesture_ID', 'gesture_num')
                           and 80 feature columns.

    Returns:
        pd.DataFrame: Normalized dataframe with metadata untouched.
    """
    # Extract metadata and feature columns
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    # Convert features to NumPy array for efficient processing
    feature_matrix = df[feature_cols].values.astype(float)  # Ensure numeric type
    # Compute global standard deviation across all features
    global_std = np.std(feature_matrix, ddof=1)  # Using ddof=1 for sample std
    # Normalize features (set global std to 1)
    if global_std > 0:
        feature_matrix /= global_std
    else:
        raise ValueError()

    # Reconstruct dataframe
    normalized_df = df.copy()
    normalized_df[feature_cols] = feature_matrix
    return normalized_df


def apply_bpf_and_notch(data, lowcut=20, highcut=450, fs=2000, notch_freq=60, quality_factor=30, use_notch=False, return_as_list=True):
    # Bandpass filter using second-order sections (sos)
    sos = butter(N=4, Wn=[lowcut, highcut], btype='band', fs=fs, output='sos')
    filtered_data = sosfilt(sos, data)  # Apply bandpass filter
    
    if use_notch:
        # Notch filter (still requires normalization)
        nyquist = 0.5 * fs
        notch_b, notch_a = iirnotch(notch_freq / nyquist, quality_factor)
        filtered_data = sosfilt([[notch_b[0], notch_b[1], notch_b[2], 1, notch_a[1], notch_a[2]]], filtered_data)  # Apply notch filter
    
    return filtered_data.tolist() if return_as_list else filtered_data


def apply_filter_to_nested_dict(nested_dict, normalization_method=None, zero_threshold=1e-7, already_BPFd=False):
    """
    Apply band-pass filtering (BPF) and/or normalization to all numeric data in a nested dictionary.

    Arguments:
    - nested_dict: Dictionary of the form {Participant -> {Gesture ID -> {Gesture Num -> {Modality -> List[List[float]]}}}}
    - normalization_method: String specifying normalization ('MEANSUBTRACTION', 'MINMAXSCALER', 'STANDARDSCALER', '$B'), or None for no normalization.
    - zero_threshold: Values below this threshold are set to zero.
    - already_BPFd: Boolean flag. If False, applies BPF before normalization. If True, assumes data is already filtered and skips BPF.

    Returns:
    - A new nested dictionary with filtered and/or normalized data.
    """

    def preprocess_data(data_list, method):
        """Apply the selected preprocessing method to a 2D list (channel-wise time-series)."""
        if method is None:
            return data_list  # No normalization, return as-is

        data_array = np.array(data_list)
        #data_array[data_array < zero_threshold] = 0.0  # Zero-thresholding

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
                    if not already_BPFd:
                        # Apply BPF first, then normalization
                        filtered_data = apply_bpf_and_notch(data_list)
                    else:
                        # Assume data is already filtered
                        filtered_data = data_list

                    # Apply normalization if specified
                    preprocessed_dict[participant][gesture_id][gesture_num][modality] = preprocess_data(filtered_data, normalization_method)

    return preprocessed_dict


def plot_selected_channels(nested_dict, pid, gestureID, gestureNum, modality, 
                           channels_to_plot=[1, 4, 8, 16], plot_type="line"):
    """
    Extracts and plots the specified channels for a given PID, gesture ID, gesture number, and modality.
    Each channel is plotted as a separate subplot in a stacked format.
    
    Args:
    - nested_dict (dict): The data dictionary.
    - pid (str): Participant ID.
    - gestureID (str): Gesture ID.
    - gestureNum (str): Gesture Number.
    - modality (str): Modality ("EMG" or "IMU").
    - channels_to_plot (list): List of 1-based channel indices to plot.
    - plot_type (str): "line" for line plots, "scatter" for scatter plots.
    """
    # Check if the data exists
    try:
        data = nested_dict[pid][gestureID][gestureNum][modality]  # List of lists (channels x timepoints)
    except KeyError:
        print(f"Data not found for PID={pid}, Gesture ID={gestureID}, Gesture Num={gestureNum}, Modality={modality}")
        return

    if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
        print("Data format is unexpected. Expected a list of lists (channels x timepoints).")
        return

    # Convert 1-based indexing to 0-based indexing
    max_channels = len(data)
    selected_channels = [ch - 1 for ch in channels_to_plot if 0 <= ch - 1 < max_channels]

    # Time axis
    timepoints = range(len(data[0]))  # Assuming all channels have the same length

    # Create subplots
    fig, axes = plt.subplots(len(selected_channels), 1, figsize=(12, 8), sharex=True)

    if len(selected_channels) == 1:
        axes = [axes]  # Ensure axes is iterable when only one subplot

    for ax, ch in zip(axes, selected_channels):
        if plot_type == "scatter":
            ax.scatter(timepoints, data[ch], s=1, label=f'Channel {ch + 1}', alpha=0.7)
        else:  # Default to line plot
            ax.plot(timepoints, data[ch], label=f'Channel {ch + 1}')

        ax.set_ylabel(f'Ch {ch + 1}')
        #ax.legend()
        #ax.grid(True)

    axes[-1].set_xlabel("Timepoints")
    plt.suptitle(f"PID: {pid} | Gesture ID: {gestureID} | Gesture Num: {gestureNum} | Modality: {modality}")
    plt.show()


def normalize_gestures_by_std(nested_dict, num_channels=16):
    """
    Normalizes each gesture by stacking all channels into a 1D vector,
    normalizing such that std = 1, and then reshaping it back into the original channel structure.

    Arguments:
    - nested_dict: Dictionary of the form {Participant -> {Gesture ID -> {Gesture Num -> {Modality -> List[List[float]]}}}}
    - num_channels: Number of channels per modality (default: 16)

    Returns:
    - A new nested dictionary with normalized data.
    """
    processed_dict = {}

    for participant, gestures in nested_dict.items():
        processed_dict[participant] = {}

        for gesture_id, trials in gestures.items():
            processed_dict[participant][gesture_id] = {}

            for gesture_num, modalities in trials.items():
                processed_dict[participant][gesture_id][gesture_num] = {}

                for modality, data_list in modalities.items():
                    # Convert to numpy array
                    data_array = np.array(data_list)

                    # Stack all channels into a 1D vector
                    long_vector = data_array.flatten()

                    # Normalize the entire vector to have std = 1
                    std_dev = np.std(long_vector)
                    if std_dev > 0:
                        normalized_vector = long_vector / std_dev
                    else:
                        normalized_vector = long_vector  # Avoid division by zero if std_dev == 0

                    # Reshape back into (num_channels, time)
                    reshaped_data = normalized_vector.reshape(num_channels, -1)

                    # Convert back to list and store
                    processed_dict[participant][gesture_id][gesture_num][modality] = reshaped_data.tolist()

    return processed_dict
