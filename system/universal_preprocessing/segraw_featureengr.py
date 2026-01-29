import numpy as np
import pandas as pd
from itertools import combinations


def zero_order(channel_data, fs=2000):
    """
    Computes the zero-order spectral moment (log and raw) of the signal.
    """
    # Step 1: Square the signal (power)
    signal_squared = np.abs(channel_data) ** 2
    # Step 2: Integrate (sum) the squared values to get total power
    total_power = np.sum(signal_squared) * (1/fs)  # m0
    # Step 3: Take the logarithm of the total power
    log_total_power = np.log(total_power)
    return log_total_power, total_power


def second_order(channel_data, zero_order_raw, fs=2000):
    """
    Computes the second-order spectral moment (log and raw) of the signal. 
    """
    # Step 1: First derivative of the signal
    first_deriv = np.gradient(channel_data, edge_order=2)
    # Step 2: Square the signal (power) at each frequency of the first deriv
    signal_squared = np.abs(first_deriv) ** 2
    # Step 3: Integrate (sum) the squared values to get the total power
    total_power = np.sum(signal_squared) * (1/fs)  # m2
    # Step 4: Take the logarithm of the total power normalized by zero order
    log_total_power = np.log(total_power / (zero_order_raw ** 2))
    return log_total_power, total_power


def fourth_order(channel_data, zero_order_raw, fs=2000):
    """
    Computes the fourth-order spectral moment (log and raw) of the signal.
    """
    # Step 1: First derivative of the signal
    first_deriv = np.gradient(channel_data, edge_order=2)
    # Step 2: Second derivative of the signal
    second_deriv = np.gradient(first_deriv, edge_order=2)
    # Step 3: Square the signal (power) at each frequency of the second deriv
    signal_squared = np.abs(second_deriv) ** 2
    # Step 4: Integrate (sum) the squared values to get the total power
    total_power = np.sum(signal_squared) * (1/fs)  # m4
    # Step 5: Take the logarithm of the total power normalized by zero order
    log_total_power = np.log(total_power / (zero_order_raw ** 4))
    return log_total_power, total_power


def sparsity(zero_order_raw, second_order_raw, fourth_order_raw):
    """
    Computes the sparsity of the signal based on zero, second, and fourth-order moments.
    """

    # This sparsity calc is kind of wack, so I'm switching it to a more standard one...
    ## This is not definitionally true... I'm going to remove it
    ### Oh it does actually break my calculation below cause of the square root...
    ## Sanity Check: Ensure zero_order is greater than or equal to second and fourth for each sensor
    #assert np.all(zero_order_raw >= second_order_raw), "Error: zero_order_raw must be >= second_order_raw"
    #assert np.all(zero_order_raw >= fourth_order_raw), "Error: zero_order_raw must be >= fourth_order_raw"
    ## Step 1: Compute the square roots (element-wise)
    #sqrt_second_diff = np.sqrt(zero_order_raw - second_order_raw) 
    #sqrt_fourth_diff = np.sqrt(zero_order_raw - fourth_order_raw) 
    ## Step 2: Element-wise multiplication (not dot product)
    #elementwise_product = sqrt_second_diff * sqrt_fourth_diff
    ## Step 3: Avoid division by zero
    #if elementwise_product == 0.0:
    #    raise ValueError("elementwise_product is zero!")
    ##elementwise_product = np.maximum(elementwise_product, 1e-10)  # Ensure denominator is nonzero
    ## Step 4: Compute the sparseness ratio per sensor
    #sparseness = zero_order_raw / elementwise_product
    ## Step 5: Compute logarithm of sparseness
    #sparsity_log = np.log(sparseness)

    sparsity = np.sqrt(zero_order_raw * fourth_order_raw) / (second_order_raw + 1e-10)
    sparsity_log = np.log(sparsity + 1e-10)
    
    return sparsity_log


def irregularity_factor(channel_data, zero_order_raw, second_order_raw, fourth_order_raw, fs=2000):
    """
    Computes the irregularity factor of the signal based on zero, second, and fourth-order moments.
    """
    #epsilon = 1e-10
    # Step 1: Compute first derivative of the signal
    first_deriv = np.gradient(channel_data, edge_order=2)
    # Step 2: Absolute value of the first derivative
    signal_abs = np.abs(first_deriv)
    # Step 3: Integrate (sum) the absolute values to get WL
    WL = np.sum(signal_abs) * (1/fs)
    # Step 4: Compute IF = sqrt(m2^2 / (m0 * m4))
    m0 = zero_order_raw
    m2 = second_order_raw
    m4 = fourth_order_raw
    # Ensure numerical safety in division
    #denominator = np.maximum(m0 * m4, epsilon)  # Avoid divide-by-zero
    denominator = m0 * m4
    if denominator == 0.0:
        raise ValueError("denominator is zero!")
    IF = np.sqrt((m2 ** 2) / denominator)
    # Step 5: Compute log(IF / WL), ensuring WL is nonzero
    if WL == 0.0:
        raise ValueError("WL is zero!")
    moment_log = np.log(IF / WL)
    
    return moment_log


def create_khushaba_spectralmomentsFE_vectors(nested_dict):
    """
    Creates spectral moments features (zero_order, second_order, fourth_order, sparsity, irregularity_factor) 
    for each gesture in the nested dictionary and returns them in a single DataFrame.

    Returns:
    - A DataFrame containing the computed spectral moments features for the entire dataset.
    """

    # Initialize an empty list to store the results
    result_data = []

    # Iterate over all participants, gesture IDs, and gesture numbers
    for pid, gestures in nested_dict.items():
        for gesture_name, gesture_data in gestures.items():
            for gesture_num, modality_data in gesture_data.items():
                # Initialize dict to store the result for the current gesture
                result_dict = {
                    'Participant': pid,
                    'Gesture_ID': gesture_name,
                    'Gesture_Num': gesture_num
                }
                
                # Apply the feature engineering functions to each EMG channel
                for modality_str, single_gesture_data in modality_data.items():
                    # What's the difference between emg_channel and channel_data?
                    ## emg_channel is just "EMG" ... (should that be EMGX) with the channel num? ...
                    ## emg_channel is indexed by modality so that's what was intended...
                    for channel_idx, channel_data in enumerate(single_gesture_data):
                        zero_res, m0 = zero_order(channel_data)
                        second_res, m2 = second_order(channel_data, m0)
                        fourth_res, m4 = fourth_order(channel_data, m0)
                        # Should these be saved by modality, or by "EMG1", "EMG2", etc
                        ## I think the latter... want the headers to be for the respective channels
                        ## Note PID, gesture_ID, and gesture_num are all saved above
                        emg_channel_idx = modality_str+str(channel_idx)
                        result_dict.update({
                            f'{emg_channel_idx}_zero_order': zero_res,
                            f'{emg_channel_idx}_second_order': second_res,
                            f'{emg_channel_idx}_fourth_order': fourth_res,
                            f'{emg_channel_idx}_sparsity': sparsity(m0, m2, m4),
                            f'{emg_channel_idx}_irregularity_factor': irregularity_factor(channel_data, m0, m2, m4),
                        })
                # Append the result for the current gesture to the result list
                result_data.append(result_dict)
    # Convert the result list to a DataFrame (everything should have the same lengths now)
    result_df = pd.DataFrame(result_data)
    return result_df

######################################################################

"""
0) mean value (MV)
    MV = 1/N (\sum^N_{i=1}x_i)
    N: length of signal
    x_n: EMG signal in a segment
"""
def MV(s,fs=None):
    N = len(s)
    return 1/N*sum(s)


"""
1) mean absolute value (MAV)
    MAV = 1/N (\sum^N_{i=1}|x_i|)
    N: length of signal
    x_n: EMG signal in a segment
"""
def MAV(s,fs=None):
    N = len(s)
    return 1/N*sum(abs(s))


"""
2) standard deviation (STD)
    STD = \sqrt{1/(N-1) \sum^N_{i=1}(x_i-xbar)^2}
"""
def STD(s,fs=None):
    N = len(s)
    sbar = np.mean(s)
    return np.sqrt(1/(N-1)*sum((s-sbar)**2))


"""
3) variance of EMG (Var)
    Var = 1/(N-1)\sum^N_{i=1} x_i^2
"""
def Var(s,fs=None):
    N = len(s)
    return 1/(N-1)*sum(s**2)


"""
4) waveform length
    WL = sum (|x_i-x_{i-1})
"""
def WL(s,fs=None):
    return (sum(abs(s[1:]-s[:-1]))) / 1.0 # make sure convert to float64


"""
10) correlation coefficient (Cor)
    Cor(x,y)
    x, y: each pair of EMG channels in a time window
"""
def Cor(x,y,fs=None):
    xbar = np.mean(x)
    ybar = np.mean(y)
    num = abs(sum((x-xbar)*(y-ybar)))
    den = np.sqrt(sum((x-xbar)**2)*sum((y-ybar)**2))
    return num/den


"""
17) Hjorth mobility parameter (HMob)
    derivative correct?
"""
def HMob(s, fs):
    dt = 1/fs # 1/2000
    ds = np.gradient(s,dt) # compute derivative 
    return np.sqrt(Var(ds)/Var(s))


"""
18) Hjorth complexity parameter
    compares the similarity of the shape of a signal with
    pure sine wave 
    HCom = mobility(dx(t)/dt) / mobility(x(t))
"""
def HCom(s, fs):
    dt = 1/fs
    ds = np.gradient(s,dt)
    return HMob(ds,fs) / HMob(s,fs)


def create_abbaspour_FS_vectors(nested_dict, fs=2000):
    """
    Creates spectral moments features (zero_order, second_order, fourth_order, sparsity, irregularity_factor) 
    for each gesture in the nested dictionary and returns them in a single DataFrame.

    Returns:
    - A DataFrame containing the computed spectral moments features for the entire dataset.
    """

    # Initialize an empty list to store the results
    result_data = []

    # Iterate over all participants, gesture IDs, and gesture numbers
    for pid, gestures in nested_dict.items():
        for gesture_name, gesture_data in gestures.items():
            for gesture_num, modality_data in gesture_data.items():
                # Initialize dict to store the result for the current gesture
                result_dict = {
                    'Participant': pid,
                    'Gesture_ID': gesture_name,
                    'Gesture_Num': gesture_num
                }

                # Store all EMG channels for correlation computation
                all_channels = []

                # Apply the feature engineering functions to each EMG channel
                for modality_str, single_gesture_data in modality_data.items():
                    for channel_idx, channel_data in enumerate(single_gesture_data):
                        channel_data_npy = np.array(channel_data)
                        # Construct the channel name (e.g., "EMG1", "EMG2", etc.)
                        emg_channel_idx = f"{modality_str}{channel_idx}"
                        
                        # Compute and store per-channel features
                        result_dict.update({
                            f'{emg_channel_idx}_var': Var(channel_data_npy),
                            f'{emg_channel_idx}_wl': WL(channel_data_npy),
                            f'{emg_channel_idx}_hmob': HMob(channel_data_npy, fs=fs),
                            f'{emg_channel_idx}_hcom': HCom(channel_data_npy, fs=fs)
                        })

                        # Store channel data for later correlation calculation
                        all_channels.append((emg_channel_idx, channel_data_npy))

                # Compute correlation between every pair of channels **once per gesture**
                num_channels = len(all_channels)
                if num_channels > 1:
                    # Stack all channels into a matrix (each row = 1 channel)
                    channel_matrix = np.stack([data for _, data in all_channels])

                    # Compute pairwise correlations (returns square matrix)
                    correlation_matrix = np.corrcoef(channel_matrix)

                    # Extract upper triangular correlations (avoiding redundant pairs & self-correlation)
                    for i in range(num_channels):
                        for j in range(i + 1, num_channels):
                            ch1, ch2 = all_channels[i][0], all_channels[j][0]
                            result_dict[f'EMG_Cor_{ch1}a{ch2}'] = correlation_matrix[i, j]

                # Append the result for the current gesture to the result list
                result_data.append(result_dict)
    # Convert the result list to a DataFrame (everything should have the same lengths now)
    result_df = pd.DataFrame(result_data)
    return result_df