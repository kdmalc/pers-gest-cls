import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import seaborn as sns


def variance_explained(X, y):
    y = y.clone().detach()
    X = torch.tensor(X).float()
    model = LinearRegression().fit(X, y)
    pred = torch.tensor(model.predict(X)).float()
    total_var = torch.var(y, dim=0)
    explained_var = torch.var(pred, dim=0)
    return (explained_var / total_var).item()


def recreate_f_to_h_fig(my_model, input_data, gesture_labels, enc_participant_ids,
                        participant_ids_str, rms_vals, \
                        selected_participants_lst=None, num_random_users_to_plot=7, use_legend=False):

    # LSTM Forward Pass to get the LSTM hidden states
    my_model.eval()
    with torch.no_grad():
        input_data = input_data.to(next(my_model.parameters()).device)
        x = my_model.conv(input_data)
        x = x.permute(0, 2, 1)
        x = my_model.ln1(x)
        x, (hn, cn) = my_model.lstm(x)

    # Extract LSTM Activations (also convert to npy for PCA later)
    ## This is the representation space
    lstm_outputs = [x.detach().cpu().numpy() for x in hn]
    num_layers = len(lstm_outputs)

    # Select users to plot
    # Use provided participant list if given
    unique_participants = np.unique(enc_participant_ids)
    if selected_participants_lst is not None:
        selected_participants = np.array(selected_participants_lst)
    else:
        selected_participants = np.random.choice(unique_participants, size=num_random_users_to_plot, replace=False)
    mask = np.isin(enc_participant_ids, selected_participants)
    filtered_lstm_outputs = [layer[mask] for layer in lstm_outputs]
    filtered_gesture_labels = np.array(gesture_labels)[mask]
    filtered_participant_ids = np.array(enc_participant_ids)[mask]
    filtered_rms_vals = np.array(rms_vals)[mask]

    color_vars = [filtered_gesture_labels, filtered_participant_ids, filtered_rms_vals]
    color_labels = ['Gesture', 'Participant', 'RMS Power']
    colormaps = [sns.color_palette("bright", as_cmap=False),
                 sns.color_palette("muted", as_cmap=False),
                 'viridis']

    fig, axs = plt.subplots(nrows=num_layers, ncols=3, figsize=(3*num_layers, 3*num_layers))

    if num_layers == 1:
        axs = np.expand_dims(axs, axis=0)  # Ensure axs[i][j] indexing still works

    for col_idx, col_title in enumerate(["By Gesture", "By Participant", "By EMG RMS Power"]):
        axs[0, col_idx].set_title(col_title, fontsize=14)

    for row_idx in range(num_layers):
        axs[row_idx, 0].annotate(f"LSTM Layer {row_idx + 1}",
                                 xy=(-0.12, 0.5), xycoords='axes fraction',
                                 fontsize=14, ha='right', va='center', rotation=90)

    for layer_idx, layer in enumerate(filtered_lstm_outputs):
        proj = PCA(n_components=2).fit_transform(layer)

        for col_idx, (var, label, cmap_obj) in enumerate(zip(color_vars, color_labels, colormaps)):
            ax = axs[layer_idx, col_idx]

            if label == 'RMS Power':
                norm = plt.Normalize(vmin=np.min(var), vmax=np.max(var))
                colors = plt.colormaps[cmap_obj](norm(var))
            else:
                categories = np.unique(var)
                palette = cmap_obj
                category_colors = palette[:len(categories)]
                category_to_color = {cat: category_colors[i] for i, cat in enumerate(categories)}
                colors = np.array([category_to_color[v] for v in var])

            ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=30)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Legends
    for col_idx, (label, cmap_obj) in enumerate(zip(color_labels, colormaps)):
        ax = axs[0, col_idx]
        var = color_vars[col_idx]

        if label == 'RMS Power':
            vmin, vmax = np.min(var), np.max(var)
            cmap = plt.colormaps[cmap_obj]
            handles = [
                plt.Line2D([], [], marker='o', linestyle='', color=cmap(0), label=f"Low: {vmin:.2f}"),
                plt.Line2D([], [], marker='o', linestyle='', color=cmap(1.0), label=f"High: {vmax:.2f}")
            ]
            if use_legend:
                ax.legend(handles=handles, title=label, loc='upper right', fontsize=8)

        else:
            categories = np.unique(var)
            palette = cmap_obj
            category_colors = palette[:len(categories)]
            if label == 'Participant':
                encoded_ids = np.array(enc_participant_ids)
                string_ids = np.array(participant_ids_str)
                enc_to_str = {enc: string_ids[encoded_ids == enc][0] for enc in categories}
                handles = [
                    plt.Line2D([], [], marker='o', linestyle='', color=category_colors[i],
                               label=enc_to_str.get(cat, str(cat)))
                    for i, cat in enumerate(categories)
                ]
            else:
                handles = [
                    plt.Line2D([], [], marker='o', linestyle='', color=category_colors[i], label=str(cat))
                    for i, cat in enumerate(categories)
                ]
            if use_legend:
                ax.legend(handles=handles, title=label, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


    # ---- Panel i: Variance explained ----
    var_explained = []
    for layer in lstm_outputs:
        row = []
        for y in [gesture_labels, enc_participant_ids, rms_vals]:
            y_tensor = y.clone().detach().float().reshape(-1, 1)
            row.append(variance_explained(layer, y_tensor))
        var_explained.append(row)
    var_explained = np.array(var_explained).squeeze()

    plt.figure(figsize=(3, 3))
    x = np.arange(3)
    labels = ["Layer1", "Layer2", "Layer3"]
    markers = ['o', 's', '^']
    colors = ['C0', 'C1', 'C2']

    for i, (label, color, marker) in enumerate(zip(['Gest', 'User', 'EMG RMS'], colors, markers)):
        plt.plot(var_explained[:, i], label=label, color=color)
        plt.scatter(x, var_explained[:, i], color=color, marker=marker, s=60)

    plt.xticks(x, labels)
    plt.ylabel("Proportion of Variance Explained")
    plt.xlabel("LSTM Layer")
    plt.title("Panel i: Variance Explained by Variable")
    plt.legend()

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


########################################################
# Panels b-e that aren't doing what I want, paper is a bit undefined tho
########################################################

from scipy.signal import butter, filtfilt
from scipy.fft import fft

# Define hardcoded values as clear named constants
NUM_FILTERS_SELECTED = 10
NUM_EMG_SAMPLES_SELECTED = 15
PLOT_CHANNEL = 0


def highpass_filter(data, fs=2000, cutoff=40):
    b, a = butter(4, cutoff / (fs / 2), btype='high')
    return filtfilt(b, a, data)


def get_freq_response(signal, fs=2000):
    freq = np.fft.rfftfreq(len(signal), d=1/fs)
    spectrum = np.abs(fft(signal))[:len(freq)]
    return freq, spectrum / np.max(spectrum)


def rms_power(signal):
    return np.sqrt(np.mean(signal**2, axis=-1))



def old_panels_b_to_e_fig(my_model, input_data):
    # ---- Panels b-e: 2x2 subplot layout ----

    filters = my_model.conv.weight.data.cpu().numpy()
    indices = np.random.choice(input_data.shape[0], size=NUM_EMG_SAMPLES_SELECTED, replace=False)
    pseudoMUAPs = [highpass_filter(input_data[i, PLOT_CHANNEL].cpu().numpy()) for i in indices]

    all_specs = []
    for f in filters:
        chan = np.argmax(np.sum(f**2, axis=1))
        _, s = get_freq_response(f[chan])
        all_specs.append(s)
    median_spec = np.median(np.vstack(all_specs), axis=0)
    mean_spec = np.mean([get_freq_response(sig)[1] for sig in pseudoMUAPs], axis=0)

    conv_rms = []
    for f in filters[:NUM_FILTERS_SELECTED]:
        chan = np.argmax(np.sum(f**2, axis=1))
        rms = rms_power(f[chan])
        conv_rms.append(rms)
    median_rms = np.median(np.vstack(conv_rms), axis=0)
    muap_rms = [rms_power(sig.reshape(1, -1)) for sig in pseudoMUAPs]
    mean_rms = np.mean(muap_rms, axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # Panel b
    for i in range(5):
        axs[0, 0].plot(filters[i][0], label=f'Filter {i+1}')
    axs[0, 0].set_title("Conv Filter Weights (Ch 0)")
    axs[0, 0].legend()

    # Panel c
    for sig in pseudoMUAPs:
        axs[0, 1].plot(sig, color='orange', alpha=0.3)
    mean_muap = np.mean(pseudoMUAPs, axis=0)
    axs[0, 1].plot(mean_muap, color='orange', linewidth=2.0, label='Mean pseudoMUAP')
    axs[0, 1].set_title("Filtered EMG Waveforms")
    axs[0, 1].legend()

    # Panel d
    for i in range(NUM_FILTERS_SELECTED):
        chan_idx = np.argmax(np.sum(filters[i]**2, axis=1))
        freq, spec = get_freq_response(filters[i][chan_idx])
        axs[1, 0].plot(freq, spec, color='blue', alpha=0.3)
    axs[1, 0].plot(freq, median_spec, color='blue', linewidth=2.0, label='Median Filter')
    for sig in pseudoMUAPs:
        freq, spec = get_freq_response(sig)
        axs[1, 0].plot(freq, spec, color='orange', alpha=0.3)
    axs[1, 0].plot(freq, mean_spec, color='orange', linewidth=2.0, label='Mean pseudoMUAP')
    axs[1, 0].set_xlim(0, 500)
    axs[1, 0].set_title("Frequency Response")
    axs[1, 0].legend()

    # Panel e
    for f in filters[:NUM_FILTERS_SELECTED]:
        chan = np.argmax(np.sum(f**2, axis=1))
        rms = rms_power(f[chan])
        axs[1, 1].plot(rms / np.max(rms), color='blue', alpha=0.3)
    axs[1, 1].plot(median_rms / np.max(median_rms), color='blue', linewidth=2.0, label='Median Filter')
    for rms in muap_rms:
        axs[1, 1].plot(rms / np.max(rms), color='orange', alpha=0.3)
    axs[1, 1].plot(mean_rms / np.max(mean_rms), color='orange', linewidth=2.0, label='Mean pseudoMUAP')
    axs[1, 1].set_title("RMS Power Comparison")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
