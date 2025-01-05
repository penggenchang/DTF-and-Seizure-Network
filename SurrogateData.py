import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from scipy.signal import hilbert, fftconvolve
from scipy.fft import fft, ifft

Fs = 1000
# for 1000
SZ_path = ["F:\CSV\\UTSW_PatientID\\ictal_1.csv","F:\CSV\\UTSW_PatientID\\ictal_3.csv",
           "F:\CSV\\UTSW_PatientID\\ictal_3.csv"]
sz_sample = "F:\CSV\\sz2_full.csv"




notable_channels = ['CH1','CH2','CH3','CH4','CH5','CH6','CH7','CH8','CH9','CH10','CH11','CH12',
                    'CH13','CH14','CH15','CH16','CH17','CH18','CH19','CH20']  # for 1000

num_eegchannel = int(len(notable_channels))


def Data_split(input_df, Timesize,overlap):  # Timesize in Sec.,
    win_size = Timesize * Fs
    #raw_SEEG = input_df.drop(channel_irrelevant, axis=1)
    raw_SEEG = input_df.loc[:,notable_channels]

    if overlap == 50:
        num_epoch = int(2*raw_SEEG.shape[0] /win_size-1)
        effect_size = int(win_size/2)
    else:
        num_epoch = raw_SEEG.shape[0] / win_size  # default non-lapping
        effect_size = win_size

    df_series = []
    count = 0
    while True:
        if count > num_epoch:
            break

        df_series.append(raw_SEEG.iloc[count*effect_size:count*effect_size+win_size])
        count += 1

    return df_series,count

raw_data_path = SZ_path[2]  ### simply for test
raw_data = pd.read_csv(raw_data_path)

sz_series, _ = Data_split(input_df=raw_data, Timesize=2, overlap=50)

def compute_dtf(data, model_order=10, f_sample=Fs, freq_band=(13, 30)):
    """
    Computes Directed Transfer Function (DTF) summed over a specified frequency band,
    with normalization applied to each channel.

    Args:
        data (numpy.ndarray): Input data (channels x timepoints).
        model_order (int): Order of the MVAR model.
        f_sample (int): Sampling frequency in Hz.
        freq_band (tuple): Frequency band of interest (low, high) in Hz.

    Returns:
        numpy.ndarray: Normalized DTF values as a channel x channel matrix.
    """
    # Fit MVAR model using statsmodels
    data = data
    var_model = VAR(data)
    results = var_model.fit(model_order)

    A_list = [results.coefs[i] for i in range(model_order)]
    if len(A_list) != model_order:
        raise ValueError("MVAR coefficients are not properly shaped. Check input data or model order!")


    # Initialize the DTF matrix (channels x channels)
    dtf_matrix = np.zeros((num_eegchannel, num_eegchannel))

    # Frequency range of interest
    f_min, f_max = freq_band
    freqs = np.linspace(f_min, f_max, 256)  # Higher resolution for integration
    delta_f = freqs[1] - freqs[0]  # Frequency bin width

    # Compute DTF
    for f in freqs:
        omega = 2 * np.pi * f / f_sample
        H = np.eye(num_eegchannel,dtype=np.complex128)
        for lag, A_lag in enumerate(A_list):
            scalar_value = np.exp(-1j * omega * (lag + 1))
            #print(f"H shape: {H.shape}")
            #print(f"A_lag shape: {A_lag.shape}")
            #print(f"Scalar shape: {np.exp(-1j * omega * (lag + 1)).shape}")

            H -= A_lag * scalar_value
        H_inv = np.linalg.pinv(H)
        H_abs2 = np.abs(H_inv) ** 2
        H_norm = np.sum(H_abs2, axis=1,keepdims=True)  # Normalize across outgoing edges

        for i in range(num_eegchannel):
            for j in range(num_eegchannel):
                dtf_matrix[i, j] += (np.abs(H_inv[i, j]) ** 2 / H_norm[i]) * delta_f  # Integration

    # Normalize the DTF matrix
    #dtf_matrix = np.sum(dtf_matrix, axis=1, keepdims=True)  # Normalize row-wise (outgoing connections)
    dtf_matrix /= np.sum(dtf_matrix, axis=1, keepdims=True)  # Normalize row-wise (outgoing connections)

    return dtf_matrix


def generate_surrogate(data, method='phase_randomization'):
    """
    Generates surrogate data using specified method.

    Args:
        data (numpy.ndarray): Input data (channels x timepoints).
        method (str): 'phase_randomization' or 'time_shuffling'.

    Returns:
        numpy.ndarray: Surrogate data (channels x timepoints).
    """
    n_channels, n_samples = data.shape
    surrogate_data = np.zeros_like(data)

    if method == 'phase_randomization':
        for ch in range(n_channels):
            # FFT, randomize phase, and inverse FFT
            fft_vals = fft(data[ch])
            random_phases = np.exp(2j * np.pi * np.random.rand(n_samples // 2 - 1))
            fft_vals[1:n_samples // 2] *= random_phases
            fft_vals[-1:-n_samples // 2:-1] = np.conj(fft_vals[1:n_samples // 2])
            surrogate_data[ch] = np.real(ifft(fft_vals))
    elif method == 'time_shuffling':
        for ch in range(n_channels):
            surrogate_data[ch] = np.random.permutation(data[ch])
    else:
        raise ValueError("Invalid surrogate method. Use 'phase_randomization' or 'time_shuffling'.")

    return surrogate_data

eeg_data = sz_series[10]  # Replace with real EEG data

# Convert DataFrame to numpy array
sample_data = eeg_data.to_numpy()

# Compute DTF on original data
model_order = 10
dtf_original = compute_dtf(sample_data, model_order=model_order, f_sample=Fs)

# Generate surrogate data: Phase randomization
surrogate_data_phase = generate_surrogate(sample_data, method='phase_randomization')
dtf_surrogate_phase = compute_dtf(surrogate_data_phase, model_order=model_order, f_sample=Fs)

# Generate surrogate data: Time-point shuffling
#surrogate_data_shuffle = generate_surrogate(data, method='time_shuffling')
#dtf_surrogate_shuffle, _ = compute_dtf(surrogate_data_shuffle, model_order=model_order, f_sample=Fs)

# Visualization

import matplotlib.pyplot as plt
import seaborn as sns  # For better-looking heatmaps (optional)

def plot_dtf_comparison(original_dtf, surrogate_dtf):
    """
    Plots the original DTF, surrogate DTF, and their differences.

    Args:
        original_dtf (numpy.ndarray): Original DTF matrix (channels x channels).
        surrogate_dtf (numpy.ndarray): Surrogate DTF matrix (channels x channels).
    """
    # Compute the difference
    dtf_difference = original_dtf - surrogate_dtf

    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Original DTF
    sns.heatmap(original_dtf, ax=axes[0], cmap='OrRd', annot=False, cbar=True)
    axes[0].set_title("Original DTF")
    axes[0].set_xticks(np.arange(num_eegchannel) + 0.5)
    axes[0].set_xticklabels(notable_channels, rotation=45)
    axes[0].set_yticklabels(notable_channels, rotation=0)

    # Plot Surrogate DTF
    sns.heatmap(surrogate_dtf, ax=axes[1], cmap='OrRd', annot=False, cbar=True)
    axes[1].set_title("Surrogate DTF")
    axes[1].set_xticks(np.arange(num_eegchannel) + 0.5)
    axes[1].set_xticklabels(notable_channels, rotation=45)
    axes[1].set_yticklabels(notable_channels, rotation=0)

    # Plot Difference
    sns.heatmap(dtf_difference, ax=axes[2], cmap='OrRd', annot=False, cbar=True, center=0)
    axes[2].set_title("Difference (Original - Surrogate)")
    axes[2].set_xticks(np.arange(num_eegchannel) + 0.5)
    axes[2].set_xticklabels(notable_channels, rotation=45)
    axes[2].set_yticklabels(notable_channels, rotation=0)

    plt.tight_layout()
    plt.show()

# Example usage
# Assuming `original_dtf` and `surrogate_dtf` are numpy arrays of shape (n_channels, n_channels)

# Simulated example data
#plot_dtf_comparison(dtf_original, dtf_surrogate_phase)

def compute_significance(original_data, num_surrogates=300, model_order=10, f_sample=Fs, freq_band=(13, 30), method='phase_randomization'):
    """
    Computes the significance of original DTF compared to surrogate data.

    Args:
        original_data (numpy.ndarray): Input data (channels x timepoints).
        num_surrogates (int): Number of surrogate datasets to generate.
        model_order (int): Order of the MVAR model.
        f_sample (int): Sampling frequency in Hz.
        freq_band (tuple): Frequency band of interest (low, high) in Hz.
        method (str): Method for generating surrogates ('phase_randomization' or 'time_shuffling').

    Returns:
        dict: Dictionary containing:
            - 'original_dtf': Original DTF matrix.
            - 'mean_surrogate_dtf': Mean DTF matrix of surrogates.
            - 'std_surrogate_dtf': Standard deviation of surrogate DTFs.
            - 'z_scores': Z-score matrix comparing original to surrogates.
    """
    # Compute the original DTF
    original_dtf = compute_dtf(sample_data, model_order=model_order, f_sample=Fs)

    # Generate surrogates and compute their DTFs
    surrogate_dtf_list = []
    for i in range(num_surrogates):
        surrogate_data_phase = generate_surrogate(sample_data, method='phase_randomization')
        surrogate_dtf = compute_dtf(surrogate_data_phase, model_order=model_order, f_sample=Fs)
        surrogate_dtf_list.append(surrogate_dtf)

    # Convert list of surrogate DTFs to a 3D numpy array (num_surrogates x channels x channels)
    surrogate_dtf_array = np.array(surrogate_dtf_list)

    # Compute mean and standard deviation of surrogate DTFs
    mean_surrogate_dtf = np.mean(surrogate_dtf_array, axis=0)
    std_surrogate_dtf = np.std(surrogate_dtf_array, axis=0)

    # Compute z-scores
    z_scores = (original_dtf - mean_surrogate_dtf) / std_surrogate_dtf

    # Return results as a dictionary
    return {
        'original_dtf': original_dtf,
        'surrogate_dtf_array': surrogate_dtf_array,
        'mean_surrogate_dtf': mean_surrogate_dtf,
        'std_surrogate_dtf': std_surrogate_dtf,
        'z_scores': z_scores}


def identify_insignificant_connections(z_scores, channel_list, threshold=1.96):
    """
    Identifies insignificant DTF connections where z-scores are below the threshold.

    Args:
        z_scores (numpy.ndarray): Z-score matrix.
        channel_list (list): List of channel names.
        threshold (float): Threshold for significance (default: 1.96).

    Returns:
        list: List of tuples (channel_from, channel_to, z_score) for insignificant connections.
    """
    insignificant_connections = []
    num_channels = len(channel_list)

    for i in range(num_channels):
        for j in range(num_channels):
            if i != j and abs(z_scores[i, j]) < threshold:  # Ignore diagonal and significant connections
                insignificant_connections.append((channel_list[i], channel_list[j], z_scores[i, j]))

    return insignificant_connections


results = compute_significance(original_data=sample_data, num_surrogates=300, model_order=10, f_sample=Fs, freq_band=(13, 30), method='phase_randomization')

insignificant_connections = identify_insignificant_connections(z_scores=results["z_scores"], channel_list=notable_channels, threshold=1.96)


def plot_dtf_histogram(original_dtf_value, surrogate_dtf_values, channel_from, channel_to):
    """
    Plots a histogram comparing the surrogate DTF values with the original DTF value.

    Args:
        original_dtf_value (float): Original DTF value for the connection.
        surrogate_dtf_values (numpy.ndarray): Surrogate DTF values for the connection.
        channel_from (str): Name of the source channel.
        channel_to (str): Name of the target channel.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(surrogate_dtf_values, bins=20, color='skyblue', edgecolor='k', alpha=0.7, label="Surrogate DTFs")
    plt.axvline(original_dtf_value, color='red', linestyle='--', linewidth=2, label="Original DTF")

    from scipy.stats import norm
    mean_surrogate = np.mean(surrogate_dtf_values)
    std_surrogate = np.std(surrogate_dtf_values)
    x = np.linspace(min(surrogate_dtf_values), max(surrogate_dtf_values), 100)
    y = norm.pdf(x, mean_surrogate, std_surrogate)  # Normal distribution curve

    plt.title(f"DTF Distribution for {channel_from} → {channel_to}")
    plt.xlabel("DTF Value")
    plt.ylabel("Frequency")

    plt.legend()
    plt.tight_layout()
    plt.show()

specific_connection = ('LA1', 'LB1')  # The connection you are interested in
# Extract indices for the specific connection
channel_from, channel_to = specific_connection

i = notable_channels.index(channel_from)
j = notable_channels.index(channel_to)

# Extract original DTF value and surrogate DTF values
original_dtf_value = results['original_dtf'][i, j]
surrogate_dtf_values = results['surrogate_dtf_array'][:, i, j]

# Plot histogram for the specific connection
def plot_specific_connection_histogram(original_dtf_value, surrogate_dtf_values, channel_from, channel_to):
    """
    Plots a histogram comparing surrogate DTF values with the original DTF value for a specific connection.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(surrogate_dtf_values, bins=20, color='skyblue', edgecolor='k', alpha=0.7, label="Surrogate DTFs")
    plt.axvline(original_dtf_value, color='red', linestyle='--', linewidth=2, label="Original DTF")
    from scipy.stats import norm
    mean_surrogate = np.mean(surrogate_dtf_values)
    std_surrogate = np.std(surrogate_dtf_values)
    x = np.linspace(min(surrogate_dtf_values), max(surrogate_dtf_values), 100)
    y = norm.pdf(x, mean_surrogate, std_surrogate)  # Normal distribution curve
    plt.plot(x, y, color='black', linewidth=2, linestyle='-', label=r"$Z$-scored Normal Distribution")

    plt.title(f"DTF Distribution for {channel_from} → {channel_to} with 300 times shuffle")

    plt.xlabel(r"$\beta$ Band DTF Values in $Z$-score Normalized")
    plt.ylabel("Distribution Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function to plot the histogram

plot_specific_connection_histogram(original_dtf_value, surrogate_dtf_values, channel_from, channel_to)




def analyze_insignificant_connections(original_data, num_surrogates=100, model_order=10, f_sample=1000, freq_band=(13, 30), method='phase_randomization', threshold=1.96, channel_list=None):
    """
    Identifies insignificant DTF connections and plots histograms of DTF distributions.

    Args:
        original_data (numpy.ndarray): Input data (channels x timepoints).
        num_surrogates (int): Number of surrogate datasets.
        model_order (int): Order of the MVAR model.
        f_sample (int): Sampling frequency in Hz.
        freq_band (tuple): Frequency band of interest.
        method (str): Surrogate generation method.
        threshold (float): Z-score threshold for significance.
        channel_list (list): List of channel names.

    Returns:
        None
    """
    # Compute significance
    results = compute_significance(original_data, num_surrogates=num_surrogates, model_order=model_order, f_sample=f_sample, freq_band=freq_band, method=method)
    original_dtf = results['original_dtf']
    surrogate_dtf_array = np.array(results['mean_surrogate_dtf'])
    z_scores = results['z_scores']

    # Identify insignificant connections
    insignificant_connections = identify_insignificant_connections(z_scores, channel_list, threshold)
    print("\nInsignificant Connections (Z-score < {:.2f}):".format(threshold))
    for conn in insignificant_connections:
        print(f"{conn[0]} → {conn[1]}: Z-score = {conn[2]:.2f}")

    # Plot histograms for each insignificant connection
    surrogate_dtf_list = np.array(results['mean_surrogate_dtf'])
    for conn in insignificant_connections:
        i = channel_list.index(conn[0])
        j = channel_list.index(conn[1])
        original_dtf_value = original_dtf[i, j]
        surrogate_dtf_values = surrogate_dtf_list[:, i, j]  # All surrogate DTF values for this connection
        plot_dtf_histogram(original_dtf_value, surrogate_dtf_values, conn[0], conn[1])

    return 1


