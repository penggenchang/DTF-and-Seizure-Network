import networkx as nx
import numpy as np
import pandas as pd
import itertools
import sklearn

from sklearn import svm,model_selection,metrics,feature_selection
from sklearn import preprocessing, manifold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import random
import scipy

from scipy.signal import spectrogram
from itertools import combinations
from sklearn.metrics import mean_squared_error
from scipy import signal,stats
from scipy.stats import zscore
from scipy.special import comb
import statistics
import seaborn as sns
import sails
import statsmodels as sm
from statsmodels.tsa.api import VAR


SZ_path = ["...\\Patient_XXXX\\ictal_1.csv","...\\Patient_XXXX\\ictal_2.csv",
           "...\\Patient_XXXX\\ictal_3.csv"]   # for Patient xxxx


ictal_raw = pd.read_csv(SZ_path[0],index_col=False)

num_channels = int(ictal_raw.shape[1])

Fs = 1000

def Basic_information():
    band_id = ['delta','theta','alpha','beta','lowgamma','highgamma']
    Left_channels = ['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10']
    Right_channels = ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10']
    notable_channels = Left_channels + Right_channels
    num_nodes = len(notable_channels)
    return Left_channels, Right_channels, notable_channels

_, _, notable_channels = Basic_information()
num_nodes = len(notable_channels)
# dataframe split
def Data_split(input_df, Timesize,overlap):  # Timesize in Sec.,

    win_size = Timesize * Fs
    df_partchannel = input_df.loc[:, notable_channels]
    if overlap == 50:
        num_epoch = int(2*df_partchannel.shape[0] /win_size-1)
        effect_size = int(win_size/2)
    else:
        num_epoch = df_partchannel.shape[0] / win_size  # default non-lapping
        effect_size = win_size

    df_series = []
    count = 0
    while True:
        if count > num_epoch:
            break

        df_series.append(df_partchannel.iloc[count*effect_size:count*effect_size+win_size])
        count += 1

    return df_series,count

sz_series, frame_count = Data_split(input_df = ictal_raw, Timesize=2, overlap=50)

## now starts the FDCMM. The first step is spectrogram, and the mapping
def Compute_spectrogram(input_signal, Timesize, overlap):
    num_nodes = len(input_signal.columns)
    win_size = Timesize * Fs
    mapping_all = []
    ## in default, we have 2-second frame,

    data_value = input_signal
    freq, t_value, spec_value = signal.spectrogram(data_value, Fs, nperseg=win_size, noverlap=int(overlap / 100 * win_size),
                nfft=len(Fs), scaling='spectrum', mode='complex')
    pxx = np.abs(spec_value)
    mapping_all.append(np.transpose(pxx))
    df_mapping = pd.DataFrame(mapping_all)

    return spec_value, df_mapping


### L is libarry length, we defaultly set as L = 1000
Lib_len = 1000
def xmap(X, Y, X_M, Y_M, freq_L, lib_L):

    # Replace with the actual logic for xmap
    X_MY = np.zeros_like(X)  # Replace with logic
    Y_MX = np.zeros_like(Y)  # Replace with logic
    X1 = X  # Replace with logic
    Y1 = Y  # Replace with logic
    return X_MY, Y_MX, X1, Y1

def compute_CCM(input_df,Fs, freq_band, win_size, overlap):
    N_data, num_nodes = input_df.shape

    # Store spectrograms and processed arrays
    MY_all = []
    Y_all = []

    for node in notable_channels:
        data_node = input_df.loc[:,node]
        Y_all.append(data_node)

        # single node
        f, t, s = spectrogram(data_node, Fs, nperseg=win_size, noverlap=int(overlap / 100 * win_size),
                              nfft=len(freq_band), scaling='spectrum', mode='complex')
        pxx = np.abs(s)
        MY_all.append(pxx.T)

    # Prepare to compute CCM between node pairs
    node_pairs = list(combinations(range(num_nodes), 2))
    rho_ij = np.zeros(len(node_pairs))
    rho_ji = np.zeros(len(node_pairs))

    # Compute CCM for all pairs
    for idx, (i, j) in enumerate(node_pairs):
        X = Y_all[i]
        Y = Y_all[j]
        MX = MY_all[i]
        MY = MY_all[j]

        # Use xmap logic to compute cross-mapped reconstructions (implement separately)
        X_MY, Y_MX, X1, Y1 = xmap(X, Y, MX, MY, freq_L=len(freq_band), lib_L=Lib_len)

        # Compute rho values
        rho_X = max(0, 1 - (mean_squared_error(X_MY, X1) / np.var(X1)))
        rho_Y = max(0, 1 - (mean_squared_error(Y_MX, Y1) / np.var(Y1)))

        rho_ij[idx] = rho_X
        rho_ji[idx] = rho_Y

    # Populate rho_all matrix
    rho_all = np.zeros((num_nodes, num_nodes))
    for idx, (i, j) in enumerate(node_pairs):
        rho_all[i, j] = rho_ij[idx]
        rho_all[j, i] = rho_ji[idx]

    return rho_all


# Example placeholder for xmap function (to be implemented)



