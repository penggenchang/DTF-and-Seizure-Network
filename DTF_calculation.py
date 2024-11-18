import networkx as nx
import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.cm import ScalarMappable

import sklearn
from sklearn import svm,model_selection,metrics,feature_selection
from sklearn import preprocessing, manifold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import random
import scipy
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

# channel ranking using LDA
band_id = ['delta','theta','alpha','beta','lowgamma','highgamma']

Left_channels = ['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10']
Right_channels = ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10']
notable_channels = Left_channels + Right_channels
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
#print(frame_count)

# by default, we consider model order 10; from MVAR to H transfer matrix
def AR_to_H(input_df,freq):

    k_ar = 10
    # MVAR estimate
    data_input = input_df.loc[:, notable_channels]
    data_values = data_input.to_numpy()

    model = VAR(data_values)
    results = model.fit(k_ar)
    # print(results.summary())
    coeff_matrices = getattr(results, 'params')
    # t = [len(a) for a in coeff_matrices]
    # re-organize outputs:
    #error_values = coeff_matrices[0]
    A_lists = []

    for k in range(k_ar):
        A = coeff_matrices[num_nodes * k + 1:num_nodes * k + num_nodes + 1]
        A_tmp = np.array(A)
        A_matrix = A_tmp.transpose()
        A_lists.append(A_matrix)

    idx_range = np.linspace(-2 * np.pi * 1j * 1 * freq / Fs, -2 * np.pi * 1j * k_ar * freq / Fs, num=k_ar)
    exp_range = np.exp(idx_range)

    A_sum = []

    for m in range(k_ar):
        A_m = A_lists[m]
        exp_m = exp_range[m]
        term_m = A_m * exp_m
        A_sum.append(term_m)

        # transform the list of A into array form
        # A_series = np.array(A_lists)
    A_freq = sum(A_sum)
        # transform into H matrix
    H_freq = np.linalg.inv(np.identity(num_nodes) - A_freq)
        # second, into DTF matrix
    matrix_normalized = np.zeros((num_nodes, num_nodes))
    for k in range(num_nodes):  # this is the outflow
        vec_column = H_freq[:, k]
        vec_product = np.inner(np.conjugate(vec_column), vec_column)
        vec_product = np.abs(vec_product)
        for t in range(num_nodes):
            value_tmp = H_freq[t, k]
            value_abs = np.abs(value_tmp) ** 2
            matrix_normalized[t, k] = value_abs / vec_product

    df_matrix = pd.DataFrame(matrix_normalized, columns=notable_channels, index=notable_channels)

    return df_matrix

### summarize over frequency bands

def Sub_band(input_df, band):
    matrices_list = []
    input_df = input_df
    for f in range(band[0],band[-1]):
        df_f = AR_to_H(input_df= input_df,freq=f)
        matrices_list.append(df_f)

    matrix_total = sum(matrices_list)
    return matrix_total



## new to summarize DTF sequences
def DTF_summation():
    DTF_list = []
    for frame in sz_series:
        dtf_per_frame = Sub_band(input_df=frame,band=[13,30]) # this is whole connected;
        DTF_list.append(dtf_per_frame)
    dtf_sum = sum(DTF_list)
    matrix_total = dtf_sum.div(frame_count)

    return matrix_total


