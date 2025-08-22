import networkx as nx
import numpy as np
import pandas as pd
import itertools


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

## assuming that a dataframe is the input, we need channel index
def Channel_check(input_df, channel_list):
    column_idx = input_df.columns.tolist()
    # to ensure the channel list be consistent
    if column_idx == channel_list:
        return channel_list
        

# by default, we consider model order 10; from MVAR to H transfer matrix
# input dataframe, model_order I usually choose 10, freq is the individual freq bin, Fs is sampling rate=1000
def AR_to_H(input_df,model_order=10, freq, Fs=1000):

    k_ar = moder_order
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

    df_freq = pd.DataFrame(matrix_normalized, columns=notable_channels, index=notable_channels)

    return df_freq

### summarize over frequency bands, band is in the format [f1, f2] to indicate ranges

def Sub_band(input_df, band):
    matrices_list = []
    input_df = input_df
    for f in range(band[0],band[-1]):
        df_f = AR_to_H(input_df= input_df,freq=f)
        matrices_list.append(df_f)

    #matrix_total = sum(matrices_list)  # use pd operation to sum
    matrix_band = pd.concat(matrices_list).groupby(level=0).sum()
    return matrix_band