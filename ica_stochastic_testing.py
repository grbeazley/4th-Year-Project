from __future__ import absolute_import, division, print_function, unicode_literals
#import tensorflow as tf
#import matplotlib as mpl
#import os

from utilities import load_data, log_returns, normalise
from ica import whiten_data, comp_ica, rhd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_sto_vol(time_series):
    if len(time_series.shape) != 1:
        if np.argmax(time_series.shape) != 1:
            # Input matrix is N x m so transpose
            time_series = time_series.T
        for series_idx in range(time_series.shape[0]):
            plt.figure(series_idx)
            lgr = log_returns(time_series[series_idx, :])
            plt.scatter(np.arange(len(lgr)), lgr, s=5)
    else:
        lgr = log_returns(time_series)
        plt.scatter(np.arange(len(lgr)), lgr, s=5)


def recover(time_series):
    long_axis = np.argmax(time_series.shape)
    return np.cumsum(time_series, axis=long_axis)


# Set the random seed for reproducibility
np.random.seed(0)

stem = "Data Sets\\FTSEICA_sto_vol\\"

names = {"FTSE 250 Historical Data.csv": ['Price'],
         "FTSE 100 Historical Data.csv": ['Price', 'High', 'Low'],
         "BA.L.csv": ['Adj Close'],
         "BATS.L.csv": ['Adj Close'],
         "BP.csv": ['Adj Close'],
         "GSK.L.csv": ['Adj Close'],
         "LLOY.L.csv": ['Adj Close'],
         "NG.L.csv": ['Adj Close'],
         "ULVR.L.csv": ['Adj Close'],
         "United Kingdom 1-Year Bond Yield Historical Data.csv": ['Price'],
         "United Kingdom 3-Month Bond Yield Historical Data.csv": ['Price'],
         "United Kingdom 30-Year Bond Yield Historical Data.csv": ['Price'],
         "VOD.L.csv": ['Adj Close'],
         }

data_frame = load_data(stem, names)

# Take only series values from the data frame
data = data_frame.values[1:, :].astype('float')

# Take difference
data_returns = data[:, :-1] - data[:, 1:]

# Calculate the number of time series
num_series = len(data[:, 0])

# Take the dates from the data frame for plotting
dates = data_frame.values[0, :]

# Compute whitened data
data_whitened = whiten_data(data)

# Compute centred data
data_norm = normalise(data)

calc_data = whiten_data(data_returns)

# Compute independent components
icas, mix_matrix = comp_ica(calc_data)

plt.figure(0)
for i in range(num_series):
    plt.subplot(num_series, 1, i+1)
    plt.plot(icas[:, i])
    plt.ylim([-10, 10])

plt.show()

# IC_order = []
# for k in range(num_columns):
#     # Check all combinations, decreasing each time
#     rhds = np.zeros(num_columns-k)
#     mask = np.ones(num_columns, dtype=bool)
#     mask[IC_order] = False
#     invW_trunc = invW.T[:, mask]
#     unMixed_trunc = unMixed[:, mask]
#     # print(mask)
#
#     for i in range(num_columns-k):
#         # Check RHD by dropping one of the remaining ICs in turn
#         mask_iter = np.ones(num_columns-k, dtype=bool)
#         mask_iter[i] = False
#         invW_trunc_iter = invW_trunc[:, mask_iter]
#         model = np.dot(invW_trunc_iter, unMixed_trunc[:, mask_iter].T)
#         for j in range(num_columns):
#             rhds[i] += rhd(model[j, :], Xw[j, :])
#
#     plt.figure(k+1)
#     plt.plot(rhds)
#     plt.xlabel('Component Index')
#     plt.ylabel('Relative Hamming Distance')
#     index = np.argmax(rhds)
#     print(index)
#     if isinstance(index, list):
#         index = index[0]
#     IC_order.append(index)

rhds = np.zeros(num_series)
for i in range(num_series):
    # Check all RHD values for different combinations
    mask = np.ones(num_series, dtype=bool)
    mask[i] = False
    invW_trunc = mix_matrix.T[:, mask]
    model = np.dot(invW_trunc, icas[:, mask].T)
    for j in range(num_series):
        rhds[i] += rhd(model[j, :], calc_data[j, :])

plt.figure(15)
plt.plot(rhds)
plt.xlabel('Component Index')
plt.ylabel('Relative Hamming Distance')

