from utilities import load_data, log_returns, normalise, is_normal
from ica import whiten_data, comp_ica, rhd, adj_rhd
from stochastic_volatility import gen_univ_sto_vol

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_sto_vol(time_series, conv_type=None):
    if len(time_series.shape) != 1:
        if np.argmax(time_series.shape) != 1:
            # Input matrix is N x m so transpose
            time_series = time_series.T
        for series_idx in range(time_series.shape[0]):
            plt.figure()
            if conv_type == 'log':
                # Use log returns
                ratio = log_returns(time_series[series_idx, :])
            elif conv_type == 'div':
                # Use just a ratio
                ratio = time_series[series_idx, 1:] / time_series[series_idx, :-1]
            else:
                ratio = time_series[series_idx, :]
            # Plot scatter graphs for each time series
            plt.scatter(np.arange(len(ratio)), ratio, s=5)
    else:
        # Single dimension passed
        if conv_type == 'log':
            ratio = log_returns(time_series)
        else:
            ratio = time_series
        plt.scatter(np.arange(len(ratio)), ratio, s=5)


def recover(time_series):
    long_axis = np.argmax(time_series.shape)
    return np.cumsum(time_series, axis=long_axis)


# Set the random seed for reproducibility
np.random.seed(10)

# Boolean parameter for switching
ftse = False

if ftse:
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
             # "United Kingdom 1-Year Bond Yield Historical Data.csv": ['Price'],
             "United Kingdom 3-Month Bond Yield Historical Data.csv": ['Price'],
             "United Kingdom 30-Year Bond Yield Historical Data.csv": ['Price'],
             "VOD.L.csv": ['Adj Close'],
             }

    data_frame = load_data(stem, names)

    # Take only series values from the data frame
    data = data_frame.values[1:, :].astype('float')

    # Take difference
    data_returns = np.log(data[:, 1:] / data[:, :-1])

    # Calculate the number of time series
    num_series = len(data[:, 0])

    # Take the dates from the data frame for plotting
    dates = data_frame.values[0, :]

    # Compute whitened data
    data_whitened = whiten_data(data)

    # Compute centred data
    data_norm = normalise(data)

    calc_data, whiten_matrix = whiten_data(normalise(data_returns))

else:
    # Generate truck and trailer series
    N = 2400
    prim_data = gen_univ_sto_vol(N, a=0.99, b=0.2, c=0.1)
    data = np.array([prim_data,
                     prim_data + 0.1*np.random.randn(N),
                     prim_data + 0.1*np.random.randn(N),
                     0.5*np.random.randn(N)])

    num_series = 4

    # Compute centred data
    data_norm = normalise(data)

    # Compute whitened data
    data_whitened, whiten_matrix = whiten_data(data)

    calc_data = data_whitened


# Test Gaussianity of data
kurts = is_normal(calc_data)

# Compute independent components
icas, mix_matrix = comp_ica(calc_data)

plt.figure()
num_points = len(icas[0, :])
for i in range(num_series):
    plt.subplot(num_series, 1, i+1)
    plt.scatter(np.arange(num_points), icas[i, :], s=1)
    plt.ylim([-10, 10])

plt.show()

rhds = np.zeros(num_series)
adj_rhds = np.zeros(num_series)

for i in range(num_series):
    # Check all RHD values for different combinations
    mask = np.ones(num_series, dtype=bool)
    mask[i] = False
    invW_trunc = mix_matrix[:, mask]
    model = np.dot(invW_trunc, icas[mask, :])

    # Un Whiten the result of the de-mixing
    whiten_inv = np.linalg.inv(whiten_matrix)
    model_recovered = np.dot(whiten_inv, model)

    for j in range(num_series):
        rhds[i] += rhd(model_recovered[j, :], data_norm[j, :])
        adj_rhds[i] += adj_rhd(model_recovered[j, :], data_norm[j, :])


plt.figure()
plt.plot(rhds/num_series)
plt.xlabel('Component Index')
plt.ylabel('Relative Hamming Distance')

plt.figure()
plt.plot(adj_rhds/num_series)
plt.xlabel('Component Index')
plt.ylabel('Adjusted Relative Hamming Distance')

# plot_sto_vol(icas, None)

