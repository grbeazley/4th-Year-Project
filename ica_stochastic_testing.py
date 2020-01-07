from utilities import load_data, log_returns, normalise, is_normal, scale_uni
from ica import whiten_data, comp_ica, rhd, adj_rhd
from stochastic_volatility import gen_univ_sto_vol, gen_multi_sto_vol

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


def plot_compare(model_data, true_data):
    # Allows fast visual comparison between volatility plots
    # Assumes input does not require conversion
    if model_data.shape != true_data.shape:
        print("Cannot Compare data with different shapes")
        return

    long_axis = np.argmax(model_data.shape)
    short_axis = np.argmin(model_data.shape)
    num_values = model_data.shape[long_axis]
    idxs = np.arange(num_values)

    for series_idx in range(model_data.shape[short_axis]):
        plt.figure()
        plt.scatter(idxs, model_data[series_idx, :], s=2)
        plt.scatter(idxs, true_data[series_idx, :], s=2)
        title = str(series_idx) + ": Mean Squared Error: " \
                + str(np.mean(np.square(true_data[series_idx, :] - model_data[series_idx, :])))
        plt.title(title)
        plt.legend(['Model', 'True'])


def recover(time_series, start_value=1):
    # Inverts the log returns operation
    long_axis = np.argmax(time_series.shape)
    factors = np.exp(time_series)
    # Set first value in series to the start value
    factors[:, 0] = start_value
    return np.cumprod(factors, axis=long_axis)


# Set the random seed for reproducibility
np.random.seed(0)

# Boolean parameter for switching
ftse = False
multi = True
if ftse:
    multi = False

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

    # Compute centred data
    data_norm = normalise(data_returns)

    # Store a reference variable for use in comparison
    data_reference = data_norm

    # Compute whitened data
    calc_data, whiten_matrix = whiten_data(data_norm)
elif multi:
    num = 2500
    num_series = 5

    # Generate pseudo random phi matrix around a prior
    add_fac, mult_fac = scale_uni(0.8, 0.99)
    diag_val_phi = (np.random.rand(5) + add_fac) / mult_fac
    phi = np.diag(diag_val_phi)
    phi = phi + np.random.randn(num_series, num_series) * (1-np.max(abs(diag_val_phi)))/5

    # Generate pseudo random sigma eta matrix around a prior
    add_fac, mult_fac = scale_uni(0.3, 0.7)
    diag_val_eta = (np.random.rand(5) + add_fac) / mult_fac
    sigma_eta = np.diag(diag_val_eta)
    low_tri = np.tril(np.random.randn(num_series, num_series) * (np.max(abs(diag_val_eta)))/5)
    sigma_eta = sigma_eta + low_tri + low_tri.T - 2*np.diag(np.diag(low_tri))

    data = gen_multi_sto_vol(num, num_series, phi=phi, var_latent=sigma_eta, var_observed=0.2)
    # Compute centred data
    data_reference = normalise(data)

    # Compute whitened data
    data_whitened, whiten_matrix = whiten_data(data_reference)

    calc_data = data_whitened

else:
    # Generate truck and trailer series
    N = 2400
    prim_data = gen_univ_sto_vol(N, a=0.99, b=0.2, c=0.1)
    noise = 0.1 * np.random.randn(N)
    data = np.array([prim_data,
                     prim_data + 0.1 * np.random.randn(N),
                     prim_data + 0.1 * np.random.randn(N),
                     0.5 * np.random.randn(N)])

    num_series = 4

    # Compute centred data
    data_reference = normalise(data)

    # Compute whitened data
    data_whitened, whiten_matrix = whiten_data(data_reference)

    calc_data = data_whitened

# Test Gaussianity of data
kurts = is_normal(calc_data)

# Compute independent components
icas, mix_matrix = comp_ica(calc_data)

plt.figure()
num_points = len(icas[0, :])
for i in range(num_series):
    plt.subplot(num_series, 1, i + 1)
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
        rhds[i] += rhd(model_recovered[j, :], data_reference[j, :])
        adj_rhds[i] += adj_rhd(model_recovered[j, :], data_reference[j, :])

plt.figure()
plt.plot(rhds / num_series)
plt.xlabel('Component Index')
plt.ylabel('Relative Hamming Distance')

# plt.figure()
# plt.plot(adj_rhds/num_series)
# plt.xlabel('Component Index')
# plt.ylabel('Adjusted Relative Hamming Distance')

# plot_sto_vol(icas, None)


# Plot Visual Comparison when most useful component removed
i = np.argmax(rhds)

mask = np.ones(num_series, dtype=bool)
mask[i] = False
invW_trunc = mix_matrix[:, mask]
model = np.dot(invW_trunc, icas[mask, :])

# Un Whiten the result of the de-mixing
whiten_inv = np.linalg.inv(whiten_matrix)
model_recovered = np.dot(whiten_inv, model)

plot_compare(model_recovered, data_reference)
