from utilities import load_data, normalise, is_normal, scale_uni, moving_average
from ica import whiten_data, comp_ica, rhd, adj_rhd
from stochastic_volatility import gen_univ_sto_vol, gen_multi_sto_vol
from plot_utils import plot_sto_vol, plot_compare, plot_components

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def recover(time_series, start_value=1):
    # Inverts the log returns operation
    long_axis = np.argmax(time_series.shape)
    factors = np.exp(time_series)
    # Set first value in series to the start value
    factors[:, 0] = start_value
    return np.cumprod(factors, axis=long_axis)


# Set the random seed for reproducibility
np.random.seed(1)

# Boolean parameter for switching
ftse = False
multi = False
truck_trailer = False
arma = True
if ftse:
    multi = False
    truck_trailer = False
    arma = False

if ftse:
    stem = "Data Sets\\FTSEICA_sto_vol\\"

    names = {"FTSE 250 Historical Data.csv": ['Price'],
             "FTSE 100 Historical Data.csv": ['Price', 'High', 'Low'],
             #"BA.L.csv": ['Adj Close'],
             #"BATS.L.csv": ['Adj Close'],
             #"BP.csv": ['Adj Close'],
             #"GSK.L.csv": ['Adj Close'],
             #"LLOY.L.csv": ['Adj Close'],
             #"NG.L.csv": ['Adj Close'],
             #"ULVR.L.csv": ['Adj Close'],
             # "United Kingdom 1-Year Bond Yield Historical Data.csv": ['Price'],
             "United Kingdom 3-Month Bond Yield Historical Data.csv": ['Price'],
             #"United Kingdom 30-Year Bond Yield Historical Data.csv": ['Price'],
             #"VOD.L.csv": ['Adj Close'],
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

    data_hidden = moving_average(np.log(np.abs(normalise(data_returns)+1e-10)), n=1)
    algo = "energyICA"

    # Compute centred data
    data_norm = normalise(data_hidden)

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

elif truck_trailer:
    # Generate truck and trailer series, using one driving process and different observations
    N = 2400
    # prim_data = gen_univ_sto_vol(N, a=0.99, b=0.2, c=0.1)
    mu, a, b, c = 0, 0.99, 0.6, 0.1
    noise_var = 0.08
    x_prev = np.random.randn()

    trajectory_hidden = np.zeros(N)

    # Create array of hidden state variables
    for i in range(N):
        x = mu + a*(x_prev-mu) + np.sqrt(b)*np.random.randn()
        trajectory_hidden[i] = x
        x_prev = x

    def hidden_to_observed(trajectory, c):
        N = len(trajectory)
        trajectory_obs = np.zeros(N)
        for j in range(N):
            trajectory_obs[j] = np.sqrt(c) * np.exp(trajectory[j] / 2) * np.random.randn()

        return trajectory_obs

    data = np.array([hidden_to_observed(trajectory_hidden, c),
                     hidden_to_observed(trajectory_hidden, c)
                     ])

    num_series = 2

    # Compute centred data
    data_reference = np.log(np.abs(data))

    n_p = 1
    print(n_p)
    data_reference = moving_average(data_reference, n=n_p)
    algo = "energyICA"

    # Compute whitened data
    data_whitened, whiten_matrix = whiten_data(data_reference)

    calc_data = data_reference

elif arma:
    N = 2400
    # prim_data = gen_univ_sto_vol(N, a=0.99, b=0.2, c=0.1)
    mu, a, b, c = 0, 0.99, 0.6, 0.1
    noise_var = 0.08
    num_series = 2

    trajectory = np.zeros([num_series, N])

    for k in range(num_series):
        x_prev = np.random.randn()
        for i in range(N):
            x = mu + a*(x_prev-mu) + np.sqrt(b)*np.random.randn()
            trajectory[k, i] = x
            x_prev = x

    plot_components(trajectory, 'True Series')

    true_mix_matrix = np.array([[0.6, 0.5],
                                [0.5, 0.6]])

    # De-correlates the rows of the mixing matrix
    true_mix_matrix[1, :] = true_mix_matrix[1, :] - np.dot(np.dot(true_mix_matrix[1, :], true_mix_matrix[0, :]),
                                                           true_mix_matrix[0, :])

    true_mix_matrix_norm = true_mix_matrix.T / np.sqrt((true_mix_matrix ** 2).sum(axis=1))

    true_mix_matrix_norm = true_mix_matrix_norm.T

    data = np.dot(true_mix_matrix_norm, trajectory)

    # Compute centred data
    data_reference = normalise(data)
    algo = "fastICA"

    # Compute whitened data
    data_whitened, whiten_matrix = whiten_data(data_reference)

    calc_data = data_whitened



else:
    # Generate two separate series, using one driving process and different observations
    N = 5000
    # prim_data = gen_univ_sto_vol(N, a=0.99, b=0.2, c=0.1)
    mu, a, b, c = 0, 0.98, 0.3, 0.05
    noise_var = 0.08

    num_series = 2

    data_unmixed = np.zeros([num_series, N])

    for i in range(num_series):
        data_unmixed[i, :] = gen_univ_sto_vol(N, mu=mu, a=a, b=b, c=c)[1:]

    plot_components(data_unmixed, 'True Series')

    true_mix_matrix = np.random.rand(num_series, num_series) - 0.5

    true_mix_matrix = np.array([[1, 0],
                                [1, 0.1]])

    # De-correlates the rows of the mixing matrix
    true_mix_matrix[1, :] = true_mix_matrix[1, :] - np.dot(np.dot(true_mix_matrix[1, :], true_mix_matrix[0, :]), true_mix_matrix[0, :])

    true_mix_matrix_norm = true_mix_matrix.T / np.sqrt((true_mix_matrix ** 2).sum(axis=1))

    true_mix_matrix_norm = true_mix_matrix_norm.T

    data = np.dot(true_mix_matrix_norm, data_unmixed)

    # Compute centred data
    data_reference = normalise(data)

    # Compute whitened data
    data_whitened, whiten_matrix = whiten_data(data_reference)

    calc_data = data_whitened

plot_components(data_reference, 'Input Data')

# Test Gaussianity of data
kurtosis_values = is_normal(calc_data)

# Compute independent components
ics, mix_matrix = comp_ica(calc_data, algorithm=algo)

plot_components(ics, 'Independent Components')

rhds = np.zeros(num_series)
adj_rhds = np.zeros(num_series)

for i in range(num_series):
    # Check all RHD values for different combinations
    mask = np.ones(num_series, dtype=bool)
    mask[i] = False
    invW_trunc = mix_matrix[:, mask]
    model = np.dot(invW_trunc, ics[mask, :])

    # Un Whiten the result of the de-mixing
    whiten_inv = np.linalg.inv(whiten_matrix)
    model_recovered = np.dot(whiten_inv, model)

    for j in range(num_series):
        rhds[i] += rhd(model_recovered[j, :], data_reference[j, :])
        adj_rhds[i] += adj_rhd(model_recovered[j, :], data_reference[j, :])

# plt.figure()
# plt.plot(rhds / num_series)
# plt.xlabel('Component Index')
# plt.ylabel('Relative Hamming Distance')

# plot_sto_vol(ics, None)

# Plot Visual Comparison when most useful component removed
i = np.argmin(rhds)

mask = np.ones(num_series, dtype=bool)
mask[i] = False
invW_trunc = mix_matrix[:, mask]
model = np.dot(invW_trunc, ics[mask, :])

# Un Whiten the result of the de-mixing
whiten_inv = np.linalg.inv(whiten_matrix)
model_recovered = np.dot(whiten_inv, model)

if not(truck_trailer or ftse or multi):
    print("Distance: ", np.sum(np.abs(np.dot(np.linalg.inv(mix_matrix), true_mix_matrix_norm))) - num_series)

plot_compare(model_recovered, data_reference)

print(max(rhds)/num_series, min(rhds)/num_series)