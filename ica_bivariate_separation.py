from ica import whiten_data, comp_ica, rhd
from plot_utils import plot_compare, plot_components
from utilities import normalise, is_normal, moving_average, reverse_moving_average
import matplotlib.pyplot as plt
import numpy as np


def hidden_to_observed(trajectory, cc):
    N = len(trajectory)
    trajectory_obs = np.zeros(N)
    for j in range(N):
        trajectory_obs[j] = np.sqrt(cc) * np.exp(trajectory[j] / 2) * np.random.randn()

    return trajectory_obs


# Set the random seed for reproducibility
np.random.seed(0)

# Generate truck and trailer series, using one driving process and different observations
N = 2400
num_series = 2

mu, a, b, c = 0, 0.95, 1, 1
x_prev = np.random.randn()

trajectory_hidden = np.zeros(N)

# Create array of hidden state variables
for i in range(N):
    x = mu + a * (x_prev - mu) + np.sqrt(b) * np.random.randn()
    trajectory_hidden[i] = x
    x_prev = x

data = np.array([hidden_to_observed(trajectory_hidden, c),
                 hidden_to_observed(trajectory_hidden, c)
                 ])

data_abs = np.abs(data)

# Take the logs of the absolute values
data_hidden = np.log(data_abs)

# Normalise the data and store the parameters
data_hidden_norm, mean, stds = normalise(data_hidden, return_params=True)

# Compute whitened data
data_whitened, whiten_matrix = whiten_data(data_hidden_norm)
whiten_inv = np.linalg.inv(whiten_matrix)

plot_components(data_hidden_norm, 'Input Data')

# Test Gaussianity of data
kurtosis_values = is_normal(data_whitened)

# Compute independent components
ics, mix_matrix = comp_ica(data_whitened, algorithm="energyICA")

plot_components(ics, 'Independent Components')

rhds = np.zeros(num_series)
mse = np.zeros(num_series)

for i in range(num_series):
    # Check all RHD values for different combinations
    mask = np.ones(num_series, dtype=bool)
    mask[i] = False
    invW_trunc = mix_matrix[:, mask]
    model = np.dot(invW_trunc, ics[mask, :])

    # Un Whiten the result of the de-mixing
    model_correlated = np.dot(whiten_inv, model)

    # Undo the normalisation
    model_scaled = (model_correlated * stds) + mean

    # Undo the moving average step
    # model_indvdl = reverse_moving_average(model_scaled, data_hidden[:, :av_points-1], av_points)

    # Undo the log step (goes back to observed process)
    model_recovered = np.exp(model_scaled)

    for j in range(num_series):
        rhds[i] += rhd(model_recovered[j, :], data_abs[j, :])

    # plot_compare(model_recovered, data_abs)
    # mse[i] = np.mean(np.square(data_abs - model_recovered))

    # mse_track[k] = min(mse)


# plt.figure()
# plt.plot(rhds)

# plt.xlabel('Component Index')
# plt.ylabel('Relative Hamming Distance')
