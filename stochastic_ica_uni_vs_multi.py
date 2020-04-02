import numpy as np
from ica import whiten_data, comp_ica, rhd
from plot_utils import plot_components, plot_compare, plot
from stochastic_volatility import gen_multi_sto_vol, gen_univ_sto_vol, gen_univ_mrkv
from utilities import normalise, is_normal, scale_uni
from particle_filter_gradient_lg_nrml import ParticleFilter as lg_nrml_ParticleFilter
from particle_filter_gradient_gamma import ParticleFilter as gamma_ParticleFilter

np.random.seed(0)

# Create 2400 training and 100 test
train = 1000
test = 100
num = train + test
num_series = 2

# Generate pseudo random phi matrix around a prior
# add_fac, mult_fac = scale_uni(0.7, 0.9)
# diag_val_phi = (np.random.rand(num_series) + add_fac) / mult_fac
# phi = np.diag(diag_val_phi)
# phi = phi + (np.random.rand(num_series, num_series) - 0.5) * 4 * ((1-np.max(diag_val_phi))/num_series)

# phi = np.array([[0.75, 0.25, 0, 0],
#                 [0.25, 0.75, 0, 0],
#                 [0.25, 0.25, 0.25, 0.25],
#                 [0, 0, 0.75, 0.25]], dtype=float)
#
# phi *= 0.93

# Generate pseudo random sigma eta matrix around a prior
# add_fac, mult_fac = scale_uni(0.3, 0.7)
# diag_val_eta = (np.random.rand(num_series) + add_fac) / mult_fac
# sigma_eta = np.diag(diag_val_eta)
# low_tri = np.tril(np.random.randn(num_series, num_series) * (np.max(abs(diag_val_eta)))/num_series)
# sigma_eta = sigma_eta + low_tri + low_tri.T - 2*np.diag(np.diag(low_tri))

# sigma_eta = np.eye(num_series) #* np.sqrt(0.6)
#
# data_h, data_y = gen_multi_sto_vol(num,
#                                    num_series,
#                                    phi=phi,
#                                    var_latent=sigma_eta,
#                                    var_observed=0.5,
#                                    return_hidden=True)


def hidden_to_observed(trajectory, c_var):
    N = len(trajectory)
    trajectory_obs = np.zeros(N)
    for j in range(N):
        trajectory_obs[j] = np.sqrt(c_var) * np.exp(trajectory[j] / 2) * np.random.randn()

    return trajectory_obs


# Set the random seed for reproducibility
np.random.seed(0)

# Generate truck and trailer series, using one driving process and different observations
mu, a, b, c = 0, 0.95, 1, 1
x_prev = np.random.randn()

trajectory_hidden = np.zeros(num)

# Create array of hidden state variables
for i in range(num):
    x = mu + a * (x_prev - mu) + np.sqrt(b) * np.random.randn()
    trajectory_hidden[i] = x
    x_prev = x

data_y = np.array([hidden_to_observed(trajectory_hidden, c),
                   hidden_to_observed(trajectory_hidden, c)])

data_h = np.array([trajectory_hidden, trajectory_hidden])


plot_components(data_y, 'Input Data Raw')
plot_components(data_h, 'Input Hidden State')

data_train = data_y[:, :train]
data_test = data_y[:, train:]

true_test = data_h[:, train]
true_train = data_h[:, train:]

######################### ICA ###########################
input("Press Enter to run ICA...")
data_abs = np.abs(data_train)

# Take the logs of the absolute values
data_hidden = np.log(data_abs)

# Normalise the data and store the parameters
# data_hidden_av_norm, means, stds = normalise(data_hidden, return_params=True)
means = np.mean(data_hidden, axis=1, keepdims=True)
data_hidden_av_norm = data_hidden - means

# Compute whitened data
data_whitened, whiten_matrix = whiten_data(data_hidden_av_norm)
whiten_inv = np.linalg.inv(whiten_matrix)

# plot_components(data_whitened, 'Input Data Whitened')

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
    # model_scaled = (model_correlated * stds) + means
    model_scaled = model_correlated + means

    # Undo the log step (goes back to observed process)
    model_recovered = np.exp(model_scaled)

    for j in range(num_series):
        rhds[i] += rhd(model_recovered[j, :], data_abs[j, :])

    # plot_compare(model_recovered, data_abs)
    mse[i] = np.mean(np.square(data_abs - model_recovered))

print(mse)
plot(rhds/num_series)

demix_matrix = np.dot(np.linalg.inv(mix_matrix), whiten_matrix)

############### Particle Filter Paramater Optimisation ################
input("Press Enter to run particle filter...")
N = 500

gamma = True
lg_nrml = False

# Un Whiten the result of the de-mixing
# model_correlated = np.dot(whiten_inv, ics)

# Undo the normalisation
# model_scaled = (model_correlated * stds) + mean

# Undo the log step (goes back to observed process)
# model_recovered = np.exp(model_scaled)

signal_idxs = [0]
noise_idxs = [1]
filters = {}

if gamma:

    for ica_idx in signal_idxs:
        # Calibrate a model based on the training data using a gamma distribution for noise

        # Obtain row vector from demixing matrix
        alphas = demix_matrix[ica_idx, :]
        if np.sum(alphas) < 0:
            # Majority negative so change sign
            alphas = -alphas

        # Calculate k and theta values for gamma distributed noise


        filters[ica_idx] = gamma_ParticleFilter(ics[ica_idx, :],
                                                num_particles=N,
                                                a=0.8,
                                                b=0.5,
                                                k=kk,
                                                theta=thth,
                                                learn_rate=0.001,
                                                num_iterations=15,
                                                true_hidden=true_train[ica_idx, :])

        # particle_filter_1.filter_pass()
        # particle_filter_1.plot_filter_pass()
        filters[ica_idx].calibrate_model()
        filters[ica_idx].plot_params(ica_idx)

        filters[ica_idx].adap_learn_rate = 0.0001
        filters[ica_idx].calibrate_model(50)
        filters[ica_idx].plot_params(ica_idx)
        # particle_filter_1.plot_filter_pass()

elif lg_nrml:
    pass

############################## Prediction ####################
input("Press Enter to perform prediction...")

predicted_components = {}
num_predict_steps = test

for ica_idx in signal_idxs:
    aa = float(input("Please Enter value for a"))
    bb = float(input("Please Enter value for b"))
    cc = 1.23

    x_prev = filters[ica_idx].estimate_history[-1]

    prediction = gen_univ_mrkv(test, a=aa, b=bb, c=cc, x0=x_prev)

    ic_complete = np.zeros(num)
    ic_complete[:train] = ics[ica_idx, :]
    ic_complete[train:] = prediction[1:]

    predicted_components[ica_idx] = ic_complete

for ica_idx in noise_idxs:
    # Predict a noise process based on the statistics of the training data
    noise = ics[ica_idx]
    noise_mean, noise_std = np.mean(noise), np.var(noise)

    # Assume noise can be approximated by a Gaussian
    prediction = noise_mean + (np.random.randn(num_predict_steps) * noise_std)

    ic_complete = np.zeros(num)
    ic_complete[:train] = ics[ica_idx]
    ic_complete[train:] = prediction

    predicted_components[ica_idx] = ic_complete

# Gather all components with predictions back into matrix
ics_with_predictions = np.zeros([num_series, train + num_predict_steps])
for i in range(num_series):
    # Extract the ics from the dictionary
    ics_with_predictions[i, :] = predicted_components[i]

predicted_signals = np.dot(mix_matrix, ics_with_predictions)

predicted_signals_correlated = np.dot(whiten_inv, predicted_signals)

signals_scaled = (predicted_signals_correlated * stds) + mean

# Undo the log step (goes back to observed process)
recovered_signals_exp = np.exp(signals_scaled)
