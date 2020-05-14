import numpy as np
from ica import whiten_data, comp_ica, rhd
from plot_utils import *
from stochastic_volatility import *
from utilities import normalise, is_normal, scale_uni, load_data, moving_average
from particle_filter_gamma import ParticleFilterGamma
from pdfs import comp_k_theta_from_alphas
from data_gen import *

# Set the random seed for reproducibility
np.random.seed(0)

######################## Import #################################
# Create 2400 training and 100 test
train = 1000
test = 100
num = train + test
num_series = 4


# data_h, data_y = load_bivariate(num)
data_h, data_y = load_msv(num, num_series)
# data_y, dates = load_oil(plot_comps=False)
# data_h = np.ones_like(data_y)

#################### Pre-processing #######################
plot_components(data_y, 'Input Data Raw')
plot_components(data_h, 'Input Hidden State')

data_train = data_y[:, :train] + 1e-6
data_test = data_y[:, train-1:] + 1e-6

true_train = data_h[:, :train]
true_test = data_h[:, train-1:]

######################### ICA ###########################
# input("Press Enter to run ICA...")
data_abs = np.abs(data_train)

# Take the logs of the absolute values
data_hidden = np.log(data_abs)
# data_hidden = data_train

plot_components(data_hidden, 'Input Hidden')

# Normalise the data and store the parameters
# data_hidden_av_norm, means, stds = normalise(data_hidden, return_params=True)
# means = np.mean(data_hidden, axis=1, keepdims=True)
means = np.zeros([num_series, 1])
data_hidden_av_norm = data_hidden - means
# data_whitened = data_hidden

# Compute whitened data
# data_whitened, whiten_matrix = whiten_data(data_hidden_av_norm)
data_whitened, whiten_matrix = data_hidden_av_norm, np.eye(num_series)
whiten_inv = np.linalg.inv(whiten_matrix)

# plot_components(data_whitened, 'Input Data Whitened')

# Test Gaussianity of data
kurtosis_values = is_normal(data_whitened)

# Number of dimensions to find
num_ics = 2

# Compute independent components
ics, unmix_matrix = comp_ica(data_whitened, algorithm="energyICA", reduce_dims=num_series-num_ics)

mix_matrix = np.linalg.pinv(unmix_matrix)

demix_matrix = np.dot(unmix_matrix, whiten_matrix)

plot_components(ics, 'Independent Components')


rhds = np.zeros(num_ics)
mse = np.zeros(num_ics)

for i in range(num_ics):
    # Check all RHD values for different combinations
    mask = np.ones(num_ics, dtype=bool)
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
bar(mse)
plt.pause(0.01)

demix_matrix = np.dot(unmix_matrix, whiten_matrix)

############### Particle Filter Paramater Optimisation ################
input("Press Enter to run particle filter...")
N = 1000


signal_idxs = [0]
noise_idxs = [1]
filters = {}

for ica_idx in signal_idxs:
    # Calibrate a model based on the training data using a gamma distribution for noise

    # Obtain row vector from demixing matrix
    alphas = demix_matrix[ica_idx, :]
    sgn = 1
    if np.sum(alphas) < 0:
        # Majority negative so change sign
        alphas = -alphas
        sgn = -1

    # Calculate k and theta values for gamma distributed noise
    kk, thth = comp_k_theta_from_alphas(alphas)
    true_hidden = np.dot(alphas, true_train)

    hidden = sgn*ics[ica_idx, :]
    # hidden = (sgn*ics[ica_idx, :]) - np.mean(sgn*ics[ica_idx, :])

    filters[ica_idx] = ParticleFilterGamma(true_obs=np.exp(hidden)*7,
                                           num_particles=N,
                                           a=0.6,
                                           b=0.4,
                                           c=1,
                                           correction=np.sum(alphas),
                                           k=kk,
                                           theta=thth,
                                           learn_rate=0.2/train,
                                           num_iterations=150,
                                           true_hidden=true_hidden,
                                           learn_a=0.5,
                                           learn_b=10,
                                           learn_c=10,
                                           multi=True)

    # particle_filter_1.filter_pass()
    # filters[ica_idx].plot_filter_pass()
    filters[ica_idx].calibrate_model()
    filters[ica_idx].plot_params(ica_idx)

    # filters[ica_idx].adap_learn_rate = 0.01/train
    # filters[ica_idx].calibrate_model(50)
    # filters[ica_idx].plot_params(ica_idx)



############################## Prediction ######################
input("Press Enter to perform prediction...")

predicted_components = {}
num_predict_steps = test

for ica_idx in signal_idxs:
    aa = float(input("Please Enter value for phi"))
    bb = float(input("Please Enter value for sigma^2"))
    cc = float(input("Please Enter value for beta^2"))

    x_prev = filters[ica_idx].estimate_history[-1]

    prediction = predict_univ_sto_vol(1000, test, aa, bb, cc)

    ic_complete = np.zeros(num)
    ic_complete[:train] = ics[ica_idx, :]
    ic_complete[train:] = prediction[1:]

    predicted_components[ica_idx] = ic_complete

    rolling_confidence(prediction, num_steps=6)

# for ica_idx in noise_idxs:
#     # Predict a noise process based on the statistics of the training data
#     noise = ics[ica_idx]
#     noise_mean, noise_std = np.mean(noise), np.var(noise)
#
#     # Assume noise can be approximated by a Gaussian
#     prediction = noise_mean + (np.random.randn(num_predict_steps) * noise_std)
#
#     ic_complete = np.zeros(num)
#     ic_complete[:train] = ics[ica_idx]
#     ic_complete[train:] = prediction
#
#     predicted_components[ica_idx] = ic_complete

# Gather all components with predictions back into matrix
# ics_with_predictions = np.zeros([num_series, train + num_predict_steps])
# for i in range(num_series):
#     # Extract the ics from the dictionary
#     ics_with_predictions[i, :] = predicted_components[i]
#
# predicted_signals = np.dot(mix_matrix, ics_with_predictions)
#
# predicted_signals_correlated = np.dot(whiten_inv, predicted_signals)
#
# signals_scaled = (predicted_signals_correlated * stds) + mean
#
# # Undo the log step (goes back to observed process)
# recovered_signals_exp = np.exp(signals_scaled)
