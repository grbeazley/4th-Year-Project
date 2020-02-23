import numpy as np
from ica import whiten_data, comp_ica, rhd
from plot_utils import plot_components, plot_compare, plot
from stochastic_volatility import gen_multi_sto_vol, gen_univ_sto_vol
from utilities import normalise, is_normal, scale_uni, moving_average
from particle_filter_gradient import ParticleFilter

np.random.seed(1)

# Create 2400 training and 100 test
train = 2400
test = 100
num = train + test
num_series = 2

# Generate pseudo random phi matrix around a prior
# add_fac, mult_fac = scale_uni(0.85, 0.95)
# diag_val_phi = (np.random.rand(num_series) + add_fac) / mult_fac
# phi = np.diag(diag_val_phi)
# phi = phi + np.random.randn(num_series, num_series) * (1-np.max(diag_val_phi))/num_series

phi = np.array([[0.95, 0],
                [0.95, 0]])

# Generate pseudo random sigma eta matrix around a prior
# add_fac, mult_fac = scale_uni(0.3, 0.7)
# diag_val_eta = (np.random.rand(num_series) + add_fac) / mult_fac
# sigma_eta = np.diag(diag_val_eta)
# low_tri = np.tril(np.random.randn(num_series, num_series) * (np.max(abs(diag_val_eta)))/num_series)
# sigma_eta = sigma_eta + low_tri + low_tri.T - 2*np.diag(np.diag(low_tri))

sigma_eta = np.eye(2) #* np.sqrt(0.6)

data_h, data_y = gen_multi_sto_vol(num,
                                   num_series,
                                   phi=phi,
                                   var_latent=sigma_eta,
                                   var_observed=1,
                                   return_hidden=True)

plot_components(data_y, 'Input Data Raw')
plot_components(data_h, 'Input Hidden State')

data_train = data_y[:, :train]
data_test = data_y[:, train:]

######################### BIVARIATE ICA ###########################
input("Press Enter to run ICA...")
data_abs = np.abs(data_train)

# Take the logs of the absolute values
data_hidden = np.log(data_abs)

# Normalise the data and store the parameters
data_hidden_av_norm, mean, stds = normalise(data_hidden, return_params=True)

# Compute whitened data
data_whitened, whiten_matrix = whiten_data(data_hidden_av_norm)
whiten_inv = np.linalg.inv(whiten_matrix)

plot_components(data_whitened, 'Input Data Whitened')

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

    # Undo the log step (goes back to observed process)
    model_recovered = np.exp(model_scaled)

    for j in range(num_series):
        rhds[i] += rhd(model_recovered[j, :], data_abs[j, :])

    plot_compare(model_recovered, data_abs)
    mse[i] = np.mean(np.square(data_abs - model_recovered))

plot(rhds)

############### Particle Filter Paramater Optimisation ################
input("Press Enter to run particle filter...")
N = 100

# Un Whiten the result of the de-mixing
model_correlated = np.dot(whiten_inv, ics)

# Undo the normalisation
model_scaled = (model_correlated * stds) + mean

# Undo the log step (goes back to observed process)
model_recovered = np.exp(model_scaled)

plot_components(model_recovered)

particle_filter_1 = ParticleFilter(model_recovered[0, :],
                                   num_particles=N,
                                   a=0.7,
                                   b=0.9,
                                   c=0.9,
                                   num_iterations=150)

# particle_filter_1.filter_pass()
# particle_filter_1.plot_filter_pass()
particle_filter_1.calibrate_model()
particle_filter_1.plot_params()
# particle_filter_1.plot_filter_pass()

############################## Prediction ####################
aa = 0.95
bb = 0.8
cc = 0.95

data_test_prediction = gen_univ_sto_vol(100, a=aa, b=bb, c=cc, x0=model_scaled[0, -1])

ic_complete = np.zeros(num)
ic_complete[:train] = model_recovered[0, :]
ic_complete[train:] = np.abs(data_test_prediction[1:])

ic_complete_log = np.log(ic_complete)
ic_complete_log_norm = (ic_complete_log - np.mean(ic_complete)) / np.std(ic_complete_log)

mask = [True, False]
invW_trunc = mix_matrix[:, mask]

recovered_signals = np.outer(invW_trunc, ic_complete_log_norm)

signals_correlated = np.dot(whiten_inv, recovered_signals)

# model_scaled_complete = (model_correlated * stds) + mean

# Undo the log step (goes back to observed process)
recovered_signals_exp = np.exp(signals_correlated)
