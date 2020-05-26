import numpy as np
from ica import whiten_data, comp_ica, rhd
from plot_utils import *
from stochastic_volatility import *
from utilities import normalise, is_normal
from particle_filter_gamma import ParticleFilterGamma
from pdfs import comp_k_theta_from_alphas
from data_gen import *
from particle_filter_multi import ParticleFilterMulti

# Set the random seed for reproducibility
np.random.seed(0)

######################## Import #################################
# Create 2400 training and 100 test
train = 4616
test = 405
num = train + test
num_series = 6


# data_h, data_y = load_bivariate(num)
# data_h, data_y = load_msv(num, num_series)
# data_y, dates = load_oil(plot_comps=False)
data_y, dates, price_data = load_port(plot_comps=False, return_raw=True)
data_h = np.ones_like(data_y)

labels = ["Oil Price", "Total S.A.", "Chevron", "Apple", "Intel", "AMD"]

#################### Pre-processing #######################
plot_components(data_y, 'Input Data Raw', global_lims=[-0.2, 0.2], labels=labels)
plot_components(data_y, 'Input Data Raw')
# plot_components(data_h, 'Input Hidden State')

data_y_norm, y_means, y_stds = normalise(data_y, return_params=True)

data_train = data_y_norm[:, :train]
data_test = data_y_norm[:, train:]

true_train = data_h[:, :train]
true_test = data_h[:, train:]

######################### ICA ###########################
input("Press Enter to run ICA...")
data_abs = np.abs(data_train)

# Take the logs of the absolute values
data_hidden = np.log(data_abs)

plot_components(data_hidden, 'Input Hidden')

# Centre the data and store the parameters
means = np.mean(data_hidden, axis=1, keepdims=True)
# means = np.zeros([num_series, 1])
data_hidden_av_norm = data_hidden - means

plot_components(data_hidden_av_norm)

# Compute whitened data
data_whitened, whiten_matrix = whiten_data(data_hidden_av_norm)
# data_whitened, whiten_matrix = data_hidden_av_norm, np.eye(num_series)
whiten_inv = np.linalg.inv(whiten_matrix)

plot_components(data_whitened, 'Input Data Whitened')

# Test Gaussianity of data
kurtosis_values = is_normal(data_whitened)

# Number of dimensions to find
num_ics = 6

# Compute independent components
ics, unmix_matrix = comp_ica(data_whitened, algorithm="energyICA", reduce_dims=num_series-num_ics, tau=1)

# ics = np.dot(unmix_matrix, data_hidden_av_norm)

mix_matrix = np.linalg.pinv(unmix_matrix)

demix_matrix = np.dot(unmix_matrix, whiten_matrix)

plot_components(ics, 'Independent Components')


rhds = np.zeros(num_ics)
mse_mat = np.zeros([num_series, num_ics])
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
    model_scaled = model_correlated + means

    # Undo the log step (goes back to observed process)
    model_recovered = np.exp(model_scaled)

    for j in range(num_series):
        rhds[i] += rhd(model_recovered[j, :], data_abs[j, :])

    # plot_compare(model_recovered, data_abs)
    mse[i] = np.mean(np.square(data_abs - model_recovered))
    mse_mat[:, i] = np.mean(np.square(data_abs - model_recovered), axis=1)

print(mse)
# bar(mse)
stack_bar(mse_mat/num_series, labels=labels)
# stack_bar(mse_mat/num_series)
plt.pause(0.01)


############### Particle Filter Paramater Optimisation ################
input("Press Enter to run particle filter...")
N = 5000
np.random.seed(10)

signal_idxs = [0]
noise_idxs = [1,2,3,4,5]
filters = {}

num_predict = test
ics_prds = np.zeros([num_ics, num_predict])

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
    true_hidden = np.dot(alphas, true_train) / 2
    # true_hidden = true_train[0, :]
    scale = 0.1

    hidden = sgn*ics[ica_idx, :]
    # hidden = (sgn*ics[ica_idx, :]) - np.mean(sgn*ics[ica_idx, :])

    filters[ica_idx] = ParticleFilterGamma(true_obs=np.exp(hidden)*scale,
                                           num_particles=N,
                                           a=0.891,
                                           b=0.191,
                                           c=0.616,
                                           correction=np.sum(alphas),
                                           k=kk,
                                           theta=thth,
                                           learn_rate=0.1/train,
                                           num_iterations=10,
                                           true_hidden=true_hidden,
                                           learn_a=0.5,
                                           learn_b=10,
                                           learn_c=10,
                                           multi=True)

    # filters[ica_idx].plot_filter_pass()
    filters[ica_idx].calibrate_model()
    filters[ica_idx].filter_pass()
    filters[ica_idx].plot_params(ica_idx)
    filters[ica_idx].plot_likelihood()

    # filters[ica_idx].adap_learn_rate = 0.01/train
    # filters[ica_idx].calibrate_model(50)
    # filters[ica_idx].plot_params(ica_idx)

    input("Press Enter to perform signal prediction...")
    test_hidden = np.log(np.abs(data_test))
    test_hidden_norm = test_hidden - means
    ics_test = np.dot(demix_matrix, test_hidden_norm)
    prds = filters[ica_idx].one_step_multi(np.exp(sgn*(ics_test[ica_idx, :num_predict]))*scale)
    ics_prds[ica_idx, :] = np.mean(prds, axis=0) - np.log(scale)

############################## Prediction ######################

input("Press Enter to perform noise prediction...")
predicted_components = {}

phi = np.array([[0.82, 0.1, 0.01, 0.05, 0.04, 0.05],
                [0.18, 0.53, 0.02, -0.01, 0.05, 0.02],
                [0.25, 0.18, 0.22, 0.15, -0.01, -0.01],
                [0.16, 0.0, -0.02, 0.65, 0.17, 0.05],
                [0.17, 0.03, -0.02, 0.11, 0.63, 0.08],
                [0.16, 0.0, -0.01, 0.03, 0.12, 0.61]], dtype=float)

eta = np.array([0.218, 0.271, 0.236, 0.364, 0.433, 0.452])
beta = np.array([1.35, 1.48, 1.55, 2.65, 2.46, 2.15])

pf = ParticleFilterMulti(data_train,
                         num_particles=6000,
                         p=num_series,
                         a=0.8,
                         b=0.6,
                         c=2,
                         # true_hidden=data_h,
                         num_iterations=10,
                         learn_rate=0.1 / train,
                         learn_phi=0.5,
                         learn_eta=1,
                         learn_beta=1,
                         phi=phi,
                         eta=eta,
                         beta=beta,
                         )

pf.filter_pass()
prediction_multi_pf = pf.one_step_multi(data_test)
prdct_multi_mean = np.mean(prediction_multi_pf, axis=1)
prediction_multi_obs = np.exp(prdct_multi_mean/2) * np.sqrt(beta)[:, np.newaxis] * np.random.randn(6, 405)
multi_predict_vols = (prediction_multi_obs * y_stds) + y_means


for ica_idx in noise_idxs:
    # Predict a noise process based on the ICA weights
    noise_alphas = demix_matrix[ica_idx, :]
    samples = np.random.randn(num_series, test)

    # Assume noise can be approximated by a transformed Gaussian
    prediction = np.dot(noise_alphas, np.log(np.abs(samples)))
    prediction_norm = prediction - np.mean(prediction)


    ics_prds[ica_idx, :] = prediction_norm


hdn_prds_norm = np.dot(mix_matrix, ics_prds)
hdn_prds = hdn_prds_norm + means
plot_components(hdn_prds, "predicted hidden", global_lims=[-10, 4])
# plot_components(test_hidden, "True hidden")
vol_prds = (np.exp(hdn_prds) * y_stds) + y_means


input("Press Enter to perform portfolio optimisation...")

days = np.arange(train, train+test)
weights = np.zeros([num_series, len(days)])
weights_naive = np.zeros([num_series, len(days)])
weights_same = np.zeros([num_series, len(days)])
weights_msv = np.zeros([num_series, len(days)])

dy_b = 8
dy_f = 0
targ_ret = 0.1

for i, day in enumerate(days):
    ret_means = compute_exp_mean(price_data[:, day-100:day]) * 100

    # Predicting same variance as yesterday
    covar_same = np.cov(np.hstack((data_y[:, day-dy_b+1:day-dy_f], data_y[:, day-dy_f-1, None]))*100)
    weights_same[:, i] = compute_weights(covar_same, ret_means, targ_ret)

    # Predicting without
    covar_naive = np.cov(data_y[:, day-dy_b:day-dy_f] * 100)
    weights_naive[:, i] = compute_weights(covar_naive, ret_means, targ_ret)

    # Predicting with MSV prediction
    covar_msv = np.cov(np.hstack((data_y[:, day-dy_b+1:day-dy_f], multi_predict_vols[:, i, None]))*100)
    weights_msv[:, i] = compute_weights(covar_msv, ret_means, targ_ret)

    # Using model prediction
    # signs = comp_sign(data_y[:, day-dy_f])
    signs = np.ones(num_series)
    covar = np.cov(np.hstack((data_y[:, day-dy_b+1:day-dy_f], signs * vol_prds[:, i, None]))*100)
    weights[:, i] = compute_weights(covar, ret_means, targ_ret)

returns_1_raw, returns_1, variance_1 = comp_return(weights, price_data, days, True)
returns_naive_raw, returns_naive, variance_naive = comp_return(weights_naive, price_data, days, True)
returns_same_raw, returns_same, variance_same = comp_return(weights_same, price_data, days, True)
returns_msv_raw, returns_msv, variance_msv = comp_return(weights_msv, price_data, days, True)

# Buy and hold 1/N proportion
weights_buy_hold = np.ones_like(weights) / 6
returns_bh_raw, returns_buy_hold, variance_buy_hold = comp_return(weights_buy_hold, price_data, days, True)

plot(dates[days], np.exp(returns_1))
plt.plot(dates[days], np.exp(returns_msv))
plt.plot(dates[days], np.exp(returns_naive))
# plt.plot(dates[days], np.exp(returns_same))
plt.plot(dates[days], np.exp(returns_buy_hold))

plt.legend(['Simplified Prediction', 'Multivariate Prediction', 'Naïve', "Buy & Hold"])

print(np.mean(variance_1), np.mean(variance_naive), np.mean(variance_buy_hold))
# print(np.mean(variance_1), np.mean(variance_same), np.mean(variance_buy_hold))
print(np.mean(returns_1_raw*100/np.sqrt(variance_1)),np.mean(returns_naive_raw*100/np.sqrt(variance_naive)))
# print(np.mean(returns_1_raw*100/np.sqrt(variance_1)),np.mean(returns_same_raw*100/np.sqrt(variance_same)))
