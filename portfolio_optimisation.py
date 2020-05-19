import numpy as np
from plot_utils import *


def compute_exp_mean(price_data, log_rets=True):
    # Computes the EWA mean of the percentage change in price
    num_days = price_data.shape[1]

    # Percentage change per day
    if log_rets:
        pc_day_change = np.log(price_data[:, 1:]) - np.log(price_data[:, :-1])
    else:
        pc_day_change = ((price_data[:, 1:] - price_data[:, :-1]) / price_data[:, :-1])

    # Exponential weights
    weights = np.geomspace(0.001, 1, num=num_days-1)#**0.4

    return np.sum(pc_day_change * (weights/np.sum(weights)), axis=1)


def compute_weights(var_mat, returns, target_return):
    # Computes the optimal portfolio weights to minimise variance/risk
    var_inverse = np.linalg.inv(var_mat)
    weights = np.dot(var_inverse, returns) / np.dot(np.dot(returns.T, var_inverse), returns)
    return weights * target_return


def comp_variance(weights, data, day):
    # Calculates variance of percentage returns

    covar = np.cov(data[:, day] * 100)

    variance = np.dot(weights, np.dot(weights, covar))

    return variance


def comp_return(weights, price_data, day_range=None, return_raw=False):
    num_days = np.shape(weights)[1]
    num_stocks = np.shape(weights)[0]

    if day_range is None:
        day_range = np.arange(num_days)

    returns = np.zeros(num_days)
    variance = np.zeros(num_days)

    data = np.log(price_data[:, 1:]) - np.log(price_data[:, :-1])

    for i, day in enumerate(day_range):
        returns[i] = np.dot(weights[:, i], np.log(price_data[:, day]) - np.log(price_data[:, day-1]))
        variance[i] = comp_variance(weights[:, i], data, day)

    if return_raw:
        return returns, np.cumsum(returns), variance
    else:
        return np.cumsum(returns), variance
