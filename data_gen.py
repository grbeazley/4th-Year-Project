from utilities import load_data
import numpy as np
from stochastic_volatility import hidden_to_observed, gen_multi_sto_vol
from plot_utils import *
from portfolio_optimisation import *


def load_oil(plot_comps=True, return_raw=False):

    stem = "Data Sets\\Oil\\"

    names = {"BP.L.csv": ['Adj Close'],
             "CVX.csv": ['Adj Close'],
             "OGZPY.csv": ['Adj Close'],
             "PBR.csv": ['Adj Close'],
             # "PSX.csv": ['Adj Close'],
             "RDSA.L.csv": ['Adj Close'],
             "SLB.csv": ['Adj Close'],
             "TOT.csv": ['Adj Close'],
             "XOM.csv": ['Adj Close'],
             "Crude.csv": ['Adj Close'],
             }

    data_frame = load_data(stem, names)

    # Take only series values from the data frame
    data = data_frame.values[1:, :].astype('float')

    # data_pos = np.where(data <= 0, 0.05, data)
    data_pos = np.abs(data)

    # Take difference
    data_returns = np.log(data_pos[:, 1:]) - np.log(data_pos[:, :-1])

    # Calculate the number of time series
    num_series = len(data[:, 0])

    # Take the dates from the data frame for plotting
    dates = data_frame.values[0, 1:]

    if plot_comps:
        plot_components(data_returns, dates=dates, global_lims=[-0.2, 0.2])

    if return_raw:
        return data_returns, dates, data_pos
    else:
        return data_returns, dates


def load_port(plot_comps=True, return_raw=False):

    stem = "Data Sets\\Daily_portfolio\\"

    # names = {"GBPEUR=X.csv": ['Adj Close'],
    #          "GBPJPY=X.csv": ['Adj Close'],
    #          "GBPNZD=X.csv": ['Adj Close'],
    #          "GBPUSD=X.csv": ['Adj Close'],
    #          "AAPL.csv": ['Adj Close'],
    #          }

    names = {"Crude.csv": ['Adj Close'],
             "TOT.csv": ['Adj Close'],
             "CVX.csv": ['Adj Close'],
             # "Gold.csv": ['Adj Close'],
             "AAPL.csv": ['Adj Close'],
             "INTC.csv": ['Adj Close'],
             "AMD.csv": ['Adj Close'],
             #"W=F.csv": ['Adj Close'],
             }

    data_frame = load_data(stem, names)

    # Take only series values from the data frame
    data = data_frame.values[1:, :].astype('float')

    # data_pos = np.where(data <= 0, 0.05, data)
    data_pos = np.abs(data)

    # Take difference
    data_returns = np.log(data_pos[:, 1:]) - np.log(data_pos[:, :-1])

    # Take the dates from the data frame for plotting
    dates = data_frame.values[0, 1:]

    if plot_comps:
        plot_components(data_returns, dates=dates, global_lims=[-0.2, 0.2])

    if return_raw:
        return data_returns, dates, data_pos
    else:
        return data_returns, dates


def load_bivariate(num):
    # Generate truck and trailer series, using one driving process and different observations
    mu, a, b, c = 0, 0.95, 0.5, 0.1
    x_prev = np.sqrt(b / (1 - a**2)) * np.random.randn()

    trajectory_hidden = np.zeros(num + 1)

    # Create array of hidden state variables
    for i in range(num):
        x = mu + a * (x_prev - mu) + np.sqrt(b) * np.random.randn()
        trajectory_hidden[i] = x
        x_prev = x

    data_y = np.array([hidden_to_observed(trajectory_hidden, c),
                       hidden_to_observed(trajectory_hidden, c)])

    data_h = np.array([trajectory_hidden, trajectory_hidden])

    return data_h, data_y


def load_msv(num, num_series):
    # Generate pseudo random phi matrix around a prior
    # add_fac, mult_fac = scale_uni(0.7, 0.9)
    # diag_val_phi = (np.random.rand(num_series) + add_fac) / mult_fac
    # phi = np.diag(diag_val_phi)
    # phi = phi + (np.random.rand(num_series, num_series) - 0.5) * 4 * ((1-np.max(diag_val_phi))/num_series)

    # phi = np.array([[1, 0, 0, 0],
    #                 [1, 0, 0, 0],
    #                 [0, 0, 1, 0],
    #                 [0, 0, 1, 0]], dtype=float)

    phi = np.array([[0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                    [0.1, 0.1, 0.1, 0.7]], dtype=float)

    phi *= 0.95

    # Generate pseudo random sigma eta matrix around a prior
    # add_fac, mult_fac = scale_uni(0.3, 0.7)
    # diag_val_eta = (np.random.rand(num_series) + add_fac) / mult_fac
    # sigma_eta = np.diag(diag_val_eta)
    # low_tri = np.tril(np.random.randn(num_series, num_series) * (np.max(abs(diag_val_eta)))/num_series)
    # sigma_eta = sigma_eta + low_tri + low_tri.T - 2*np.diag(np.diag(low_tri))

    # sigma_eta = np.eye(num_series) * np.sqrt(0.5)

    data_h, data_y = gen_multi_sto_vol(num,
                                       num_series,
                                       phi=phi,
                                       var_latent=0.5,
                                       var_observed=0.1,
                                       return_hidden=True)

    return data_h, data_y


if __name__ == "__main__":
    # load_forex(True)
    # data, dates, price_data = load_oil(plot_comps=True, return_raw=True)
    data, dates, price_data = load_port(plot_comps=False, return_raw=True)

    days = np.arange(4616, 5021)
    weights = np.zeros([6, len(days)])
    for i, day in enumerate(days):
        means = compute_exp_mean(price_data[:, day-10:day-1]) * 100
        covar = np.cov(data[:, day-10:day-1]*100)
        weights[:, i] = compute_weights(covar, means, 0.25)

    returns_1, variance_1 = comp_return(weights, price_data, days)

    weights_2 = np.ones_like(weights) / 6
    returns_2, variance_2 = comp_return(weights_2, price_data, days)

    plot(np.exp(returns_1))
    plt.plot(np.exp(returns_2))

    # plot(variance_1)
    # plt.plot(variance_2)

    # plot(np.exp(returns_1)/np.sqrt(variance_1))
    # plt.plot(np.exp(returns_2)/np.sqrt(variance_2))





