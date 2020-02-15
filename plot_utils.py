# File contains functions to assist in plotting data series of unusual forms
import numpy as np
from matplotlib import pyplot as plt
from utilities import log_returns


def plot(*args):
    # Wrapper for plt.plot
    plt.figure()
    plt.plot(*args)

def plot_sto_vol(time_series, conv_type=None):
    # Produces a series of plots of stochastic volatility data
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


def plot_components(series_data, title=''):
    # Creates figure with subplots for each of the m components in the m x N input data
    figure = plt.figure()
    figure.suptitle(title)
    num_points = len(series_data[0, :])
    num_series = len(series_data[:, 0])
    ymax = np.max(series_data)
    ymin = np.min(series_data)
    for i in range(num_series):
        if i == 0:
            plt.title(title)
        plt.subplot(num_series, 1, i + 1)
        plt.scatter(np.arange(num_points), series_data[i, :], s=0.5)
        plt.ylim([ymin, ymax])
        frame1 = plt.gca()
        if i != num_series - 1:
            frame1.axes.get_xaxis().set_ticks([])

    plt.show()
