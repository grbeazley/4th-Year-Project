# File contains functions to assist in plotting data series of unusual forms
import numpy as np
from matplotlib import pyplot as plt
from utilities import log_returns
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset


def plot(*args):
    # Wrapper for plt.plot
    plt.figure()
    plt.plot(*args)


def scatter(*args):
    # Wrapper for the plt.scatter function
    plt.figure()
    if len(args) == 1:
        # Only data passed
        plt.scatter(np.arange(len(args[0])), *args, s=0.5)
    else:
        plt.scatter(*args, s=0.5)


def hist(*args):
    # Wrapper for the plt.hist function
    plt.figure()
    if len(args) == 1:
        # Only data passed
        plt.hist(*args, bins=100)
    else:
        plt.hist(*args)


def hist_norm(*args, **kwargs):
    # Wrapper for the plt.hist function
    plt.figure()
    if (len(args) + len(kwargs)) == 1:
        # Only data passed
        plt.hist(*args, **kwargs, bins=100, density=True)
    else:
        plt.hist(*args, **kwargs, density=True)


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


def plot_components(series_data, title='', line=False):
    # Creates figure with subplots for each of the m components in the m x N input data
    figure = plt.figure()
    figure.suptitle(title)
    num_points = len(series_data[0, :])
    num_series = len(series_data[:, 0])
    ymax = np.max(series_data)
    ymin = np.min(series_data)

    if line:
        for i in range(num_series):
            if i == 0:
                plt.title(title)
            plt.subplot(num_series, 1, i + 1)
            plt.plot(series_data[i, :])
            plt.ylim([ymin, ymax])
            frame1 = plt.gca()
            if i != num_series - 1:
                frame1.axes.get_xaxis().set_ticks([])
    else:
        for i in range(num_series):
            if i == 0:
                plt.title(title)
            plt.subplot(num_series, 1, i + 1)
            plt.scatter(np.arange(num_points), series_data[i, :], s=0.5)
            plt.ylim([ymin, ymax])
            frame1 = plt.gca()
            if i != num_series - 1:
                frame1.axes.get_xaxis().set_ticks([])

    plt.draw()


def plot_inset(*args, ax_lims=None, inset_pos=None):
    fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
    ax.plot(*args)
    if inset_pos is None:
        inset_pos = [0.2, 0.4, 0.5, 0.5]

    if ax_lims is None:
        ax_lims = [0, 1, 0, 1]

    ax2 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax, inset_pos)
    ax2.set_axes_locator(ip)
    ax2.plot(*args)
    x1, x2, y1, y2 = ax_lims
    ax2.set_xlim(x1, x2)  # apply the x-limits
    ax2.set_ylim(y1, y2)  # apply the y-limits

    mark_inset(ax, ax2, loc1=2, loc2=3, fc="none", ec='0.5')