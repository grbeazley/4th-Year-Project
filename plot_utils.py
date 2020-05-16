# File contains functions to assist in plotting data series of unusual forms
import numpy as np
from scipy.stats import norm
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


def bar(*args, **kwargs):
    # Wrapper for the plt.bar function
    plt.figure()

    if len(args) == 1:
        # Only data passed
        # plt.bar(str(np.arange(len(args[0]))), *args, width=0.4, **kwargs)
        plt.bar([str(i+1) for i in range(len(args[0]))], *args, width=0.4, **kwargs)
    else:
        plt.bar(*args, s=0.5, **kwargs)


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


def rolling_confidence(data, num_steps=10):
    # Data is an N x M matrix of trajectories
    M = data.shape[1]
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    plt.figure()
    plt.plot(means, linewidth=2, label='Prediction')

    rgb_start = [20, 118, 189]
    rgb_end = [173, 205, 240]
    colors = col_gradient(rgb_start, rgb_end, num_steps)[::-1] / 256
    std_steps = norm.ppf(np.linspace(0.6, 0.95, num_steps)[::-1])

    for i in range(num_steps):
        plt.fill_between(np.arange(M), means - stds * std_steps[i], means + stds * std_steps[i],
                         color=colors[i], alpha=.1)


def col_gradient(start_rgb, end_rgb, num_points):
    # Computes colour gradient, end value not included
    gradient = np.array(end_rgb) - np.array(start_rgb)
    color_gradient = np.zeros([num_points, 3])

    for i in range(num_points):
        color_gradient[i, :] = np.round(np.array(start_rgb) + (i/num_points) * gradient)

    return color_gradient


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


def plot_components(series_data, title='', line=False, global_lims=None, dates=None):
    # Creates figure with subplots for each of the m components in the m x N input data
    figure = plt.figure()
    figure.suptitle(title)
    num_points = len(series_data[0, :])
    num_series = len(series_data[:, 0])

    set_lims = False

    if dates is None:
        isdates = False
    else:
        isdates = True

    if global_lims is None:
        global_lims = True
    elif (type(global_lims) == np.ndarray) or (type(global_lims) == list):
        ymin = global_lims[0]
        ymax = global_lims[1]
        global_lims = False
        set_lims = True
    else:
        ymax = np.max(series_data)
        ymin = np.min(series_data)
        set_lims = True

    if global_lims:
        ymax = np.max(series_data)
        ymin = np.min(series_data)
        set_lims = True

    if line:
        for i in range(num_series):
            if i == 0:
                plt.title(title)
            plt.subplot(num_series, 1, i + 1)
            plt.plot(series_data[i, :])
            if set_lims:
                plt.ylim([ymin, ymax])
            frame1 = plt.gca()
            if i != num_series - 1:
                frame1.axes.get_xaxis().set_ticks([])
    else:
        for i in range(num_series):
            if i == 0:
                plt.title(title)
            plt.subplot(num_series, 1, i + 1)
            if isdates:
                plt.scatter(dates, series_data[i, :], s=0.5)
            else:
                plt.scatter(np.arange(num_points), series_data[i, :], s=0.5)
            if set_lims:
                plt.ylim([ymin, ymax])
            frame1 = plt.gca()
            if i != num_series - 1:
                frame1.axes.get_xaxis().set_ticks([])
    plt.draw()


def plot_inset(*args, ax_lims=None, inset_pos=None, line_pos=None):
    fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
    ax.plot(*args)
    if inset_pos is None:
        inset_pos = [0.2, 0.4, 0.5, 0.5]

    if ax_lims is None:
        ax_lims = [0, 1, 0, 1]

    if line_pos is None:
        line_pos = [2,3]

    ax2 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax, inset_pos)
    ax2.set_axes_locator(ip)
    ax2.plot(*args)
    x1, x2, y1, y2 = ax_lims
    ax2.set_xlim(x1, x2)  # apply the x-limits
    ax2.set_ylim(y1, y2)  # apply the y-limits

    mark_inset(ax, ax2, loc1=line_pos[0], loc2=line_pos[1], fc="none", ec='0.5')
