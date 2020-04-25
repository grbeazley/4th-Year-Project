import numpy as np
from scipy.special import gamma as gamma_function
from scipy.stats import levy_stable
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset


def normal_pdf(x, mu, sigma_sqrd):
    # Returns a normal pdf
    norm_const = np.sqrt(2 * np.pi * sigma_sqrd)
    return np.exp(-(x-mu)**2 / 2 * sigma_sqrd) / norm_const


def log_abs_norm_pdf(x, mu=0, sigma_sqrd=1):
    norm_const = np.sqrt(2*np.pi*sigma_sqrd)
    exponent = - np.square(np.exp(x) - mu)/(2*sigma_sqrd)
    return 2 * np.exp(exponent + x) / norm_const


def folded_norm_pdf(x, mu=0, sigma_sqrd=1):
    # Evaluates |X| if X is normally distributed
    norm_const = np.sqrt(2*np.pi*sigma_sqrd)

    if (type(x) == np.ndarray) or (type(x) == list):
        # Have been passed a list or array of values
        out = np.zeros(len(x))
        if min(x) < 0:
            # Some values may be out of the range of support so compute iteratively
            for idx, value in enumerate(x):
                out[idx] = _folded_norm_individual(value, mu, sigma_sqrd)
            return out
        else:
            if mu != 0:
                # If mu not 0 use adjusted formula
                return (np.exp(-(x + mu) ** 2 / 2 * sigma_sqrd) + np.exp(-(x - mu) ** 2 / 2 * sigma_sqrd)) / norm_const
            else:
                return 2 * np.exp(-x ** 2 / 2 * sigma_sqrd) / norm_const
    else:
        # Passed a single value
        return _folded_norm_individual(x, mu, sigma_sqrd)


def _folded_norm_individual(x, mu=0, sigma_sqrd=1):
    # Helper function for folded norm, only works with single values
    norm_const = np.sqrt(2 * np.pi * sigma_sqrd)
    if x < 0:
        return 0
    if mu != 0:
        # If mu not 0 use adjusted formula
        return (np.exp(-(x + mu) ** 2 / 2 * sigma_sqrd) + np.exp(-(x - mu) ** 2 / 2 * sigma_sqrd)) / norm_const
    else:
        return 2 * np.exp(-x ** 2 / 2 * sigma_sqrd) / norm_const


def power_folded_norm_pdf(x, alpha=1, mu=0, sigma_sqrd=1):
    # Computes the pdf for the folded normal raised to a power
    if (type(x) == np.ndarray) or (type(x) == list):
        # Have been passed a list or array of values
        if min(x) < 0:
            raise Exception("Support not fully positive")
    elif x < 0:
        raise Exception("Support not fully positive")

    scale_factor = x**(-1 + 1/alpha) / np.abs(alpha)
    return folded_norm_pdf(x**(1/alpha), mu, sigma_sqrd) * scale_factor


def _gamma_pdf(x, alpha=1.0, beta=1.0):
    # Simple wrapper for the scipy gamma function
    top = beta**alpha * x**(alpha - 1) * np.exp(-beta * x)
    return top / gamma_function(alpha)


def gamma_pdf(x, k=1, theta=1):
    # Wrapper to re-parameterize gamma
    return _gamma_pdf(x, alpha=k, beta=1/theta)


def mean_power_folded_norm(a, sigma=1):
    # Calculates the mean of a power folded normal distribution
    return (2**((a+1)/2) * sigma**a * gamma_function((a+1)/2)) / np.sqrt(2*np.pi)


def variance_power_folded_norm(a, sigma=1):
    # Calculates the variance of a power folded normal distribution
    ex_2 = _variance_power_folded_norm(a, sigma)
    return ex_2 - mean_power_folded_norm(a, sigma)**2


def _variance_power_folded_norm(a, sigma=1):
    # Calculates E[X^2] where X ~ power normal distribution (a)
    return (2**(a+0.5) * sigma**(2*a) * gamma_function(a + 0.5)) / np.sqrt(2*np.pi)


def xlnx_pwr_fldd_norm(x, a, sigma=1):
    # Returns the value of
    return x*np.log(x)*power_folded_norm_pdf(x, alpha=a, sigma_sqrd=sigma)


def expectation_power_folded_norm(a):
    # Computes E[XlogX] where X ~ power normal distribution (a)
    if np.abs(a) < 0.02:
        # a too small and would cause numerical errors
        # Return -a as an approximation
        return -0.661 * a
    return integrate.quad(xlnx_pwr_fldd_norm, 0, np.inf, args=(a,))[0]


def comp_k_theta_from_alphas(alphas):
    # Computes closed form MLE for k, theta
    # Uses numerically evaluated expectations in place of sums over the data
    num = len(alphas)
    means = np.zeros(num)
    e_xlogxs = np.zeros(num)

    # Constant value for the mean of a log normal ~ log(|N(0,1)|) distribution, computed numerically
    mean_log_abs_norm = -0.6351814

    for i, a in enumerate(alphas):
        means[i] = mean_power_folded_norm(a)

        # Track the E[XlogX] for each alpha value
        e_xlogxs[i] = expectation_power_folded_norm(a)

    # The mean of the product is the product of the means
    mean = np.prod(means)

    # Set initial value of the sum for E[XlogX]
    e_xlogx = 0

    # Compute E[XlogX] for the whole distribution
    for i in range(num):
        e_xlogx += mean * e_xlogxs[i] / means[i]

    # E[logX] for whole distribution
    e_logx = np.sum(alphas) * mean_log_abs_norm

    # Combine terms in form of MLEs
    k = mean / (e_xlogx - e_logx*mean)
    theta = e_xlogx - e_logx*mean

    return k, theta


def alpha_stable_pdf(x, alpha=1, beta=1, c=1, mu=0):
    # Returns probability from a pdf
    return levy_stable.pdf((x - mu)/c, alpha, beta)


def qq_plot(a_samples, b_samples):
    # Makes a QQ plot of two sets of random samples
    percentages = np.linspace(0, 100, 100)
    quants_a = np.percentile(a_samples, percentages)
    quants_b = np.percentile(b_samples, percentages)
    plt.figure()
    plt.plot(quants_a, quants_b, ls="", marker="o")

    x = np.linspace(min(np.min(quants_a), np.min(quants_b)), max(np.max(quants_a), np.max(quants_b)))
    plt.plot(x, x, color="k", ls="--")

    plt.show()


def qq_plot_inset(a_samples, b_samples, ax_lims, inset_pos=None):
    # Plots qq_plot with inset zoomed in on the ax_lims section
    if inset_pos is None:
        inset_pos = [0.1, 0.3, 0.65, 0.65]

    percentages = np.linspace(0, 100, 100)
    quants_a = np.percentile(a_samples, percentages)
    quants_b = np.percentile(b_samples, percentages)

    fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
    ax.plot(quants_a, quants_b, ls="", marker='o')
    x = np.linspace(min(np.min(quants_a), np.min(quants_b)), max(np.max(quants_a), np.max(quants_b)))
    ax.plot(x, x, color="k", ls="--")

    ax2 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax, inset_pos)
    ax2.set_axes_locator(ip)
    ax2.plot(quants_a, quants_b, ls="", marker='o')
    ax2.plot(x, x, color="k", ls="--")
    x1, x2, y1, y2 = ax_lims
    ax2.set_xlim(x1, x2)  # apply the x-limits
    ax2.set_ylim(y1, y2)  # apply the y-limits

    mark_inset(ax, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

