import numpy as np
from scipy.special import gamma as gamma_function


def normal_pdf(x, mu, sigma_sqrd):
    # Returns a normal pdf
    norm_const = np.sqrt(2 * np.pi * sigma_sqrd)
    return np.exp(-(x-mu)**2 / 2 * sigma_sqrd) / norm_const


def log_abs_norm_pdf(x, mu=0, sigma_sqrd=1):
    norm_const = np.sqrt(2*np.pi*sigma_sqrd)
    exponent = - np.square(np.exp(x) - mu)/2*sigma_sqrd
    return np.exp(exponent + x) / norm_const


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

    scale_factor = x**(-1 + 1/alpha) / alpha
    return folded_norm_pdf(x**(1/alpha), mu, sigma_sqrd) * scale_factor


def gamma_pdf(x, alpha=1, beta=1):
    # Simple wrapper for the scipy gamma function
    top = beta**alpha * x**(alpha - 1) * np.exp(-beta * x)
    return top / gamma_function(alpha)
