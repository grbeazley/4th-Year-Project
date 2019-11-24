from __future__ import absolute_import, division, print_function, unicode_literals
#import tensorflow as tf
#import matplotlib as mpl
#import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
FIRST 5 FUNCTIONS: CODE TAKEN FROM https://github.com/akcarsten/Independent_Component_Analysis
"""


def centre(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = x - mean
    return centered, mean


def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T))/n


def whiten(x):
    # Calculate the covariance matrix
    coVarM = covariance(x)

    # Single value decomposition
    U, S, V = np.linalg.svd(coVarM)

    # Calculate diagonal matrix of eigenvalues
    d = np.diag(1.0 / np.sqrt(S))

    # Calculate whitening matrix
    whiteM = np.dot(U, np.dot(d, U.T))

    # Project onto whitening matrix
    Xw = np.dot(whiteM, x)

    return Xw, whiteM


def fastIca(signals, alpha=1, thresh=1e-8, iterations=5000):
    m, n = signals.shape

    # Initialize random weights
    W = np.random.rand(m, m)

    for c in range(m):
        w = W[c, :].copy().reshape(m, 1)
        w = w / np.sqrt((w ** 2).sum())

        i = 0
        lim = 100
        while (lim > thresh) & (i < iterations):
            # Dot product of weight and signal
            ws = np.dot(w.T, signals)

            # Pass w*s into contrast function g
            wg = np.tanh(ws * alpha).T

            # Pass w*s into g prime
            wg_ = (1 - np.square(np.tanh(ws))) * alpha

            # Update weights
            wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

            # Decorrelate weights
            wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
            wNew = wNew / np.sqrt((wNew ** 2).sum())

            # Calculate limit condition
            lim = np.abs(np.abs((wNew * w).sum()) - 1)

            # Update weights
            w = wNew

            # Update counter
            i += 1

        W[c, :] = w.T
    return W


def sign(r):
    # Computes the sign of r
    if isinstance(r, (list, np.ndarray)):
        # Assume an array
        neg_indexes = np.where(r < 0)
        pos_indexes = np.where(r > 0)
        r[neg_indexes] = -1
        r[pos_indexes] = 1
        return r
    else:
        # Assume it is a single value
        if r > 0:
            return 1
        if r == 0:
            return 0
        else:
            return -1


def rhd(model, true):
    # Function to calculate the Relative Hamming Distance (RHD)
    num_points = len(model)
    true_r = sign(true[1:] - true[:-1])
    model_r = sign(model[1:] - model[:-1])
    squared_diff = np.sum(np.square(true_r - model_r))
    return squared_diff / (num_points - 1)


def whiten_data(data):
    # Whitens input m x N time series data
    data_centred, _ = centre(data)
    data_whitened, _ = whiten(data_centred)
    return data_whitened


def comp_ica(data):
    # data is an m x N matrix where N is the number of data points and m is the number of series
    # Returns calculated independent components and the mixing matrix to recombine them
    # independent * mixing matrix = original signals
    unmix_matrix = fastIca(data, alpha=1)
    mix_matrix = np.linalg.inv(unmix_matrix)
    latent_signals = np.dot(data.T, unmix_matrix.T)
    return latent_signals, mix_matrix
