import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

"""
CODE TAKEN FROM https://github.com/akcarsten/Independent_Component_Analysis
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


def whiten(X):
    # Calculate the covariance matrix
    coVarM = covariance(X)

    # Single value decomposition
    U, S, V = np.linalg.svd(coVarM)

    # Calculate diagonal matrix of eigenvalues
    d = np.diag(1.0 / np.sqrt(S))

    # Calculate whitening matrix
    whiteM = np.dot(U, np.dot(d, U.T))

    # Project onto whitening matrix
    Xw = np.dot(whiteM, X)

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
    # TODO make it work for vectors
    if r > 0:
        return 1
    if r == 0:
        return 0
    else:
        return -1


def rhd(model, true):
    # Function to calculate the Relative Hamming Distance (RHD)
    num_points = len(model)


N = 255

stem = "Data Sets\\FTSEICA\\"

names = {"BARC Historical Data.csv": [1],
         "BP Historical Data.csv": [1],
         "FTSE 100 Futures Historical Data.csv": [1],
         "FTSE 100 Historical Data.csv": [1, 2, 3, 4, 5],
         "LLOY Historical Data.csv": [1],
         "TW Historical Data.csv": [1],
         "Glen Historical Data.csv": [1],
         "United Kingdom 3-Month Bond Yield Historical Data.csv": [1],
         "United Kingdom 1-Month Bond Yield Historical Data.csv": [1],
         "VOD Historical Data.csv": [1],
         }

paths = [stem + name for name in names.keys()]
num_columns = sum([len(cols) for cols in names.values()])
data = np.zeros([N, num_columns])

i = 0
for path in paths:
    columns = names[path[len(stem):]]
    data_in = pd.read_csv(path).values[:, columns]
    if data_in.shape[0] != N:
        # Data contains bank holidays etc. so remove
        mask = np.ones(data_in.shape[0], dtype=bool)
        if data_in.shape[0] == 263:
            mask[[49, 114, 129, 139, 140, 218, 222, 223]] = False
        elif data_in.shape[0] == 261:
            mask[[49, 114, 129, 139, 140, 222]] = False
        else:
            print("Unexpected length")
        data_in = data_in[mask]
    data[:, i:i + len(columns)] = data_in
    i += len(columns)

data_difference = (data[:-1, :] - data[1:, :]).T

X = np.flip(data_difference, 1)

# Center signals
Xc, meanX = centre(X)

# Whiten mixed signals
Xw, whiteM = whiten(Xc)


# Check if covariance of whitened matrix equals identity matrix
print(np.round(covariance(Xw)))

W = fastIca(Xw,  alpha=1)

# Un-mix signals using
unMixed = Xw.T.dot(W.T)

for i in range(num_columns):
    plt.subplot(num_columns, 1, i+1)
    plt.plot(unMixed[:, i])

plt.show()

# Re add mean
unMixed_scaled = (unMixed.T + meanX).T
