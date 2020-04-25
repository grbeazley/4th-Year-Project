import numpy as np
from utilities import comp_sign

"""
SOME FUNCTIONS: CODE TAKEN FROM https://github.com/akcarsten/Independent_Component_Analysis
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


def cf_tanh(w, alpha=1, derivative=False):
    # Computes the contrast function, g for g = tanh(u)
    if derivative:
        # Derivative required so return g prime
        return (1 - np.square(np.tanh(w))) * alpha
    else:
        # No derivative so return normal function
        return np.tanh(w*alpha)


def cf_cosh(w, alpha=1, derivative=False):
    # Computes the contrast function, for g = log(cosh(u))
    if derivative:
        # Derivative required so return g prime
        return np.tanh(alpha*w)
    else:
        # No derivative so return normal function
        return np.log(np.cosh(alpha*w)) / alpha


def cf_krts(w, derivative=False):
    return w


def fastICA(signals, alpha=1, thresh=1e-8, iterations=5000, contrast_func=cf_tanh, reduce_dims=0):
    m, n = signals.shape

    # Initialize random weights
    W = np.random.rand(m, m)

    for c in range(m):
        # Iterate through rows in W
        w = W[c, :].copy().reshape(m, 1)
        w = w / np.sqrt((w ** 2).sum())

        i = 0
        lim = 100
        while (lim > thresh) & (i < iterations):
            # Dot product of weight and signal
            ws = np.dot(w.T, signals)

            # Pass w*s into contrast function g
            wg = contrast_func(ws, alpha, derivative=False).T

            # Pass w*s into g prime
            wg_ = contrast_func(ws, alpha, derivative=True)

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


def energyICA(signals_in, thresh=1e-9, iterations=10000, tau=1, reduce_dims=0):
    # Performs fastICA using a time lag as proposed by Hyvarinen 2001

    m, n = signals_in.shape

    # Reduce dimension of output if required
    m_def = m - reduce_dims

    assert tau < n, "tau too large"

    signals_shifted = signals_in[:, :-tau]
    signals = signals_in[:, tau:]

    # Initialize random weights
    W = np.random.rand(m_def, m)
    M_all = np.zeros([n-tau, m, m])

    for i in range(n-tau):
        # Iterate over all shifted elements to create matrix M
        M_all[i, :, :] = np.outer(signals[:, i], signals_shifted[:, i])

    M_mean = M_all.mean(axis=0)

    M = M_mean + M_mean.T

    for c in range(m_def):
        # Iterate through rows in W
        w = W[c, :].copy().reshape(m, 1)

        # Normalise w
        w = w / np.sqrt((w ** 2).sum())

        i = 0
        lim = 100
        while (lim > thresh) & (i < iterations):
            # Dot product of weight and signal
            ws = np.dot(w.T, signals)

            # Dot product of weight and shifted signal
            ws_shifted = np.dot(w.T, signals_shifted)

            # Compute element-wise squared values
            ws_shifted_squared = np.square(ws_shifted)
            ws_squared = np.square(ws)

            # Create terms in equation (7)
            wNew_1 = (signals * ws * ws_shifted_squared).mean(axis=1)
            wNew_2 = (signals_shifted * ws_squared * ws_shifted).mean(axis=1)
            wNew_3 = -2*w - np.dot(M, w) * np.dot(np.dot(w.T, M), w)

            # Update weights and normalise
            wNew = wNew_1 + wNew_2 + wNew_3.squeeze()
            wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])

            w_norm = wNew / np.sqrt((wNew ** 2).sum())

            # Calculate limit condition
            lim = np.abs(np.abs((w_norm * w).sum()) - 1)

            # Update weights
            w = w_norm

            # Update counter
            i += 1

        W[c, :] = w.T

    return W


def rhd(model, true):
    # Function to calculate the Relative Hamming Distance (RHD)
    num_points = len(model)
    true_r = comp_sign(true[1:] - true[:-1])
    model_r = comp_sign(model[1:] - model[:-1])
    squared_diff = np.sum(np.square(true_r - model_r))
    return squared_diff / (num_points - 1)


def adj_rhd(model, true):
    # Function to calculate the adjusted Relative Hamming Distance (RHD)
    # That is, the sign of the value of the time series is used not the sign of the difference
    num_points = len(model)
    true_r = comp_sign(true)
    model_r = comp_sign(model)
    squared_diff = np.sum(np.square(true_r - model_r))
    return squared_diff / (num_points - 1)


def whiten_data(data):
    # Whitens input m x N time series data
    data_centred, _ = centre(data)
    data_whitened, whiten_matrix = whiten(data_centred)
    return data_whitened, whiten_matrix


def comp_ica(data, algorithm="fastICA", reduce_dims=0):
    # data is an m x N matrix where N is the number of data points and m is the number of series
    # Returns calculated independent components and the mixing matrix to recombine them
    # independent * mixing matrix = original signals

    # Select algorithm to compute components with
    if algorithm == "fastICA":
        algo_func = fastICA
    elif algorithm == "energyICA":
        algo_func = energyICA
    else:
        algo_func = fastICA

    unmix_matrix = algo_func(data, reduce_dims=reduce_dims)
    # mix_matrix = np.linalg.inv(unmix_matrix)
    latent_signals = np.dot(data.T, unmix_matrix.T).T
    return latent_signals, unmix_matrix


#################################################### DEPRECATED CODE ###################################################

# def vsobi(data, contrast_func, u_i, derivative=False):
#
#
#     return u_drvtv


def demixing_optimiser(data, approximation, contrast_func=cf_krts):
    # Computes the maximisation algorithm given in

    num_components, n = data.shape

    epsilon = 0.1
    change = epsilon + 1
    norm_demix_matrix_old = np.identity(num_components)

    # TODO refactor this variable
    t = np.zeros([num_components, num_components])

    while change > epsilon:
        for i in range(num_components):
            # TODO may need transposing
            # TODO refactor approximation
            t[i, :] = approximation(data, contrast_func, norm_demix_matrix_old[i, :])

        norm_demix_matrix_new = np.linalg.inv(np.matmul(t, t.T))

        # Calculate the size of the change on this iteration to test convergence
        change = np.linalg.norm(norm_demix_matrix_new - norm_demix_matrix_old)

        norm_demix_matrix_old = norm_demix_matrix_new

    return norm_demix_matrix_old

### RHD Iterative
    # IC_order = []
    # for k in range(num_columns):
    #     # Check all combinations, decreasing each time
    #     rhds = np.zeros(num_columns-k)
    #     mask = np.ones(num_columns, dtype=bool)
    #     mask[IC_order] = False
    #     invW_trunc = invW.T[:, mask]
    #     unMixed_trunc = unMixed[:, mask]
    #     # print(mask)
    #
    #     for i in range(num_columns-k):
    #         # Check RHD by dropping one of the remaining ICs in turn
    #         mask_iter = np.ones(num_columns-k, dtype=bool)
    #         mask_iter[i] = False
    #         invW_trunc_iter = invW_trunc[:, mask_iter]
    #         model = np.dot(invW_trunc_iter, unMixed_trunc[:, mask_iter].T)
    #         for j in range(num_columns):
    #             rhds[i] += rhd(model[j, :], Xw[j, :])
    #
    #     plt.figure(k+1)
    #     plt.plot(rhds)
    #     plt.xlabel('Component Index')
    #     plt.ylabel('Relative Hamming Distance')
    #     index = np.argmax(rhds)
    #     print(index)
    #     if isinstance(index, list):
    #         index = index[0]
    #     IC_order.append(index)