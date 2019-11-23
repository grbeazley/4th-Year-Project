from __future__ import absolute_import, division, print_function, unicode_literals
#import tensorflow as tf
#import matplotlib as mpl
#import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

N = 255
np.random.seed(0)

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
         # "BCS.csv": [5],
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

W = fastIca(Xw,  alpha=1)
invW = np.linalg.inv(W.T)

# Un-mix signals using
unMixed = Xw.T.dot(W.T)

plt.figure(0)
for i in range(num_columns):
    plt.subplot(num_columns, 1, i+1)
    plt.plot(unMixed[:, i])
    plt.ylim([-10, 10])

plt.show()

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

rhds = np.zeros(num_columns)
for i in range(num_columns):
    # Check all RHD values for different combinations
    mask = np.ones(num_columns, dtype=bool)
    mask[i] = False
    invW_trunc = invW.T[:, mask]
    model = np.dot(invW_trunc, unMixed[:, mask].T)
    for j in range(num_columns):
        rhds[i] += rhd(model[j, :], Xw[j, :])

plt.figure(1)
plt.plot(rhds)
plt.xlabel('Component Index')
plt.ylabel('Relative Hamming Distance')


# Re add mean
unMixed_scaled = (unMixed.T + meanX).T
