from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

# Import data from csv
path = "Data Sets\\Electricity\\UK Elec Hourly - no weekends.csv"
df_from_file = pd.read_csv(path).values

price = df_from_file[:,2]
demand = df_from_file[:,1]

price = (price - np.mean(price)) / np.std(price)
demand = (demand - np.mean(demand)) / np.std(demand)

true = demand[24:]

# Initialise empty vector
all_data = np.zeros([len(true), 48])

# Create n 1x12 vectors and store them in a matrix
for idx in range(len(true)):
    all_data[idx, :] = np.append(price[idx:idx + 24], demand[idx:idx + 24])


split = 0.8
split_idx = round(len(true) * split)
train_data, test_data = all_data[:split_idx], all_data[split_idx:]
train_true, test_true = true[:split_idx], true[split_idx:]
shfl_idx = np.random.permutation(len(train_true))
train_data, train_true = train_data[shfl_idx], train_true[shfl_idx]

train_data = train_data.astype(np.float32)
train_true = train_true.astype(np.float32)

test_true = test_true.astype(np.float32)
test_data = test_data.astype(np.float32)

# ### TENSOR FLOW ###

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(48),
  tf.keras.layers.Dense(64, activation='tanh'),
  tf.keras.layers.Dense(32, activation='tanh'),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

model.fit(train_data, train_true, epochs=200, verbose=1)

model.evaluate(test_data,  test_true, verbose=2)
