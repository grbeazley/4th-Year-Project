#
# Implements a neural network with 12-32-1 architecture and tanh activation function
#

from utilities import sym_sigmoid, dev_sym_sigmoid
import numpy as np
import pandas as pd
import matplotlib


def forward(data, weights, out_weights, bias):
    # Calculate the vector of outputs from the hidden layer
    update = sym_sigmoid(np.matmul(data, weights) + bias)

    # Calculate the final output step of the network
    return np.dot(update, out_weights)


def calc_error(data, true, weights, out_weights, bias):
    sum_error = 0
    for data_row, true_value in zip(data, true):
        sum_error += 0.5 * (forward(data_row, weights, out_weights, bias) - true_value)**2
    return sum_error


# Create a 2D numpy array of date and hourly value
data_in = pd.read_csv("Data Sets\\GBP-USD Hourly.csv").values

# Get true value as every 4th entry
true = data_in[12::4, 1]

data = np.zeros([len(true), 12])

for i in range(len(true)):
    data[i, :] = data_in[4*i:4*i+12, 1]


# Hyperparameter definition
alpha = 0.001  # Learning rate

# Initialise a random weights matrix for the middle layer
# Set the values to between +- 2.4/Ii
weights = (np.random.rand(12, 32) - 0.5) * (4.8 / 12)

# Initialise the bias to all zeros
bias = np.zeros(32)

# Initialise random weights for the final layer (note different Ii)
out_weights = (np.random.rand(32) - 0.5) * (4.8 / 32)
out_bias = 0

# Run for specified number of epochs
epochs = 1000

# Iterate through all of the data for each epoch
for n in range(epochs):
    for data_row, true_value in zip(data, true):
        # Calculate the output of the network
        # output = forward(data_row, weights, out_weights, bias)

        update = sym_sigmoid(np.matmul(data_row, weights) + bias)

        output = np.dot(update, out_weights)

        # Calculate the error of the network
        error = true_value - output

        # Update the final layer weights using gradient descent
        out_weights_grad = update * - error
        out_weights -= alpha * out_weights_grad

        # Update the hidden layer weights & bias using gradient descent
        derivative = dev_sym_sigmoid(update)
        weights_grad = - error * np.outer(data_row, (out_weights * derivative))
        bias_grad = -error * out_weights * derivative

        weights -= alpha * weights_grad
        bias -= alpha * bias_grad

    if n % 50 == 0:
        # Only calculate error every 50th iteration
        print(calc_error(data, true, weights, out_weights, bias))
