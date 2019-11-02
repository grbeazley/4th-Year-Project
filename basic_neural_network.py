#
# Implements a neural network with 12-32-1 architecture and tanh activation function
#

from utilities import sym_sigmoid, dev_sym_sigmoid
import numpy as np

# Create a dummy data vector for testing
data = np.linspace(0, 11, 12)

# Initialise a random weights matrix for the middle layer
# Set the values to +- 2.4/Ii
weights = (np.random.rand(12, 32) - 0.5) * (4.8 / 12)

# Initialise the bias to all zeros
bias = np.zeros(32)

# Initialise random weights for the final layer (note different Ii)
out_weights = (np.random.rand(32) - 0.5) * (4.8 / 32)

# Calculate the activation function on the product of the weights and inputs
update = sym_sigmoid(np.matmul(data, weights) + bias)

# Calculate the final output step of the network
output = np.dot(update, out_weights)


