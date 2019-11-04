#
# Implements a neural network with 12-32-1 architecture and tanh activation function
#

from utilities import sym_sigmoid, dev_sym_sigmoid
import numpy as np
import pandas as pd
import matplotlib

path = "Data Sets\\GBP-USD Hourly.csv"


class NeuralNet:
    def __init__(self, path_to_data, learning_rate=0.001):
        self.mid_weights = (np.random.rand(12, 32) - 0.5) * (4.8 / 12)
        self.out_weights = (np.random.rand(32) - 0.5) * (4.8 / 32)
        self.mid_bias = np.zeros(32)
        self.alpha = learning_rate
        self.path = path_to_data

        self.data, self.true = self.import_data()
        self.training_data, self.training_true, self.test_data, self.test_true = self.split_data(split=0.8)

    def import_data(self):
        # Create a 2D numpy array of date and hourly value
        data_from_file = pd.read_csv(self.path).values

        # Get true value as every 4th entry, starting from 12
        ground_truth = data_from_file[12::4, 1]

        # Initialise empty vector
        all_data = np.zeros([len(ground_truth), 12])

        # Create n 1x12 vectors and store them in a matrix
        for idx in range(len(ground_truth)):
            all_data[idx, :] = data_from_file[4 * idx:4 * idx + 12, 1]

        return all_data, ground_truth

    def split_data(self, split=0.8):
        # Have to take into account time series nature
        split_idx = round(len(self.true) * split)
        return self.data[:split_idx], self.true[:split_idx], self.data[split_idx:], self.true[split_idx:]

    def forward(self, data_row):
        # Calculate the vector of outputs from the hidden layer
        mid_output = sym_sigmoid(np.matmul(data_row, self.mid_weights) + self.mid_bias)
        return np.dot(mid_output, self.out_weights)

    def calc_error(self, calc_on='all'):
        # Calculates error over training, test or all
        sum_error = 0
        if calc_on == 'all':
            # Calculate the error
            for data_row, true_value in zip(self.data, self.true):
                sum_error += 0.5 * (self.forward(data_row) - true_value)**2
            return sum_error
        elif calc_on == 'training':
            # Calculate the error
            for data_row, true_value in zip(self.training_data, self.training_true):
                sum_error += 0.5 * (self.forward(data_row) - true_value) ** 2
            return sum_error
        else:
            print(calc_on, 'Not Implemented Yet')
            return None

    def train_network(self, epochs):
        for n in range(epochs):
            for data_row, true_value in zip(self.training_data, self.training_true):
                # Calculate the output of the middle layer of the network
                update = sym_sigmoid(np.matmul(data_row, self.mid_weights) + self.mid_bias)

                # Calculate the output of the network
                output = np.dot(update, self.out_weights)

                # Calculate the error of the network
                error = true_value - output

                # Update the final layer weights using gradient descent
                out_weights_grad = update * - error
                self.out_weights -= self.alpha * out_weights_grad

                # Update the hidden layer weights & bias using gradient descent
                sig_dev = dev_sym_sigmoid(update)
                weights_grad = -error * np.outer(data_row, (self.out_weights * sig_dev))
                bias_grad = -error * self.out_weights * sig_dev

                self.mid_weights -= self.alpha * weights_grad
                self.mid_bias -= self.alpha * bias_grad

            if n % 50 == 0:
                # Only calculate error every 50th iteration
                print(self.calc_error(calc_on='training'))


network = NeuralNet(path)

network.train_network(10)

