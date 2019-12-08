#
# Implements a neural network with 12-32-1 architecture and tanh activation function
#

from utilities import sym_sigmoid, dev_sym_sigmoid
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

path = "Data Sets\\GBP-USD Hourly.csv"


class NeuralNet:
    def __init__(self, path_to_data, learning_rate=0.0005):
        self.mid_weights = (np.random.rand(24, 64) - 0.5) * (4.8 / 24)
        self.out_weights = (np.random.rand(64) - 0.5) * (4.8 / 64)
        self.mid_bias = np.zeros(64)
        self.alpha = learning_rate
        self.path = path_to_data
        self.data_from_file = None

        self.data, self.true = self.import_data()
        self.training_data, self.training_true, self.test_data, self.test_true = self.split_data(split=0.8)

    def import_data(self):
        # Create a 2D numpy array of date and hourly value
        self.data_from_file = pd.read_csv(self.path).values

        # Get true value as every 4th entry, starting from 12
        ground_truth = self.data_from_file[24:, 1]

        # Initialise empty vector
        all_data = np.zeros([len(ground_truth), 24])

        # Create n 1x12 vectors and store them in a matrix
        for idx in range(len(ground_truth)):
            all_data[idx, :] = self.data_from_file[idx:idx + 24, 1]

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
            return sum_error / len(self.true)
        elif calc_on == 'training':
            # Calculate the error
            for data_row, true_value in zip(self.training_data, self.training_true):
                sum_error += 0.5 * (self.forward(data_row) - true_value) ** 2
            return sum_error / len(self.training_true)
        elif calc_on == 'test':
            # Calculate the error
            for data_row, true_value in zip(self.test_data, self.test_true):
                sum_error += 0.5 * (self.forward(data_row) - true_value) ** 2
            return sum_error / len(self.test_true)
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
                # Only calculate error every 50th epoch
                print(self.calc_error(calc_on='training'))

    def predict_multi(self, steps=0):
        prdct_values = []

        if steps == 0:
            steps = len(self.test_true)

        input_values = self.training_data[-1, :]
        for i in range(steps):
            prdct_output = self.forward(input_values)
            prdct_values.append(prdct_output)
            input_values = np.append(input_values[1:], prdct_output)

        return prdct_values

    def plot_predict_multi(self, steps=0):
        # Plots a comparison between predicted and true movement
        prdct_values = self.predict_multi(steps)
        plt.plot(self.test_true)
        plt.plot(prdct_values)
        plt.legend(['True Values', 'Predicted Values'])


network = NeuralNet(path)

network.train_network(epochs=5000)

