#
# Implements a neural network with 12-32-1 architecture and tanh activation function
#

from utilities import sym_sigmoid, dev_sym_sigmoid
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# path = "Data Sets\\Forex\\GBP-USD Hourly.csv"
path = "Data Sets\\Electricity\\UK Elec Hourly - no weekends.csv"
input_size = 24


class NeuralNet:
    def __init__(self, path_to_data, learning_rate=0.005, do_sub_sample=True):
        self.mid_weights = (np.random.rand(input_size, 32) - 0.5) * (4.8 / input_size)
        self.out_weights = (np.random.rand(32) - 0.5) * (4.8 / 32)
        self.mid_bias = np.zeros(32)
        self.out_bias = np.random.rand(1) * 2

        self.alpha = learning_rate
        self.path = path_to_data
        self.do_sub_sample = do_sub_sample

        self.data_from_file = None
        self.norm_data = None

        self.data, self.true = self.import_data()
        self.training_data, self.training_true, self.test_data, self.test_true = self.split_data(split=0.8)

    def import_data(self):
        # Create a 2D numpy array of date and hourly value
        self.data_from_file = pd.read_csv(self.path).values[:, 1]

        # Normalise input data
        self.data_from_file = self.normalise_data(self.data_from_file)

        # Get true value as every 4th entry, starting from 12
        ground_truth = self.data_from_file[input_size:]

        # Initialise empty vector
        all_data = np.zeros([len(ground_truth), input_size])

        # Create n 1x12 vectors and store them in a matrix
        for idx in range(len(ground_truth)):
            all_data[idx, :] = self.data_from_file[idx:idx + input_size]

        return all_data, ground_truth

    def normalise_data(self, data):
        # Subtracts mean and scales data
        norm_data = (data - np.mean(data))/np.std(data)

        return norm_data

    def subsample_data(self, data, true):
        # Sub Sample data for training
        data_sub = data[::4]
        true_sub = true[::4]

        return data_sub, true_sub

    def split_data(self, split=0.8):
        # Perform sub sampling if requested
        if self.do_sub_sample:
            data, true = self.subsample_data(self.data, self.true)
        else:
            # Otherwise use full data sets
            data, true = self.data, self.true

        # Have to take into account time series nature but can shuffle training data
        split_idx = round(len(true) * split)
        train_data, test_data = data[:split_idx], data[split_idx:]
        train_true, test_true = true[:split_idx], true[split_idx:]
        shfl_idx = np.random.permutation(len(train_true))
        return train_data[shfl_idx], train_true[shfl_idx], test_data, test_true

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

                # Update the final layer bias using gradient descent
                out_bias_grad = - error
                self.out_bias -= self.alpha * out_bias_grad

                # Update the hidden layer weights & bias using gradient descent
                sig_dev = dev_sym_sigmoid(update)
                weights_grad = -error * np.outer(data_row, (self.out_weights * sig_dev))
                bias_grad = -error * self.out_weights * sig_dev

                self.mid_weights -= self.alpha * weights_grad
                self.mid_bias -= self.alpha * bias_grad

            if n % 50 == 0:
                # Only calculate error every 50th epoch
                print(self.calc_error(calc_on='training'))

    def predict_multi(self, steps=-1, start=0):
        # Predicts multiple steps forward

        input_buffer = self.data[start, :]
        prdct_values = []

        for i in range(steps):
            prdct_output = self.forward(input_buffer)
            prdct_values.append(prdct_output)
            input_buffer = np.append(input_buffer[1:], prdct_output)

        return prdct_values

    def predict_sngle(self, sample='all'):
        prdct_values = []
        if sample == 'all':
            for data_row in self.data:
                prdct_values.append(self.forward(data_row))
        else:
            print(sample, 'has not been implemented yet')

        return prdct_values

    def plot_predict_multi(self, steps=-1, start=0):
        # Plots a comparison between predicted and true movement
        if steps == -1:
            steps = len(self.true[start:])
        
        steps = min(steps, len(self.true) - start)        
        prdct_values = self.predict_multi(steps, start)
        plt.plot(self.true)        
        x_vals = [i for i in range(start, start+steps)]
        plt.plot(x_vals, prdct_values)
        plt.legend(['True Values', 'Predicted Values'])

    def plot_predict_sngle(self):
        prdct_values = self.predict_sngle('all')
        plt.plot(self.true)
        plt.plot(prdct_values)
        plt.legend(['True Values', 'Predicted Values'])


network = NeuralNet(path, do_sub_sample=False)

network.train_network(epochs=100)
