#
# Implements a neural network with 12-32-1 architecture and tanh activation function
#

from utilities import sym_sigmoid, dev_sym_sigmoid, softmax
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

path = "Data Sets\\GBP-USD Hourly.csv"


class NeuralNet:
    def __init__(self, path_to_data, learning_rate=0.01, do_sub_sample=False):
        self.mid_weights = (np.random.rand(12, 32) - 0.5) * (4.8 / 12)
        self.out_weights = (np.random.rand(32, 2) - 0.5) * (4.8 / 32)
        self.mid_bias = np.zeros(32)
        self.out_bias = np.random.rand(1, 2) * 2

        self.alpha = learning_rate
        # self.batch_size = 1
        self.path = path_to_data
        self.do_sub_sample = do_sub_sample

        self.data_from_file = None
        self.norm_data = None

        self.data, self.true = self.import_data()
        self.training_data, self.training_true, self.test_data, self.test_true = self.split_data(split=0.8)

        # self.training_data = np.array([[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        #                                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        #                                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        #                                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        #                                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        #                                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        #                                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        #                                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        #                                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        #                                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        #                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                ])
        #
        # self.train_true = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])

        print("Network Created:")
        # print("Batch Size:", self.batch_size)
        print("Learning Rate:", self.alpha)

    def import_data(self):
        # Create a 2D numpy array of date and hourly value
        self.data_from_file = pd.read_csv(self.path).values[:, 1]

        # Normalise input data
        # self.data_from_file = self.normalise_data(self.data_from_file)

        # Get true value as every 4th entry, starting from 12
        ground_truth = self.data_from_file[12:]

        # Initialise empty vector
        all_data = np.zeros([len(ground_truth), 12])

        # Create n 1x12 vectors and store them in a matrix
        for idx in range(len(ground_truth)):
            all_data[idx, :] = self.data_from_file[idx:idx + 12]

        true_direction = np.zeros([len(ground_truth), 2])

        for idx, dp in enumerate(all_data):
            if dp[-1] <= ground_truth[idx]:
                # Final value of input vector is smaller than true value
                # Thus correct prediction is downwards
                true_direction[idx, :] = [0, 1]
            else:
                # Else, true direction is up
                true_direction[idx, :] = [1, 0]

        return all_data, true_direction

    def normalise_data(self, data):
        # Subtracts mean and scales data by variance
        norm_data = (data - np.mean(data)) / np.var(data)
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
        mid_output = mid_output.flatten()
        exponents = np.dot(self.out_weights.T, mid_output) + self.out_bias.flatten()
        return softmax(exponents)

    # def calc_error(self, calc_on='all'):
    #     # Calculates error over training, test or all
    #     sum_error = 0
    #     if calc_on == 'all':
    #         # Calculate the error
    #         for data_row, true_value in zip(self.data, self.true):
    #             sum_error += 0.5 * (self.forward(data_row) - true_value) ** 2
    #         return sum_error / len(self.true)
    #     elif calc_on == 'training':
    #         # Calculate the error
    #         for data_row, true_value in zip(self.training_data, self.training_true):
    #             sum_error += 0.5 * (self.forward(data_row) - true_value) ** 2
    #         return sum_error / len(self.training_true)
    #     elif calc_on == 'test':
    #         # Calculate the error
    #         for data_row, true_value in zip(self.test_data, self.test_true):
    #             sum_error += - np.dot(np.log(self.forward(data_row)), true_value)
    #         return sum_error / len(self.test_true)
    #     else:
    #         print(calc_on, 'Not Implemented Yet')
    #         return None

    def calc_cross_entropy(self, calc_on='all'):
        # Calculates the cross entropy on the specified portion of the data
        sum_error = 0
        if calc_on == 'all':
            # Calculate the error
            for data_row, true_value in zip(self.data, self.true):
                sum_error += - np.dot(np.log(self.forward(data_row)), true_value)
            return sum_error / len(self.true)
        elif calc_on == 'training':
            # Calculate the error
            for data_row, true_value in zip(self.training_data, self.training_true):
                sum_error += - np.dot(np.log(self.forward(data_row)), true_value)
            return sum_error / len(self.training_true)
        elif calc_on == 'test':
            # Calculate the error
            for data_row, true_value in zip(self.test_data, self.test_true):
                sum_error += - np.dot(np.log(self.forward(data_row)), true_value)
            return sum_error / len(self.test_true)
        else:
            print(calc_on, 'Not Implemented Yet')
            return None

    def train_network(self, epochs):
        for n in range(epochs):
            for data_row, true_value in zip(self.training_data, self.training_true):
                # Create a random choice of indexes from the training data
                # batch_idx = np.random.choice(len(self.training_data), self.batch_size, replace=False)

                # Select random choice from training data and ground truth
                # data_batch = self.training_data[i:i + self.batch_size]
                # true_batch = self.training_true[i:i + self.batch_size]

                # Calculate the output of the middle layer of the network
                update = sym_sigmoid(np.matmul(data_row, self.mid_weights) + self.mid_bias)
                update = update.flatten()

                # Calculate the output of the network
                output = softmax(np.dot(self.out_weights.T, update.T) + self.out_bias)

                # Calculate the error for use in back propagation
                error = output - true_value

                # Update the final layer weights using gradient descent
                out_weights_grad = np.outer(update, error)
                self.out_weights -= (self.alpha * out_weights_grad)

                # Update final layer bias
                self.out_bias -= error

                # Update the hidden layer weights & bias using batch gradient descent
                sig_dev = dev_sym_sigmoid(update)

                weights_grad = np.outer(data_row.T, sig_dev * np.dot(error, self.out_weights.T))
                bias_grad = np.dot(error, self.out_weights.T) * sig_dev

                # Update weights by the gradient
                self.mid_weights = self.mid_weights - self.alpha * weights_grad.astype('float64')
                self.mid_bias = self.mid_bias - self.alpha * bias_grad.astype('float64')

            if n % 1 == 0:
                # Only calculate error every 50th epoch
                print(self.calc_cross_entropy(calc_on='training'))

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
        elif sample == 'train':
            for data_row in self.training_data:
                prdct_values.append(self.forward(data_row))
        elif sample == 'test':
            for data_row in self.test_data:
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
        x_vals = [i for i in range(start, start + steps)]
        plt.plot(x_vals, prdct_values)
        plt.legend(['True Values', 'Predicted Values'])

    def plot_predict_sngle(self):
        prdct_values = self.predict_sngle('all')
        plt.plot(self.true)
        plt.plot(prdct_values)
        plt.legend(['True Values', 'Predicted Values'])


network = NeuralNet(path, do_sub_sample=False)

network.train_network(epochs=100)
