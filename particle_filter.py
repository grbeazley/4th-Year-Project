import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from stochastic_volatility import gen_univ_sto_vol


class ParticleFilter:

    def __init__(self, true_obs, num_particles=20, num_iterations=1, **kwargs):

        self.num_data = len(true_obs) - 1
        self.num_particles = num_particles
        self.num_iterations = num_iterations

        self.true_obs = true_obs
        self.process_history = np.zeros([self.num_particles, self.num_data + 1])
        self.params_history = np.zeros([3, self.num_iterations + 1])
        self.weights_history = np.zeros([self.num_particles, self.num_data + 1])

        if 'x0' in kwargs:
            self.x_0 = kwargs['x0']
        else:
            self.x_0 = np.random.randn()
        if 'a' in kwargs:
            self.a = kwargs['a']
        else:
            self.a = 0.99
        if 'b' in kwargs:
            self.b = kwargs['b']
        else:
            self.b = 1
        if 'c' in kwargs:
            self.c = kwargs['c']
        else:
            self.c = 1
        if 'mu' in kwargs:
            self.mu = kwargs['mu']
        else:
            self.mu = 0
        if 'true_hidden' in kwargs:
            self.true_hidden = kwargs['true_hidden']
            self.is_true_hidden = True
        else:
            self.true_hidden = None
            self.is_true_hidden = False

    def hidden_sample(self, x):
        noise = np.random.randn(len(x))
        return x * self.a + np.sqrt(self.b) * noise

    def observation(self, x, y):
        sigma = np.sqrt(self.c) * np.exp(x / 2)
        log_obs = -np.log(sigma) - y ** 2 / (2 * sigma ** 2)
        return np.exp(log_obs)

    def filter_pass(self):
        # Run the particle filter once through the data
        initial_sample = np.random.randn(self.num_particles)

        initial_weights = self.observation(initial_sample, test_y[0])
        initial_weights = initial_weights / np.sum(initial_weights)

        self.process_history[:, 0] = initial_sample

        weights = initial_weights

        self.weights_history[:, 0] = initial_weights
    
        particle_range = np.arange(self.num_particles)
    
        weights_norm_constant = np.sum(initial_weights)
    
        for i in range(self.num_data):
            # Choose which particles to continue with using their weights
            particle_indexes = np.random.choice(particle_range, size=self.num_particles, p=weights)
            Xn = self.process_history[particle_indexes, i]

            # Update the previous state history to be that of the chosen particles
            self.process_history[:, :i] = self.process_history[particle_indexes, :i]
    
            # Advance the hidden state by one time step
            Xn_plus_1 = self.hidden_sample(Xn)

            # Store the new state in the process history
            self.process_history[:, i + 1] = Xn_plus_1
    
            # Make the new weights the likelihood of observing the known y for a given Xn hidden state
            new_particle_weights = self.observation(Xn_plus_1, test_y[i+1])

            # Store the updated weights in the weights history
            self.weights_history[:, i + 1] = new_particle_weights

            # Calculate the normalisation constant
            weights_norm_constant = np.sum(new_particle_weights)

            # Store the new normalised weights in the weights vector
            weights = self.weights_history[:, i + 1] / weights_norm_constant

    def calibrate_model(self):
        # Run the filter pass numerous times to optimise a,b,c
        self.clear_history()
        self.params_history[:, 0] = [self.a, self.b, self.c]
        for i in tqdm(range(self.num_iterations)):
            # Run filter to populate process & weights history
            self.filter_pass()
            final_weights = self.weights_history[:, -1]
            final_weights_norm = final_weights / np.sum(final_weights)

            # Calculate a' & b'
            one_step_sum = np.sum(self.process_history[:, 1:] * self.process_history[:, :-1], axis=1)
            sqrd_prdct = self.process_history * self.process_history
            sqrd_minus_one_sum = np.sum(sqrd_prdct[:, :-1], axis=1)
            sqrd_sum = np.sum(sqrd_prdct, axis=1)

            # Terms as labelled in notes
            term1 = np.dot(final_weights_norm, one_step_sum)
            term2 = np.dot(final_weights_norm, sqrd_minus_one_sum)
            term3 = np.dot(final_weights_norm, sqrd_sum)

            self.a = term1 / term2

            self.b = (term3 - (term1**2 / term2)) / (self.num_data + 1)

            # Calculate c'
            exp_prdct = np.square(self.true_obs) * np.exp(-self.process_history)
            exp_sum = np.sum(exp_prdct, axis=1)
            self.c = np.dot(final_weights_norm, exp_sum) / (self.num_data + 1)

            # Store update to parameters
            self.params_history[:, i + 1] = [self.a, self.b, self.c]

    def clear_history(self, clear_params=False):
        # Reset history vectors to all zeros
        if clear_params:
            self.params_history = np.zeros([3, self.num_iterations + 1])
        self.process_history = np.zeros([self.num_particles, self.num_data + 1])
        self.weights_history = np.zeros([self.num_particles, self.num_data + 1])

    def plot_filter_pass(self):
        # Plot the result of the particle filter pass to check it worked correctly
        if self.is_true_hidden:
            plt.plot(self.true_hidden)
            plt.legend(['True Hidden State'])
        plt.plot(self.process_history.T, '--', linewidth=0.4)

    def plot_params(self):
        plt.plot(particle_filter.params_history.T)
        plt.legend(['a', 'b', 'c'])


aa = 0.1
bb = 2
cc = 1

np.random.seed(0)

num_data = 500
N = 200

test_x, test_y = gen_univ_sto_vol(num_data, a=aa, b=bb, c=cc, return_hidden=True)

particle_filter = ParticleFilter(test_y, num_particles=N, a=0.95, b=0.2, c=1, num_iterations=50)
# particle_filter.filter_pass()
# particle_filter.plot_filter_pass()
particle_filter.calibrate_model()
