import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from stochastic_volatility import gen_univ_sto_vol


class ParticleFilter:

    def __init__(self, true_obs, num_particles=20, num_iterations=1, **kwargs):

        self.num_data = len(true_obs) - 1
        self.num_particles = num_particles
        self.num_iterations = num_iterations

        self.burn_in = 100

        self.true_obs = true_obs
        self.particle_history = np.zeros([self.num_particles, self.num_data + 1])
        self.params_history = np.zeros([3, self.num_data + 1])
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
        # if 'mu' in kwargs:
        #     self.mu = kwargs['mu']
        # else:
        #     self.mu = 0
        if 'true_hidden' in kwargs:
            self.true_hidden = kwargs['true_hidden']
            self.is_true_hidden = True
        else:
            self.true_hidden = None
            self.is_true_hidden = False

    def hidden_sample(self, x):
        noise = np.random.randn(len(x))
        return (x * self.a) + (np.sqrt(self.b) * noise)

    def observation(self, x, y):
        sigma_sqrd = self.c * np.exp(x)
        log_obs = -0.5 * np.log(sigma_sqrd) - (y**2 / (2 * sigma_sqrd))
        return np.exp(log_obs)

    def calibrate_model(self):
        # Calibrate the model using online EM learning
        initial_sample = np.random.randn(self.num_particles)

        initial_weights = self.observation(initial_sample, self.true_obs[0])
        initial_weights = initial_weights / np.sum(initial_weights)

        self.particle_history[:, 0] = initial_sample

        weights = initial_weights

        self.weights_history[:, 0] = initial_weights
    
        particle_range = np.arange(self.num_particles)

        self.params_history[:, 0] = [self.a, self.b, self.c]

        one_step_sum = 0
        sqrd_minus_one_sum = 0
        sqrd_sum = 0
        exp_sum = 0
        is_first_time = True

        # weights_norm_constant = np.sum(initial_weights)
    
        for i in tqdm(range(self.num_data)):
            # Choose which particles to continue with using their weights
            particle_indexes = np.random.choice(particle_range, size=self.num_particles, p=weights)
            Xn = self.particle_history[particle_indexes, i]

            # Update the previous state history to be that of the chosen particles
            self.particle_history[:, :i] = self.particle_history[particle_indexes, :i]
    
            # Advance the hidden state by one time step
            Xn_plus_1 = self.hidden_sample(Xn)

            # Store the new state in the process history
            self.particle_history[:, i + 1] = Xn_plus_1
    
            # Make the new weights the likelihood of observing the known y for a given Xn hidden state
            new_particle_weights = self.observation(Xn_plus_1, self.true_obs[i+1])

            # Store the updated weights in the weights history
            self.weights_history[:, i + 1] = new_particle_weights

            # Calculate the normalisation constant
            weights_norm_constant = np.sum(new_particle_weights)

            # Store the new normalised weights in the weights vector
            weights = self.weights_history[:, i + 1] / weights_norm_constant

    # def calibrate_model(self):
    #     # Run the filter pass numerous times to optimise a,b,c
    #
    #     self.clear_history()

            if i > self.burn_in:
                # Allow some burn in
                if is_first_time:
                    one_step_sum = np.sum(self.particle_history[:, 1:i + 1] * self.particle_history[:, :i], axis=1)
                    sqrd_prdct = np.square(self.particle_history)
                    sqrd_minus_one_sum = np.sum(sqrd_prdct[:, :-1], axis=1)
                    sqrd_sum = np.sum(sqrd_prdct, axis=1)
                    exp_prdct = np.square(self.true_obs[:i+1]) * np.exp(-self.particle_history[:, :i + 1])
                    exp_sum = np.sum(exp_prdct, axis=1)
                    is_first_time = False

                # Calculate a' & b'
                one_step_sum += self.particle_history[:, i + 1] * self.particle_history[:, i]
                sqrd_sum += np.square(self.particle_history[:, i + 1])
                sqrd_minus_one_sum += np.square(self.particle_history[:, i])

                # Terms as labelled in notes
                term1 = np.dot(weights, one_step_sum)
                term2 = np.dot(weights, sqrd_minus_one_sum)
                term3 = np.dot(weights, sqrd_sum)

                # Ensure a is bounded between [0,1)
                self.a = min(max(term1 / term2, 0), 0.999)

                self.b = (term3 - (term1**2 / term2)) / (self.num_data + 1)

                # Calculate c'
                exp_sum += np.square(self.true_obs[i+1]) * np.exp(-self.particle_history[:, i + 1])
                self.c = max(np.dot(weights, exp_sum) / (self.num_data + 1), 0.1)

                # Store update to parameters
                self.params_history[:, i + 1] = [self.a, self.b, self.c]

    # def clear_history(self, clear_params=False):
    #     # Reset history vectors to all zeros
    #     raise Exception("Don't want to call this atm")
    #     if clear_params:
    #         self.params_history = np.zeros([3, self.num_data + 1])
    #     self.particle_history = np.zeros([self.num_particles, self.num_data + 1])
    #     self.weights_history = np.zeros([self.num_particles, self.num_data + 1])

    def plot_filter_pass(self):
        # Plot the result of the particle filter pass to check it worked correctly
        if self.is_true_hidden:
            plt.plot(self.true_hidden)
            plt.legend(['True Hidden State'])
        plt.plot(self.particle_history.T, '--', linewidth=0.4)

    def plot_params(self):
        plt.plot(self.params_history.T)
        plt.legend(['a', 'b', 'c'])


aa = 0.8
bb = 1
cc = 0.5

np.random.seed(0)

num_data = 5000
N = 500

test_x, test_y = gen_univ_sto_vol(num_data, a=aa, b=bb, c=cc, return_hidden=True)

particle_filter = ParticleFilter(test_y, num_particles=N, a=0.5, b=0.5, c=1, num_iterations=10)
# particle_filter.filter_pass()
# particle_filter.plot_filter_pass()
particle_filter.calibrate_model()
particle_filter.plot_params()
