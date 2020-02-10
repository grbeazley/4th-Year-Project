import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from stochastic_volatility import gen_univ_sto_vol


class ParticleFilter:

    def __init__(self, true_obs, num_particles=20, num_iterations=1, **kwargs):

        self.num_data = len(true_obs) - 1
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.learn_rate = 0.001

        self.true_obs = true_obs
        self.particle_history = np.zeros([self.num_particles, self.num_data + 1])
        self.estimate_history = np.zeros(self.num_data + 1)
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
        return x * self.a + (np.sqrt(self.b) * noise)

    def observation(self, x, y):
        sigma = np.sqrt(self.c) * np.exp(x / 2)
        log_obs = -np.log(sigma) - (y ** 2 / (2 * sigma ** 2))
        return np.exp(log_obs)

    def filter_pass(self):
        # Run the particle filter once through the data
        initial_sample = np.sqrt(self.b / (1 - self.a**2)) * np.random.randn(self.num_particles)
        initial_weights = self.observation(initial_sample, test_y[0])
        weights = initial_weights / np.sum(initial_weights)

        self.particle_history[:, 0] = initial_sample
        self.weights_history[:, 0] = weights
        self.estimate_history[0] = np.dot(weights, self.particle_history[:, 0])

        particle_range = np.arange(self.num_particles)

        for i in range(self.num_data):
            # Choose which particles to continue with using their weights
            particle_indexes = np.random.choice(particle_range, size=self.num_particles, p=weights)
            Xn = self.particle_history[np.sort(particle_indexes), i]

            # Update the previous state history to be that of the chosen particles
            self.particle_history[:, :i+1] = self.particle_history[np.sort(particle_indexes), :i+1]

            # Advance the hidden state by one time step
            Xn_plus_1 = self.hidden_sample(Xn)

            # Store the new state in the process history
            self.particle_history[:, i + 1] = Xn_plus_1

            # Make the new weights the likelihood of observing the known y for a given Xn hidden state
            new_particle_weights = self.observation(Xn_plus_1, self.true_obs[i + 1])

            # Store the updated weights in the weights history
            self.weights_history[:, i + 1] = new_particle_weights

            # Store the new normalised weights in the weights vector
            weights = new_particle_weights / np.sum(new_particle_weights)

            # Update the expectation for the best guess
            self.estimate_history[i + 1] = np.dot(weights, self.particle_history[:, i + 1])

    def calibrate_model(self):
        # Run the filter pass numerous times to optimise a,b,c
        self.params_history[:, 0] = [self.a, self.b, self.c]

        for i in tqdm(range(self.num_iterations)):
            # Run filter to populate process & weights history
            self.clear_history()
            self.filter_pass()
            final_weights = self.weights_history[:, -1]
            final_weights_norm = final_weights / np.sum(final_weights)

            # Compute the gradient of the log likelihood w.r.t. a
            sqrd_minus_one_prdct = self.particle_history[:, :-1] * self.particle_history[:, 1:]
            summand_a = (sqrd_minus_one_prdct - (np.square(self.particle_history[:, :-1]) * self.a)) / self.b
            dl_da = np.dot(np.sum(summand_a, axis=1), final_weights_norm)

            # Update parameter, ensuring it retains stationarity
            self.a = min(self.a + self.learn_rate*dl_da, 0.999)

            # Compute the gradient of the log likelihood w.r.t. b
            sqrd_prdct = np.square(self.particle_history[:, 1:] - self.a * self.particle_history[:, :-1])
            sum_b = np.sum(sqrd_prdct - self.b, axis=1) / 2*self.b**2
            dl_db = np.dot(sum_b, final_weights_norm)

            # Update parameter b
            self.b = self.b + self.learn_rate * 2.5 * dl_db

            # Compute the gradient of the log likelihood w.r.t. c
            summand_c = np.square(self.true_obs) / (2 * self.c * np.exp(self.particle_history))
            sum_c = np.sum(summand_c - self.c / 2, axis=1)
            dl_dc = np.dot(sum_c, final_weights_norm)

            # Update parameter c
            self.c = self.c + self.learn_rate * dl_dc

            # Store update to parameters
            self.params_history[:, i + 1] = [self.a, self.b, self.c]

    def clear_history(self, clear_params=False):
        # Reset history vectors to all zeros
        if clear_params:
            self.params_history = np.zeros([3, self.num_iterations + 1])
        self.particle_history = np.zeros([self.num_particles, self.num_data + 1])
        self.weights_history = np.zeros([self.num_particles, self.num_data + 1])

    def plot_filter_pass(self):
        # Plot the result of the particle filter pass to check it worked correctly
        plt.figure()
        if self.is_true_hidden:
            plt.plot(self.true_hidden)
            plt.legend(['True Hidden State'])
        plt.plot(self.particle_history.T, '--', linewidth=0.4)
        plt.plot(self.estimate_history)

    def plot_params(self):
        plt.figure()
        plt.plot(self.params_history.T)
        plt.legend(['a', 'b', 'c'])


if __name__ == "__main__":
    np.random.seed(0)

    aa = 0.9
    bb = 0.5
    cc = 0.5

    num_data = 200
    N = 200

    test_x, test_y = gen_univ_sto_vol(num_data, a=aa, b=bb, c=cc, return_hidden=True)

    particle_filter = ParticleFilter(test_y,
                                     num_particles=N,
                                     a=0.7,
                                     b=1,
                                     c=1,
                                     true_hidden=test_x,
                                     num_iterations=250)

    particle_filter.filter_pass()
    particle_filter.plot_filter_pass()
    particle_filter.calibrate_model()
    particle_filter.plot_params()
    particle_filter.plot_filter_pass()