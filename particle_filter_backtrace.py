import numpy as np
from tqdm import tqdm
from plot_utils import *



def trace_by_idx(start, stop, start_row, particle_history, index_history):
    # Start is index to start from (large number)
    # stop is index to stop at (small number)
    traced_history = np.zeros(start - stop + 1)
    propagate_idx = start_row
    for idx in range(stop, start + 1):
        traced_history[start - idx] = particle_history[propagate_idx, start + stop - idx]
        propagate_idx = index_history[propagate_idx, start + stop - idx]

    return traced_history



class ParticleFilter:

    def __init__(self, true_obs, num_particles=20, num_iterations=1, a=0.99,
                 b=1.0, c=1.0, learn_rate=0.0001, **kwargs):

        self.num_data = len(true_obs) - 1
        self.num_particles = num_particles
        self.num_iterations = num_iterations

        self.a = a
        self.b = b
        self.c = c
        self.learn_rate = learn_rate
        self.num_params = 3

        self.true_obs = true_obs
        self.particle_history = np.zeros([self.num_particles, self.num_data + 1])
        self.estimate_history = np.zeros(self.num_data + 1)
        self.params_history = np.zeros([self.num_params, self.num_iterations + 1])
        self.learn_rate_history = np.zeros(self.num_iterations + 1)
        self.weights_history = np.zeros([self.num_particles, self.num_data + 1])

        if 'true_hidden' in kwargs:
            self.true_hidden = kwargs['true_hidden']
            self.is_true_hidden = True
        else:
            self.true_hidden = None
            self.is_true_hidden = False
        if 'do_adaptive_learn' in kwargs:
            self.do_adaptive_learn = kwargs['do_adaptive_learn']
            self.alpha = 1
            self.adap_0 = (1 / self.learn_rate) - (self.num_iterations / 4)
        else:
            self.do_adaptive_learn = False
            self.alpha = 0

        self.adap_learn_rate = self.learn_rate

    def hidden_sample(self, x):
        noise = np.random.randn(len(x))
        return x * self.a + (np.sqrt(self.b) * noise)

    def _get_initial_sample(self):
        return np.sqrt(self.b / (1 - self.a**2)) * np.random.randn(self.num_particles)

    # @profile
    def filter_pass(self):
        # Run the particle filter once through the data
        initial_sample = self._get_initial_sample()
        initial_weights = self.observation(initial_sample, self.true_obs[0])
        weights = initial_weights / np.sum(initial_weights)

        self.particle_history[:, 0] = initial_sample

        self.particle_history2 = np.zeros_like(self.particle_history)
        self.particle_history2[:, 0] = initial_sample

        self.weights_history[:, 0] = weights
        self.estimate_history[0] = np.dot(weights, self.particle_history[:, 0])

        self.index_history = np.zeros_like(self.particle_history, dtype=int)
        # self.index_history[:, -1] = np.arange(self.num_particles)

        particle_range = np.arange(self.num_particles)

        for i in range(self.num_data):
            # Choose which particles to continue with using their weights
            particle_indexes = np.random.choice(particle_range, size=self.num_particles, p=weights)
            sorted_indexes = np.sort(particle_indexes)
            Xn = self.particle_history[sorted_indexes, i]

            self.index_history[:, i + 1] = sorted_indexes
            # Update the previous state history to be that of the chosen particles
            # self.particle_history[:, :i+1] = self.particle_history[sorted_indexes, :i+1]

            # Advance the hidden state by one time step
            Xn_plus_1 = self.hidden_sample(Xn)

            # Store the new state in the process history
            self.particle_history[:, i + 1] = Xn_plus_1
            self.particle_history2[:, i + 1] = Xn_plus_1

            # Make the new weights the likelihood of observing the known y for a given Xn hidden state
            new_particle_weights = self.observation(Xn_plus_1, self.true_obs[i + 1])

            # Store the updated weights in the weights history
            self.weights_history[:, i + 1] = new_particle_weights

            # Store the new normalised weights in the weights vector
            weights = new_particle_weights / np.sum(new_particle_weights)

            # Update the expectation for the best guess
            self.estimate_history[i + 1] = np.dot(weights, self.particle_history[:, i + 1])

        idx_to_prop = np.arange(self.num_particles)
        idx_stop = None
        for idx in range(self.num_data + 1):

            idx_to_prop = np.unique(self.index_history[:, self.num_data - idx][idx_to_prop])
            if len(idx_to_prop) == 1:
                # Have found final particle so stop
                idx_stop = idx + 1
                idx_to_prop = idx_to_prop[0]
                break

        if idx_stop is None:
            # no time saved just iterate through for each one
            raise Exception("not coded")
        else:
            particle_history = np.zeros_like(self.particle_history2)
            main_particle_history = trace_by_idx(self.num_data - idx_stop, 0, idx_to_prop, self.particle_history2, self.index_history)
            particle_history[:, :(self.num_data - idx_stop) + 1] = main_particle_history

            for i in range(self.num_particles):
                particle_history[i, (self.num_data - idx_stop) + 1:] = trace_by_idx(self.num_data, self.num_data - idx_stop + 1, i, self.particle_history2,
                                                                        self.index_history)

        self.particle_history3 = particle_history

    def calibrate_model(self, num_iterations=None):
        # Run the filter pass numerous times to optimise a,b,c,d
        if num_iterations is None:
            num_iterations = self.num_iterations
        else:
            self.params_history = np.zeros([self.num_params, num_iterations + 1])
            self.learn_rate_history = np.zeros(num_iterations + 1)

        self.params_history[:, 0] = [self.a, self.b, self.c]
        self.learn_rate_history[0] = self.adap_learn_rate

        for i in tqdm(range(num_iterations)):
            # Run filter to populate process & weights history
            self.clear_history()
            self.filter_pass()
            final_weights = self.weights_history[:, -1]
            final_weights_norm = final_weights / np.sum(final_weights)

            if self.do_adaptive_learn:
                self.adap_learn_rate = min((1 / (5*i+self.adap_0)) ** self.alpha, self.learn_rate)

            new_params = self._comp_param_update(final_weights_norm)

            # Store update to parameters
            self.params_history[:, i + 1] = new_params
            self.learn_rate_history[i + 1] = self.adap_learn_rate

    def clear_history(self, clear_params=False):
        # Reset history vectors to all zeros
        if clear_params:
            self.params_history = np.zeros([self.num_params, self.num_iterations + 1])
        self.particle_history = np.zeros([self.num_particles, self.num_data + 1])
        self.weights_history = np.zeros([self.num_particles, self.num_data + 1])

    def plot_filter_pass(self):
        # Plot the result of the particle filter pass to check it worked correctly
        plt.figure()
        if self.is_true_hidden:
            plt.plot(self.true_hidden)
            plt.legend(['True Hidden State'])
        # plt.plot(self.particle_history.T, '--', linewidth=0.4)
        plt.plot(self.estimate_history)

    def plot_params(self, title=""):
        plt.figure()
        plt.plot(self.params_history.T)
        plt.legend(self._get_param_symbols())
        plt.title("Parameter Evolution In Training for " + str(title))

    def plot_learn_rate(self):
        plt.figure()
        plt.plot(self.learn_rate_history)
        plt.title("Learning Rate History")
        plt.ylim([0, 1.1 * max(self.learn_rate_history)])


