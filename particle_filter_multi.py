import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from data_gen import load_port
from stochastic_volatility import *
from plot_utils import *
from utilities import make_stationary, normalise


def trace_by_idx_multi(start, stop, start_row, particle_history, index_history):
    # Start is index to start from (large number)
    # Stop is index to stop at (small number)
    traced_history = np.zeros([particle_history.shape[0], start - stop + 1])
    propagate_idx = start_row
    for idx in range(stop, start + 1):
        traced_history[:, start - idx] = particle_history[:, propagate_idx, start + stop - idx]
        propagate_idx = index_history[propagate_idx, start + stop - idx]

    return traced_history


class ParticleFilterMulti:

    def __init__(self, true_obs, num_particles=20, num_iterations=1, p=2,
                 a=0.95, b=1.0, c=1.0, learn_rate=0.0001, learn_phi=1.0, learn_eta=1.0, learn_beta=1.0, **kwargs):

        self.p = p
        self.num_data = len(true_obs[0, :]) - 1
        self.num_particles = num_particles
        self.num_iterations = num_iterations

        self.phi = np.diag(np.ones(self.p)*a)
        self.eta = np.ones(self.p)*b
        self.beta = np.ones(self.p) * c
        self.learn_rate = learn_rate

        self.learn_phi = learn_phi
        self.learn_eta = learn_eta
        self.learn_beta = learn_beta

        self.num_params = self.p**2 + 2*p

        self.true_obs = true_obs
        self.prtcl_hist = np.zeros([self.p, self.num_particles, self.num_data + 1])
        self.estimate_history = np.zeros([self.p, self.num_data + 1])
        self.params_history = np.zeros([self.num_params, self.num_iterations + 1])
        self.learn_rate_history = np.zeros(self.num_iterations + 1)
        self.weights_history = np.zeros([self.num_particles, self.num_data + 1])
        self.likelihood_history = np.zeros([1, num_iterations])

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
        if 'phi' in kwargs:
            self.phi = kwargs['phi']
        if 'eta' in kwargs:
            self.eta = kwargs['eta']
        if 'beta' in kwargs:
            self.beta = kwargs['beta']

        self.adap_learn_rate = self.learn_rate

    def hidden_sample(self, x):
        mean = np.dot(self.phi, x)
        innovations = np.random.randn(self.p, self.num_particles)
        return mean + (innovations * np.sqrt(self.eta[:, np.newaxis]))

    def _get_initial_sample(self):
        return np.random.randn(self.p, self.num_particles)

    def observation(self, x, y):
        sigma = np.exp(x/2) * np.sqrt(self.beta[:, np.newaxis])
        log_obs = -np.log(np.prod(sigma, axis=0)) - (np.dot(np.square(y), 1/np.square(sigma)))/2
        return np.exp(log_obs)

    def filter_pass(self):
        # Run the particle filter once through the data
        initial_sample = self._get_initial_sample()
        initial_weights = self.observation(initial_sample, self.true_obs[:, 0])
        weights = initial_weights / np.sum(initial_weights)

        self.prtcl_hist[:, :, 0] = initial_sample

        part_hist_mix = np.zeros_like(self.prtcl_hist)
        part_hist_mix[:, :, 0] = initial_sample

        self.weights_history[:, 0] = weights
        self.estimate_history[:, 0] = np.dot(weights, self.prtcl_hist[:, :, 0].T)

        idx_hist = np.zeros([self.num_particles, self.num_data + 1], dtype=int)

        particle_range = np.arange(self.num_particles)

        for i in range(self.num_data):
            # Choose which particles to continue with using their weights
            particle_indexes = np.random.choice(particle_range, size=self.num_particles, p=weights)
            sorted_indexes = np.sort(particle_indexes)
            Xn = part_hist_mix[:, sorted_indexes, i]

            # Update the previous state history to be that of the chosen particles
            # self.prtcl_hist[:, :, :i + 1] = self.prtcl_hist[:, sorted_indexes, :i + 1]
            idx_hist[:, i + 1] = sorted_indexes

            # Advance the hidden state by one time step
            Xn_plus_1 = self.hidden_sample(Xn)

            # Store the new state in the process history
            part_hist_mix[:, :, i + 1] = Xn_plus_1

            # Make the new weights the likelihood of observing the known y for a given Xn hidden state
            new_particle_weights = np.clip(self.observation(Xn_plus_1, self.true_obs[:, i + 1]), 0, None)

            # Store the updated weights in the weights history
            self.weights_history[:, i + 1] = new_particle_weights

            # Store the new normalised weights in the weights vector
            weights = new_particle_weights / np.sum(new_particle_weights)

            # Update the expectation for the best guess
            self.estimate_history[:, i + 1] = np.dot(weights, part_hist_mix[:, :, i + 1].T)

        # Find the dominant particle trajectory
        idx_to_prop = np.arange(self.num_particles)
        idx_stop = None
        for idx in range(self.num_data + 1):

            idx_to_prop = np.unique(idx_hist[:, self.num_data - idx][idx_to_prop])
            if len(idx_to_prop) == 1:
                # Have found final particle so stop
                idx_stop = idx + 1
                idx_to_prop = idx_to_prop[0]
                break

        if idx_stop is None:
            # No time saved just iterate through for each one
            print("No Dominant Particle Found")
            particle_history = np.zeros_like(part_hist_mix)
            for i in range(self.num_particles):
                particle_history[:, i, :] = \
                    trace_by_idx_multi(self.num_data, 0, i, part_hist_mix, idx_hist)
        else:
            # Found dominant particle, starting at idx_stop
            particle_history = np.zeros_like(part_hist_mix)
            main_particle_history = trace_by_idx_multi(self.num_data - idx_stop, 0, idx_to_prop,
                                                       part_hist_mix, idx_hist)
            particle_history[:, :, :(self.num_data - idx_stop) + 1] = main_particle_history[:, None, :]

            for i in range(self.num_particles):
                particle_history[:, i, (self.num_data - idx_stop) + 1:] \
                    = trace_by_idx_multi(self.num_data, self.num_data - idx_stop + 1, i, part_hist_mix, idx_hist)

        self.prtcl_hist = particle_history

    def calibrate_model(self, num_iterations=None):
        # Run the filter pass numerous times to optimise a,b,c,d
        if num_iterations is None:
            num_iterations = self.num_iterations
        else:
            self.params_history = np.zeros([self.num_params, num_iterations + 1])
            self.learn_rate_history = np.zeros(num_iterations + 1)
            self.likelihood_history = np.zeros([1, num_iterations])

        self.params_history[:, 0] = np.concatenate((self.phi.flatten(), self.eta.flatten(), self.beta.flatten()))
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
            self.likelihood_history[:, i] = np.sum(np.log(np.sum(self.weights_history, axis=0) / self.num_particles))

    def _comp_param_update(self, fwn):
        # Calculate the gradient with respect to the parameters and update them
        dl_dphi = np.zeros_like(self.phi)
        for i in range(self.p):
            for j in range(self.p):
                frst = ((1/self.eta[i]) * self.prtcl_hist[i, :, 1:] * self.prtcl_hist[j, :, :-1])
                scnd = ((1/self.eta[i]) * self.prtcl_hist[j, :, 1:] *
                        (np.dot(self.phi[i, :], self.prtcl_hist[:, :, 1:].transpose(1, 0, 2))))
                dl_dphi[i, j] = np.dot(fwn, np.sum(frst - scnd, axis=1))

        dl_deta = np.zeros_like(self.eta)

        for i in range(self.p):
            mu = np.dot(self.phi[i, :], self.prtcl_hist[:, :, :-1].transpose(1, 0, 2))
            smnd = np.square(self.prtcl_hist[i, :, 1:])/2 - self.prtcl_hist[i, :, 1:]*mu + np.square(mu)/2

            dl_deta[i] = np.dot(fwn, np.sum(smnd/np.square(self.eta[i]) - 0.5/self.eta[i], axis=1))

        self.eta = np.clip(self.eta + (self.learn_rate * dl_deta * self.learn_eta), 0.001, None)
        self.phi = self.phi + (self.learn_rate * dl_dphi * self.learn_phi)

        dl_dbeta = np.zeros_like(self.beta)
        for i in range(self.p):
            log_smnd = np.log(np.square(self.true_obs[i, :])) - self.prtcl_hist[i, :, :]
            smnd = 0.5 * np.exp(log_smnd - 2*np.log(self.beta[i]))
            dl_dbeta[i] = np.dot(fwn, np.sum(smnd - 0.5/self.beta[i], axis=1))

        self.beta = np.clip(self.beta + (self.learn_rate * dl_dbeta * self.learn_beta), 0.001, None)

        return np.concatenate((self.phi.flatten(), self.eta.flatten(), self.beta.flatten()))

    def clear_history(self, clear_params=False):
        # Reset history vectors to all zeros
        if clear_params:
            self.params_history = np.zeros([self.num_params, self.num_iterations + 1])
        self.prtcl_hist = np.zeros([self.p, self.num_particles, self.num_data + 1])
        self.weights_history = np.zeros([self.num_particles, self.num_data + 1])

    def plot_filter_pass(self):
        # Plot the result of the particle filter pass to check it worked correctly
        plt.figure()
        cols = ['b', 'orange', 'g', 'r', 'm', 'y', 'k']
        for i in range(self.p):
            if self.is_true_hidden:
                plt.plot(self.true_hidden[i, :].T, label="True Component " + str(i+1), color=cols[i])
                # plt.legend(['True Hidden State'])
        # plt.plot(self.prtcl_hist.T, '--', linewidth=0.4)
            plt.plot(self.estimate_history[i, :].T, '--', color=cols[i], label="Predicted Component " + str(i+1))

    def plot_params(self, title=""):
        plt.figure()
        plt.plot(self.params_history.T)
        # plt.legend(self._get_param_symbols())
        plt.title("Parameter Evolution In Training for " + str(title))

    def plot_learn_rate(self):
        plt.figure()
        plt.plot(self.learn_rate_history)
        plt.title("Learning Rate History")
        plt.ylim([0, 1.1 * max(self.learn_rate_history)])

    def update_learn_rates(self, new_rates):
        self.learn_phi = new_rates[0]
        self.learn_eta = new_rates[1]
        self.learn_beta = new_rates[2]

    def plot_likelihood(self, title=""):
        plt.figure()
        plt.plot(self.likelihood_history.T)
        plt.title("Likelihood Evolution In Training for " + str(title))

    def update_num_particles(self, new_number):
        self.num_particles = new_number
        self.prtcl_hist = np.zeros([self.p, self.num_particles, self.num_data + 1])
        self.weights_history = np.zeros([self.num_particles, self.num_data + 1])

    def one_predict_hidden(self):
        final_weights = self.weights_history[:, -1]
        fwn = final_weights / np.sum(final_weights)

        particle_indexes = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=fwn)
        x_prev = self.prtcl_hist[:, particle_indexes, -1]
        predictions = self.hidden_sample(x_prev)

        return predictions

    def one_step_multi(self, test_obs_data):
        num_points = np.shape(test_obs_data)[1]
        predictions = np.zeros([self.p, self.num_particles, num_points])

        self.true_obs = self.true_obs[:, -101:]
        self.num_data = 100

        for i in tqdm(range(num_points)):
            predictions[:, :, i] = self.one_predict_hidden()
            self.true_obs = np.hstack((self.true_obs, test_obs_data[:, i, np.newaxis]))
            self.num_data += 1
            self.clear_history()
            self.filter_pass()

        return predictions


if __name__ == "__main__":
    np.random.seed(0)
    num_data = 5000
    N = 1000
    num_dims = 6
    train = 4616

    # phi = 0.95 * np.array([[0.7, 0.1, 0.1, 0.1],
    #                        [0.1, 0.7, 0.1, 0.1],
    #                        [0.1, 0.1, 0.7, 0.1],
    #                        [0.1, 0.1, 0.1, 0.7]], dtype=float)

    phi = np.array([[0.82, 0.08, -0.02, 0.06, 0.05, 0.06],
                    [0.18, 0.53, 0.02, -0.01, 0.05, 0.02],
                    [0.25, 0.18, 0.22, 0.15, -0.01, -0.01],
                    [0.16, 0.0, -0.02, 0.65, 0.17, 0.05],
                    [0.17, 0.03, -0.02, 0.11, 0.63, 0.08],
                    [0.16, 0.0, -0.01, 0.03, 0.12, 0.61]], dtype=float)

    eta = np.array([0.734, 0.268, 0.32, 0.53, 0.57, 0.52])
    beta = np.array([2, 2, 2, 2, 2, 2])

    # phi = 0.95 * np.array([[1, 0, 0, 0],
    #                        [0, 1, 0, 0],
    #                        [0, 0, 1, 0],
    #                        [0, 0, 0, 1]], dtype=float)

    # data_h, data_y = gen_multi_sto_vol(num_data, num_dims, phi=phi, var_latent=1, var_observed=5, return_hidden=True)

    data_y, dates = load_port(False)

    data_y_norm, y_means, y_stds = normalise(data_y, return_params=True)
    # data_y_norm = data_y
    data_train = data_y_norm[:, :train]
    data_test = data_y_norm[:, train:]

    pf = ParticleFilterMulti(data_train,
                             num_particles=N,
                             p=num_dims,
                             a=0.7,
                             b=0.9,
                             c=2,
                             # true_hidden=data_h,
                             num_iterations=2,
                             learn_rate=0.1/num_data,
                             learn_phi=1,
                             learn_eta=5,
                             learn_beta=0,
                             phi=phi,
                             eta=eta,
                             beta=beta,
                             )

    pf.filter_pass()
    # pf.calibrate_model()
    # pf.plot_params()
    # pf.plot_likelihood()
    predictions = pf.one_step_multi(data_test)
    # pf.plot_filter_pass()
