import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from stochastic_volatility import gen_univ_mrkv, gen_univ_sto_vol, gen_univ_gamma
from pdfs import gamma_pdf
from plot_utils import *
from particle_filter import ParticleFilter
from scipy.special import gamma as gamma_function
from scipy.special import digamma as gamma_deriv


class ParticleFilterGamma(ParticleFilter):

    def __init__(self, true_obs, num_particles=20, num_iterations=1, a=0.99,
                 b=1.0, c=1.0, learn_rate=0.0001, k=1.0, theta=1.0, **kwargs):

        ParticleFilter.__init__(self, true_obs, num_particles, num_iterations, a,
                                b, c, learn_rate, **kwargs)

        self.k = k
        self.theta = theta

    def observation(self, x, y):
        # Probability of observing y given x
        theta = np.exp(x/2) * self.theta * np.sqrt(self.c)
        obs = gamma_pdf(y, self.k, theta)
        return obs

    def _get_initial_sample(self):
        return np.sqrt(self.b / (1 - self.a ** 2)) * np.random.randn(self.num_particles)

    @staticmethod
    def _get_param_symbols():
        return ['$\\phi$', '$\\sigma^2$', '$\\beta$']

    def _comp_param_update(self, final_weights_norm):
        # Compute the gradient of the log likelihood w.r.t. a
        sqrd_minus_one_prdct = self.particle_history[:, :-1] * self.particle_history[:, 1:]
        summand_a = (sqrd_minus_one_prdct - (np.square(self.particle_history[:, :-1]) * self.a)) / self.b
        dl_da = np.dot(np.sum(summand_a, axis=1), final_weights_norm)

        # Update parameter, ensuring it retains stationarity
        self.a = min(self.a + self.adap_learn_rate * dl_da, 0.9999)

        # Compute the gradient of the log likelihood w.r.t. b
        sqrd_prdct = np.square(self.particle_history[:, 1:] - self.a * self.particle_history[:, :-1])
        sum_b = np.sum(sqrd_prdct - self.b, axis=1) / 2 * (self.b ** 2)
        dl_db = np.dot(sum_b, final_weights_norm)

        # Update parameter b
        self.b = max(0.0001, self.b + self.adap_learn_rate * dl_db * 2.5)

        th_exp_term = np.exp(self.particle_history / 2) * self.theta
        summand_c = self.true_obs / (2 * th_exp_term * self.c ** (3 / 2))
        sum_c = np.sum(summand_c - (self.k / (2 * self.c)), axis=1)
        dl_dc = np.dot(sum_c, final_weights_norm)

        # Update parameter c
        self.c = max(0.0001, self.c + self.adap_learn_rate * dl_dc)

        # Return update to parameters
        return [self.a, self.b, self.c]


if __name__ == "__main__":
    # np.random.seed(0)

    aa = 0.85
    bb = 0.5

    c_true = 0.5
    c_start = 1

    kk = 1.3
    thth = 0.5

    num_data = 1000
    N = 250

    test_x, test_y = gen_univ_gamma(num_data, a=aa, b=bb, c=c_true, k=kk, theta=thth, return_hidden=True)
    # test_x, test_y = gen_univ_sto_vol(num_data, a=aa, b=bb, c=cc, return_hidden=True)
    # test_y = np.log(np.abs(test_y))

    particle_filter = ParticleFilterGamma(test_y,
                                          num_particles=N,
                                          a=0.5,
                                          b=0.8,
                                          c=c_start,
                                          k=kk,
                                          theta=thth,
                                          true_hidden=test_x,
                                          num_iterations=100,
                                          learn_rate=0.1/num_data,
                                          do_adaptive_learn=True
                                          )

    # particle_filter.filter_pass()
    # particle_filter.plot_filter_pass()
    particle_filter.calibrate_model()
    particle_filter.plot_params()
    # particle_filter.plot_filter_pass()
    particle_filter.plot_learn_rate()
