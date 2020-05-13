import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from stochastic_volatility import gen_univ_gamma
from pdfs import gamma_pdf
from plot_utils import *
# from particle_filter import ParticleFilter
from particle_filter_backtrace import ParticleFilterBackTrace


class ParticleFilterGamma(ParticleFilterBackTrace):

    def __init__(self, true_obs, num_particles=20, num_iterations=1, a=0.99,
                 b=1.0, c=1.0, correction=1.0, k=1.0, theta=1.0, learn_rate=0.0001, learn_a=1,
                 learn_b=15, learn_c=20, multi=False, **kwargs):

        ParticleFilterBackTrace.__init__(self, true_obs, num_particles, num_iterations, a,
                                b, c, learn_rate, **kwargs)

        self.k = k
        self.theta = theta
        self.correction = correction
        self.learn_a = learn_a
        self.learn_b = learn_b
        self.learn_c = learn_c

        # Whether to use multivariate initial sampling scheme
        self.multi = multi

    def observation(self, x, y):
        # Probability of observing y given x
        theta = np.exp((x*self.correction)/2) * self.theta * np.sqrt(self.c)
        obs = gamma_pdf(y, self.k, theta)
        return obs

    def _get_initial_sample(self):
        if self.multi:
            return np.random.randn(self.num_particles) * self.correction
        else:
            return np.sqrt(self.b / (1 - self.a ** 2)) * np.random.randn(self.num_particles)

    @staticmethod
    def _get_param_symbols():
        return ['$\\phi$', '$\\sigma^2$', '$\\beta^2$']

    def update_learn_rates(self, new_rates):
        self.learn_a = new_rates[0]
        self.learn_b = new_rates[1]
        self.learn_c = new_rates[2]

    def _comp_param_update(self, final_weights_norm):
        # Compute the gradient of the log likelihood w.r.t. a
        sqrd_minus_one_prdct = self.particle_history[:, :-1] * self.particle_history[:, 1:]
        summand_a = (sqrd_minus_one_prdct - (np.square(self.particle_history[:, :-1]) * self.a)) / self.b
        dl_da = np.dot(np.sum(summand_a, axis=1), final_weights_norm)

        # Update parameter, ensuring it retains stationarity
        self.a = min(self.a + self.adap_learn_rate * dl_da * self.learn_a, 0.9999)

        # Compute the gradient of the log likelihood w.r.t. b
        sqrd_prdct = np.square(self.particle_history[:, 1:] - self.a * self.particle_history[:, :-1])
        sum_b = np.sum(sqrd_prdct - self.b, axis=1) / 2 * (self.b ** 2)
        dl_db = np.dot(sum_b, final_weights_norm)

        # Update parameter b
        self.b = max(0.0001, self.b + self.adap_learn_rate * dl_db * self.learn_b)

        th_exp_term = np.exp(self.particle_history / 2) * self.theta
        summand_c = self.true_obs / (2 * th_exp_term * self.c ** (3 / 2))
        sum_c = np.sum(summand_c - (self.k / (2 * self.c)), axis=1)
        dl_dc = np.dot(sum_c, final_weights_norm)

        # Update parameter c
        self.c = max(0.0001, self.c + self.adap_learn_rate * dl_dc * self.learn_c)

        # Return update to parameters
        return [self.a, self.b, self.c]


if __name__ == "__main__":
    np.random.seed(0)

    aa = 0.95
    bb = 0.2

    c_true = 0.7
    c_start = 1

    kk = 1.3
    thth = 0.5
    
    M = 2000

    N = 750

    test_x, test_y = gen_univ_gamma(M, a=aa, b=bb, c=c_true, k=kk, theta=thth, return_hidden=True)
    # test_x, test_y = gen_univ_sto_vol(num_data, a=aa, b=bb, c=cc, return_hidden=True)
    # test_y = np.log(np.abs(test_y))

    particle_filter = ParticleFilterGamma(test_y,
                                          num_particles=N,
                                          a=0.5,
                                          b=0.5,
                                          c=c_start,
                                          k=kk,
                                          theta=thth,
                                          true_hidden=test_x,
                                          num_iterations=5,
                                          learn_rate=0.01/M,
                                          do_adaptive_learn=True
                                          )

    # particle_filter.filter_pass()
    # particle_filter.plot_filter_pass()
    particle_filter.calibrate_model()
    # particle_filter.plot_params()
    # particle_filter.plot_filter_pass()
    # particle_filter.plot_learn_rate()

    # part_hist3 = particle_filter.particle_history3
    # part_hist = particle_filter.particle_history
    # index_history = particle_filter.index_history
    # plot(part_hist.T)



    # plot(part_hist3.T)


    # plt.plot(main_particle_history)

    # plot(particle_history.T)





