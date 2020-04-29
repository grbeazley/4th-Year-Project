import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from stochastic_volatility import gen_univ_mrkv, gen_univ_sto_vol
from particle_filter import ParticleFilter


class ParticleFilterApproxNormal(ParticleFilter):

    def __init__(self, true_obs, num_particles=20, num_iterations=1, a=0.99,
                 b=1.0, c=1.23, d=0.0, learn_rate=0.0001, **kwargs):

        ParticleFilter.__init__(self, true_obs, num_particles, num_iterations, a,
                                b, c, learn_rate, **kwargs)

        self.d = d

    def observation(self, x, y):
        # Probability of observing y given x
        sigma = np.sqrt(self.c)
        obs = np.exp(-(y - (x + self.d))**2 / (2 * sigma**2)) / sigma
        return obs

    @staticmethod
    def _get_param_symbols():
        return ['$\\phi$', '$\\sigma^2$', 'd']

    def _comp_param_update(self, final_weights_norm):
        # Compute the gradient of the log likelihood w.r.t. a
        sqrd_minus_one_prdct = self.particle_history[:, :-1] * self.particle_history[:, 1:]
        summand_a = (sqrd_minus_one_prdct - (np.square(self.particle_history[:, :-1]) * self.a)) / self.b
        dl_da = np.dot(np.sum(summand_a, axis=1), final_weights_norm)

        # Update parameter, ensuring it retains stationarity
        self.a = min(self.a + self.adap_learn_rate*dl_da, 0.9999)

        # Compute the gradient of the log likelihood w.r.t. b
        sqrd_prdct = np.square(self.particle_history[:, 1:] - self.a * self.particle_history[:, :-1])
        sum_b = np.sum(sqrd_prdct - self.b, axis=1) / 2*(self.b**2)
        dl_db = np.dot(sum_b, final_weights_norm)

        # Update parameter b
        self.b = max(0.0001, self.b + self.adap_learn_rate * dl_db * 2.5)

        summand_d = self.true_obs - (self.particle_history + self.d)
        sum_d = np.sum(summand_d / self.c, axis=1)
        dl_dd = np.dot(sum_d, final_weights_norm)

        # Update parameter c
        self.d = self.d + self.adap_learn_rate * dl_dd

        # Store update to parameters
        return [self.a, self.b, self.d]


if __name__ == "__main__":
    np.random.seed(0)

    aa = 0.95
    bb = 0.5
    cc = 1.23
    dd = -0.6

    num_data = 200
    N = 200

    test_x, test_y = gen_univ_mrkv(num_data, a=aa, b=bb, c=cc, d=dd, return_hidden=True)
    # test_x, test_y = gen_univ_sto_vol(num_data, a=aa, b=bb, c=cc, return_hidden=True)
    # test_y = np.log(np.abs(test_y))

    particle_filter = ParticleFilter(test_y,
                                     num_particles=N,
                                     a=0.5,
                                     b=1,
                                     c=1.23,
                                     d=0,
                                     true_hidden=test_x,
                                     num_iterations=500,
                                     learn_rate=0.001,
                                     do_adaptive_learn=True)

    particle_filter.filter_pass()
    particle_filter.plot_filter_pass()
    particle_filter.calibrate_model()
    particle_filter.plot_params()
    particle_filter.plot_filter_pass()
    particle_filter.plot_learn_rate()
