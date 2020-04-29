import numpy as np
from tqdm import tqdm
from particle_filter import ParticleFilter
from matplotlib import pyplot as plt
from stochastic_volatility import gen_univ_sto_vol
from utilities import load_data


class ParticleFilterStandard(ParticleFilter):

    def observation(self, x, y):
        sigma = np.sqrt(self.c) * np.exp(x / 2)
        log_obs = -np.log(sigma) - (y ** 2 / (2 * sigma ** 2))
        return np.exp(log_obs)

    @staticmethod
    def _get_param_symbols():
        return ['$\\phi$', '$\\sigma^2$', '$\\beta$']

    def _comp_param_update(self, final_weights_norm):
        # Compute the gradient of the log likelihood w.r.t. a
        sqrd_minus_one_prdct = self.particle_history[:, :-1] * self.particle_history[:, 1:]
        summand_a = (sqrd_minus_one_prdct - (np.square(self.particle_history[:, :-1]) * self.a)) / self.b
        dl_da = np.dot(np.sum(summand_a, axis=1), final_weights_norm)

        # Update parameter, ensuring it retains stationarity
        self.a = min(self.a + self.learn_rate * dl_da, 0.9999)

        # Compute the gradient of the log likelihood w.r.t. b
        sqrd_prdct = np.square(self.particle_history[:, 1:] - self.a * self.particle_history[:, :-1])
        sum_b = np.sum(sqrd_prdct - self.b, axis=1) / 2 * self.b ** 2
        dl_db = np.dot(sum_b, final_weights_norm)

        # Update parameter b
        self.b = max(0.0001, self.b + self.learn_rate * dl_db * 2.5)

        # Compute the gradient of the log likelihood w.r.t. c
        summand_c = np.square(self.true_obs) / (2 * self.c * np.exp(self.particle_history))
        sum_c = np.sum(summand_c - self.c / 2, axis=1)
        dl_dc = np.dot(sum_c, final_weights_norm)

        # Update parameter c
        self.c = max(0.0001, self.c + self.learn_rate * dl_dc)

        # Store update to parameters
        return [self.a, self.b, self.c]


if __name__ == "__main__":
    np.random.seed(0)

    aa = 0.9
    bb = 1
    cc = 1

    num_data = 1000
    N = 500

    # stem = "Data Sets\\FTSEICA_sto_vol\\"
    #
    # names = {"Oil.csv": ['Adj Close']}
    # data_frame = load_data(stem, names)
    # Take only series values from the data frame
    # data = data_frame.values[1:, :].astype('float')
    # Take difference
    # data_returns = (np.log(data[:, 1:]) - np.log(data[:, :-1]))[0]

    test_x, test_y = gen_univ_sto_vol(num_data, a=aa, b=bb, c=cc, return_hidden=True)

    test_y = np.abs(test_y)

    particle_filter = StandardParticleFilter(test_y,
                                             num_particles=N,
                                             a=0.9,
                                             b=0.1,
                                             c=0.5,
                                             true_hidden=test_x,
                                             num_iterations=50,
                                             learn_rate=0.0001)

    particle_filter.filter_pass()
    particle_filter.plot_filter_pass()
    particle_filter.calibrate_model()
    particle_filter.plot_params()
    particle_filter.plot_filter_pass()