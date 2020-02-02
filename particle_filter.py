import numpy as np
from matplotlib import pyplot as plt
from stochastic_volatility import gen_univ_sto_vol


aa = 0.95
bb = 0.5
cc = 0.5


def hidden_sample(x, a=aa, b=bb):
    noise = np.random.randn(len(x))
    return x*a + np.sqrt(b)*noise


def observation(x, y, c=cc):
    sigma = np.sqrt(c) * np.exp(x/2)
    log_obs = -np.log(sigma) - y**2/(2*sigma**2)
    return np.exp(log_obs)


np.random.seed(0)

num_data = 200
num_particles = 20

test_x, test_y = gen_univ_sto_vol(num_data, a=aa, b=bb, c=cc, return_hidden=True)


class ParticleFilter:

    def __init__(self, true_data, num_particles=20, num_iterations=10,**kwargs):

        self.num_data = len(true_data) - 1
        self.num_particles = num_particles
        self.num_iterations = num_iterations

        self.true_data = true_data
        self.process_history = np.zeros([self.num_particles, num_data + 1])
        self.params_history = np.zeros([3, self.num_iterations])


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

    def hidden_sample(self, x):
        noise = np.random.randn(len(x))
        return x * self.a + np.sqrt(self.b) * noise

    def observation(self, x, y):
        sigma = np.sqrt(self.c) * np.exp(x / 2)
        log_obs = -np.log(sigma) - y ** 2 / (2 * sigma ** 2)
        return np.exp(log_obs)

    def initialise(self):

        initial_sample = np.random.randn(num_particles)

        initial_weights = observation(initial_sample, test_y[0])
        initial_weights = initial_weights / np.sum(initial_weights)


        self.process_history[:, 0] = initial_sample


    weights = initial_weights
    weights_history = np.zeros([num_particles, num_data + 1])
    weights_history[:, 0] = initial_weights

    particle_range = np.arange(num_particles)

    Wn = np.sum(initial_weights)

    for i in range(num_data):

        # for j in range(num_particles):

        particle_indexes = np.random.choice(particle_range, size=num_particles, p=weights)
        Xn = particle_history[particle_indexes, i]

        # particle_history[j, :] = particle_history[particle_index, :]

        Xn_plus_1 = hidden_sample(Xn)

        particle_history[:, :i] = particle_history[particle_indexes, :i]

        particle_history[:, i + 1] = Xn_plus_1

        # Use qn as fn
        new_particle_weights = observation(Xn_plus_1, test_y[i+1])

        weights_history[:, i + 1] = new_particle_weights

        Wn = np.sum(new_particle_weights)
        print(Wn)

        weights = weights_history[:, i + 1] / Wn

plt.plot(test_x.T)
plt.plot(particle_history.T, '--', linewidth=1)
plt.legend(['True Hidden State'])