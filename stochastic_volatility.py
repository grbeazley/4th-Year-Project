import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot_components, scatter, plot


def gen_univ_sto_vol(N, a=0.99, b=1.0, c=1.0, mu=0.0, return_hidden=False, **kwargs):
    """
    Generates a univariate stochastic volatility model without leverage
    X = a * (X_prev - mu) + sqrt(b) * standard normal
    Y = sqrt(c) * exp(X/2) * standard normal

    :param a: momentum term
    :param b: variance of hidden noise
    :param c: variance of observation noise
    :param mu: mean of hidden process
    :param return_hidden: bool to return hidden state
    :param N: int number of points to generate
    :param kwargs: initial state x0
    """
    if 'x0' in kwargs:
        x_prev = kwargs['x0']
    else:
        x_prev = np.sqrt(b / (1 - a**2)) * np.random.randn()

    trajectory_y = np.zeros(N+1)
    trajectory_x = np.zeros(N+1)
    trajectory_x[0] = x_prev
    trajectory_y[0] = np.sqrt(c) * np.exp(x_prev/2) * np.random.randn()

    for i in range(N):
        x = mu + a*(x_prev-mu) + np.sqrt(b)*np.random.randn()
        y = np.sqrt(c) * np.exp(x/2) * np.random.randn()
        x_prev = x
        trajectory_y[i + 1] = y
        trajectory_x[i + 1] = x

    if return_hidden:
        return trajectory_x, trajectory_y
    else:
        return trajectory_y


def hidden_to_observed(trajectory, c_var):
    # Converts hidden univariate trajectory to an observed process
    n = len(trajectory)
    trajectory_obs = np.zeros(n)
    for j in range(n):
        trajectory_obs[j] = np.sqrt(c_var) * np.exp(trajectory[j] / 2) * np.random.randn()

    return trajectory_obs


def gen_multi_sto_vol(N, m, **kwargs):
    # Generates a multivariate (m x N) stochastic volatility model using the type specified
    # No leverage used

    trajectory_y = np.zeros([m, N + 1])
    trajectory_h = np.zeros([m, N + 1])
    zero_mean = np.zeros(m)

    if 'model_type' in kwargs:
        model_type = kwargs['model_type']
    else:
        model_type = 'basic'

    if 'return_hidden' in kwargs:
        return_hidden = True
    else:
        return_hidden = False

    if model_type == "basic":
        # Create a basic model with optional latent correlation
        if 'phi' in kwargs:
            # Sets the latent relationships
            phi = kwargs['phi']
        else:
            phi = np.diag(np.ones(m)*0.95)
        if 'var_latent' in kwargs:
            # Set the latent noise variance matrix
            var_latent = kwargs['var_latent']

            if (type(var_latent) == int) or (type(var_latent) == float):
                # Been provided a single value so create diagonal matrix
                var_latent = np.diag(np.ones(m)*var_latent)
            elif type(var_latent) == np.ndarray:
                # Been provided a matrix, assume square
                dims = var_latent.shape
                assert dims[0] == dims[1]
                pass
            else:
                raise TypeError("Unexpected type passed for the var_latent")
        else:
            var_latent = np.diag(0.5*np.ones(m))
        if 'mu' in kwargs:
            # The latent mu
            mu = kwargs['mu']
        else:
            mu = zero_mean
        if 'var_observed' in kwargs:
            # Set the observed process correlation matrix
            var_observed = kwargs['var_observed']

            if (type(var_observed) == int) or (type(var_observed) == float):
                # Been provided a single value so create diagonal matrix
                var_observed = np.diag(np.ones(m)*var_observed)
            elif type(var_observed) == np.ndarray:
                # Been provided a matrix
                pass
            else:
                raise TypeError("Unexpected type passed for the var_observed")
        else:
            var_observed = np.diag(np.ones(m))

        # Calculate initial values
        h_prev = np.random.randn(m)
        trajectory_h[:, 0] = h_prev
        # Calculate observation variance and noise
        obs_var = np.diag(np.exp(h_prev / 2))
        obs_noise = np.random.multivariate_normal(zero_mean, var_observed)

        # Update observed state trajectory
        trajectory_y[:, 0] = np.dot(obs_var, obs_noise)

        for i in range(N):
            # Create latent noise vector (m x 1)
            latent_noise = np.random.multivariate_normal(zero_mean, var_latent)

            # Update latent variables
            h = mu + np.dot(phi, h_prev - mu) + latent_noise

            # Calculate observation variance and noise
            obs_var = np.diag(np.exp(h/2))
            obs_noise = np.random.multivariate_normal(zero_mean, var_observed)

            # Update observed state and trajectory
            y = np.dot(obs_var, obs_noise)
            trajectory_y[:, i + 1] = y
            trajectory_h[:, i + 1] = h
            
            # Update previous time step
            h_prev = h

    if return_hidden:
        return trajectory_h, trajectory_y
    else:
        return


def gen_univ_mrkv(N, a=0.99, b=1, c=1.23, d=0.0, mu=0.0, return_hidden=False, **kwargs):
    """
    Generates a univariate markov model
    X = a * (X_prev - mu) + sqrt(b) * standard normal
    Y = X + d + sqrt(c) * standard normal

    :param a: momentum term
    :param b: variance of hidden noise
    :param c: variance of observation noise
    :param d: mean for observation process
    :param mu: mean for hidden process
    :param return_hidden: bool to return hidden state
    :param N: int number of points to generate
    :param kwargs: x0
    """

    if 'x0' in kwargs:
        x_prev = kwargs['x0']
    else:
        x_prev = np.sqrt(b / (1 - a**2)) * np.random.randn()

    trajectory_y = np.zeros(N+1)
    trajectory_x = np.zeros(N+1)
    trajectory_x[0] = x_prev
    trajectory_y[0] = x_prev + d + np.sqrt(c) * np.random.randn()

    for i in range(N):
        x = mu + a*(x_prev-mu) + np.sqrt(b)*np.random.randn()
        y = x + d + np.sqrt(c) * np.random.randn()
        x_prev = x
        trajectory_y[i + 1] = y
        trajectory_x[i + 1] = x

    if return_hidden:
        return trajectory_x, trajectory_y
    else:
        return trajectory_y


def predict_univ_sto_vol(N, num_points, a=0.95, b=1.0, c=1.0, mu=0.0, return_hidden=False, **kwargs):
    # Allows the generation of lots of possible processes from the same initial conditions

    if 'x0' in kwargs:
        x_prev = kwargs['x0']
    else:
        x_prev = np.sqrt(b / (1 - a**2)) * np.random.randn()

    trajectory_y = np.zeros([N, num_points + 1])
    trajectory_x = np.zeros([N, num_points + 1])
    trajectory_x[:, 0] = x_prev
    trajectory_y[:, 0] = np.exp(x_prev/2) * np.sqrt(c) * np.random.randn(N)

    for i in range(num_points):
        x = mu + a * (x_prev - mu) + np.sqrt(b) * np.random.randn(N)
        y = np.exp(x/2) * np.sqrt(c) * np.random.randn(N)
        x_prev = x
        trajectory_y[:, i + 1] = y
        trajectory_x[:, i + 1] = x

    if return_hidden:
        return trajectory_x, trajectory_y
    else:
        return trajectory_y


def gen_univ_gamma(N, a=0.99, b=1.0, c=1.0, k=1.0, theta=1.0, mu=0.0, return_hidden=False, **kwargs):
    """
    Generates a univariate model with gamma distributed noise
    X = a * (X_prev - mu) + sqrt(b) * standard normal
    Y = sqrt(c) * exp(X/2) * gamma(k, theta) = gamma(k, theta*exp(X/2))

    :param a: momentum term
    :param b: variance of hidden noise
    :param c: scale factor in observed process
    :param k: shape parameter for observation process gamma
    :param theta: rate parameter for observation process gamma
    :param mu: mean for hidden process
    :param return_hidden: bool to return hidden state
    :param N: int number of points to generate
    :param kwargs: x0
    """

    if 'x0' in kwargs:
        x_prev = kwargs['x0']
    else:
        x_prev = np.sqrt(b / (1 - a**2)) * np.random.randn()

    trajectory_y = np.zeros(N+1)
    trajectory_x = np.zeros(N+1)
    trajectory_x[0] = x_prev
    trajectory_y[0] = np.random.gamma(k, np.exp(x_prev/2)*theta, 1)

    for i in range(N):
        x = mu + a*(x_prev-mu) + np.sqrt(b)*np.random.randn()
        y = np.random.gamma(k, theta, 1) * np.exp(x/2) * np.sqrt(c)
        x_prev = x
        trajectory_y[i + 1] = y
        trajectory_x[i + 1] = x

    if return_hidden:
        return trajectory_x, trajectory_y
    else:
        return trajectory_y


if __name__ == "__main__":
    np.random.seed(25)
    num = 4049
    num_dims = 2
    # traj_y, traj_h = gen_univ_sto_vol(num, a=0.995, mu=0, b=0.023, c=1, return_hidden=True)
    # scatter(traj_y)
    # traj_h, traj_y = gen_univ_sto_vol(num, a=0.90, mu=0, b=1, c=0.01, return_hidden=True)
    aa = 0.5
    bb = 0.5
    cc = 1
    dd = 0

    thth = 1.03
    kk = 0.72

    # traj_h, traj_y = gen_univ_mrkv(num, a=aa, mu=0, b=bb, c=cc, d=dd, return_hidden=True)
    traj_h, traj_y = gen_univ_gamma(num, a=aa, mu=0, b=bb, theta=thth, k=kk, return_hidden=True)
    # traj_h, traj_y = predict_univ_mrkv(N=20, num_points=num, a=aa, mu=0, b=bb, return_hidden=True, x0=0.5)
    # plt.figure()
    # plt.figure()
    # plt.scatter(np.arange(num+1), traj, s=2)
    # traj2 = traj + 0.1*np.random.randn(num)
    # plt.scatter(np.arange(len(traj)), traj, s=2)

    diag_val = 0.8
    off_diag = 0.04
    phi = np.array([[diag_val, off_diag, off_diag, off_diag, off_diag],
                    [off_diag, diag_val, off_diag, off_diag, off_diag],
                    [off_diag, off_diag, diag_val, off_diag, off_diag],
                    [off_diag, off_diag, off_diag, diag_val, off_diag],
                    [off_diag, off_diag, off_diag, off_diag, diag_val]])
    diag_val = 0.2
    off_diag = 0.01
    #
    sigma_eta = np.array([[diag_val, off_diag, off_diag, off_diag, off_diag],
                          [off_diag, diag_val, off_diag, off_diag, off_diag],
                          [off_diag, off_diag, diag_val, off_diag, off_diag],
                          [off_diag, off_diag, off_diag, diag_val, off_diag],
                          [off_diag, off_diag, off_diag, off_diag, diag_val]])

    phi = np.array([[0.495, 0.495],
                    [0.495, 0.495]])

    # traj_h, traj_y = gen_multi_sto_vol(num, num_dims, phi=phi, var_latent=1, var_observed=0.1, return_hidden=True)

    # plot_components(traj_h)
    # plot_components(traj_y)

    # scatter(traj_h)
    # scatter(traj_y)
    # scatter(np.log(np.abs(traj_y)))
    # scatter(np.log(np.abs(traj_y)))
    # plt.plot(traj_h)
    # plt.plot(traj_y)