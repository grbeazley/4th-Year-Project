import numpy as np
import matplotlib.pyplot as plt


def gen_univ_sto_vol(N, **kwargs):
    """
    Generates a univariate stochastic volatility model without leverage
    X = a * (X_prev - mu) + b * standard normal
    Y = c * exp(X/2) * standard normal
    """

    x_prev = np.random.randn()
    if 'a' in kwargs:
        a = kwargs['a']
    else:
        a = 0.99

    if 'b' in kwargs:
        b = kwargs['b']
    else:
        b = 1

    if 'c' in kwargs:
        c = kwargs['c']
    else:
        c = 1

    if 'mu' in kwargs:
        mu = kwargs['mu']
    else:
        mu = 0

    trajectory = np.zeros(N)

    for i in range(N):
        x = mu + a*(x_prev-mu) + b*np.random.randn()
        y = c * np.exp(x/2) * np.random.randn()
        x_prev = x
        trajectory[i] = y

    return trajectory


def gen_multi_sto_vol(N, m, **kwargs):
    # Generates a multivariate (m x N) stochastic volatility model using the type specified
    # No leverage used

    trajectory = np.zeros([m, N])
    zero_mean = np.zeros(m)

    if 'model_type' in kwargs:
        model_type = kwargs['model_type']
    else:
        model_type = 'basic'

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

            if type(var_latent) == int or float:
                # Been provided an integer so create diagonal matrix
                var_latent = np.diag(np.ones(m)*var_latent)
            elif type(var_latent) == np.ndarray:
                # Been provided a matrix
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

            if type(var_observed) == int or float:
                # Been provided an integer so create diagonal matrix
                var_observed = np.diag(np.ones(m)*var_observed)
            elif type(var_observed) == np.ndarray:
                # Been provided a matrix
                pass
            else:
                raise TypeError("Unexpected type passed for the var_observed")
        else:
            var_observed = np.diag(np.ones(m))

        h_prev = np.random.randn(m)

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
            trajectory[:, i] = y
            
            # Update previous time step
            h_prev = h

    return trajectory


if __name__ == "__main__":
    np.random.seed(0)
    num = 2443
    num_dims = 5
    # traj = gen_univ_sto_vol(num, a=0.99, mu=0, b=0.2, c=0.1)
    #plt.scatter(np.arange(len(traj)), traj, s=2)

    traj = gen_multi_sto_vol(num, num_dims, var_latent=0.2, var_observed=0.1)
    for j in range(num_dims):
        plt.figure()
        plt.scatter(np.arange(num), traj[j, :], s=2)
        plt.figure()
        plt.hist(traj[j, :], 50)
