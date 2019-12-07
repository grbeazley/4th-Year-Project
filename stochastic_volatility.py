import numpy as np
import matplotlib.pyplot as plt



def gen_univ_sto_vol(N,params):

    x_prev = np.random.randn()
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    trajectory = np.zeros(N)

    for i in range(N):
        x = d + a*x_prev + b*np.random.randn()
        y = c * np.exp(x/2) * np.random.randn()
        x_prev = x
        trajectory[i] = y

    return trajectory


if __name__ == "__main__":
    np.random.seed(0)
    N = 2443
    params = [0.99, 0.2, 0.03, 0]
    traj = gen_univ_sto_vol(N, params)
    plt.scatter(np.arange(len(traj)), traj, s=2)
