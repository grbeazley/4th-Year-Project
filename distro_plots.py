from plot_utils import *
from matplotlib import pyplot as plt
import numpy as np
from utilities import power_folded_norm

np.random.seed(0)

N = 20

x1 = np.random.randn(N, 100000)
# x2 = np.random.randn(100000)
e1 = np.abs(x1)**(1/N)
# e2 = np.abs(x2)**0.5
m1 = np.prod(e1, axis=0)
q = np.linspace(0, 3, 1000)

hist_norm(m1)

alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
for alpha in alphas:
    plt.plot(q, power_folded_norm(q, alpha=alpha))

plt.legend(alphas)