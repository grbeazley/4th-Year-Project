import numpy as np
import matplotlib.pyplot as plt


N = 2443
np.random.seed(0)
Xk_1 = np.random.randn()
a = 0.97
b = 0.3
c = 0.04
traj = []

for i in range(N):
    Xk = a*Xk_1 + b*np.random.randn()
    Yk = c * np.exp(Xk/2) * np.random.randn()
    Xk_1 = Xk
    traj.append(Yk)

plt.scatter(np.arange(len(traj)), traj, s=2)
