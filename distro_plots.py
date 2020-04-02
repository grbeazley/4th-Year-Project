from plot_utils import *
from matplotlib import pyplot as plt
import numpy as np
from pdfs import *
from levy import fit_levy
from scipy.stats import levy_stable, gamma
from scipy.special import gamma as gamma_function
from utilities import is_normal

import scipy.integrate as integrate

np.random.seed(0)

num_series = 10
N = 100000

x1 = np.random.randn(num_series, N)
powers = np.array([0.3, 0.3, -0.1, -0.9, 0.7])
# e1 = np.abs(x1).T**powers
# e1 = (np.abs(x1).T)**(0.49)
e1 = np.log(np.abs(x1))

# for i in range(num_series):
#     # hist_norm(e1[:, i])
#     a = powers[i]
#     # print(2**(a/2) * gamma_function((a+1)/2) / np.sqrt(2*np.pi))
#     print(2*np.abs(a)/(np.abs(a) * np.sqrt(2*np.pi)))
#     print(np.mean(e1[i]))
#     print("-------------------")

# Na = 50
# qr = np.linspace(-0.2,1,Na)
# z = np.zeros(Na)
# zg = np.zeros(Na)
# zq = np.zeros(Na)
# zp = np.zeros(Na)
#
#
# for i, a in enumerate(qr):
#     # z[i] = np.mean(np.abs(x1[0, :])**a)
#     # zg[i] = mean_power_folded_norm(a)
#     mean = mean_power_folded_norm(a)
#     var = variance_power_folded_norm(a) - mean_power_folded_norm(0.5) ** 2
#     zq[i] = mean * np.log((var + mean**2)/mean)
#
#     zp[i] = integrate.quad(xlnx_func, 0, np.inf, args=(a,))[0]
#
#
# plt.figure()
# # plt.plot(qr, z)
# # plt.plot(qr, zg)
# plt.plot(qr, zq)
# plt.plot(qr, zp)

# Q = np.array([[-0.53757991, -0.15268258, -0.24307302, -0.25496737,  0.46995185,
#         -0.40650142, -0.01397579,  0.37495092,  0.04973428,  0.24378748],
#        [ 0.01216122,  0.42128857,  0.56273381,  0.04663472,  0.33160816,
#         -0.18575038,  0.1388579 ,  0.21663408, -0.55033361,  0.10052241],
#        [ 0.37111567,  0.38275723, -0.18056109,  0.44937857, -0.06066083,
#         -0.37930306, -0.17358118,  0.35595244,  0.45610893, -0.05413121],
#        [ 0.2356257 , -0.17054269,  0.08649874, -0.09281263,  0.06700229,
#         -0.48518374,  0.57874914, -0.43426127,  0.16920323,  0.33908036],
#        [-0.43248141, -0.09400834,  0.40526072,  0.19980868, -0.54298047,
#         -0.02480198,  0.38004611,  0.30590237,  0.26682728, -0.10632794],
#        [ 0.2741275 , -0.15980365, -0.17136684,  0.00698723,  0.41807544,
#          0.41444405,  0.59955601,  0.40281087,  0.06814309, -0.11093487],
#        [-0.38849414,  0.40333427, -0.11338715,  0.2103842 ,  0.35514282,
#          0.03975577,  0.23264131, -0.37496976,  0.20769166, -0.52339207],
#        [-0.14035469, -0.47175271,  0.36579213,  0.53039729,  0.32302198,
#          0.26506795, -0.23680558, -0.21638804,  0.20794688,  0.24423601],
#        [-0.23725666,  0.48812743, -0.07563428,  0.06983163, -0.1588639 ,
#          0.454918  ,  0.11282634, -0.10914857,  0.17888946,  0.65554982],
#        [-0.12253026, -0.12545013, -0.42234213,  0.65504456, -0.16029794,
#         -0.09150524,  0.19845827, -0.03488978, -0.53717206,  0.08381032]])

Q = np.array([0.28791489,  0.62549222,  0.09314578,  0.61501408, -0.28518374,
              0.12874914, -0.23426127,  0.16920323,  0.21908036, 0.15890213])

# Q = np.array([0.9291489,  0.1249222,  0.9314578,  0.01501408, -0.28518374])

# Q = np.array([0.28791489,  0.62549222,  0.09314578,  0.61501408, -0.28518374])
#
# m1 = np.prod(e1, axis=1)
#
# mus = np.zeros(num_series)
# var_s = np.zeros(num_series)
# # for i in range(num_series):
m1 = np.dot(Q, e1)
# m1 = e1[i, :]
# mu = np.mean(m1)
# var = np.var(m1)
# mus[i] = mu
# var_s[i] = var
# q = np.linspace(0, 4, 1000)
# q = np.linspace(-10, 5, 100)
# pdf_q = alpha_stable_pdf(-q, alpha=0.5, beta=1, mu=4.5, c=0.8)

# hist_norm(m1)
# plt.plot(q, pdf_q)


# # Alpha Stable
# params, nelog = fit_levy(m1)
#
# print(params)
#
# pdf_q = alpha_stable_pdf(q, alpha=params.x[0], beta=params.x[1], mu=params.x[2], c=params.x[3])
#
# hist_norm(m1)
# plt.plot(q, pdf_q)
#
# a1 = (levy_stable.rvs(params.x[0], params.x[1], size=100000) * params.x[3]) + params.x[2]
#
# qq_plot(a1, m1)


# # Normal Distribution
# mun = np.mean(m1)
# stdn = np.std(m1)
# n1 = (np.random.randn(100000) + mun) * stdn
#
# hist_norm(m1)
# pdf_n_q = normal_pdf(q, mun, stdn)
# plt.plot(q, pdf_n_q)
#
# qq_plot(n1, m1)


# Gamma Distribution
x = np.sum(np.exp(m1))
y = np.sum(np.exp(m1)*m1)
z = np.sum(m1) * np.sum(np.exp(m1))
N = 100000
k = (N*x) / ((N*y) - z)
theta = (N*y - z)/N**2

mean = 1
var = 1

means = np.zeros(num_series)
xlnxs = np.zeros(num_series)

for i, a in enumerate(Q):
    means[i] = mean_power_folded_norm(a)
    var *= _variance_power_folded_norm(a)

mean = np.prod(means)
var_sub = var - mean**2

theta_star = var_sub/mean
k_star = mean**2 / var_sub

k_star_hat, theta_star_hat = comp_k_theta_from_alphas(Q)

print(k_star, theta_star)
print("------------------")
print(k_star_hat, theta_star_hat)
print("------------------")
print(k, theta)
r = np.linspace(0.001, 5, 1000)

hist_norm(np.exp(m1), bins=1000)
plt.plot(r, gamma_pdf(r, k=k, theta=theta))
g1 = np.random.gamma(k, theta, 100000*num_series)
qq_plot(g1, np.exp(m1))


hist_norm(np.exp(m1), bins=1000)
plt.plot(r, gamma_pdf(r, k=k_star_hat, theta=theta_star_hat))
g1 = np.random.gamma(k_star_hat, theta_star_hat, 100000*num_series)
qq_plot(g1, np.exp(m1))

# k=3
# theta = 0.5
# test = 0.5
# g1 = np.random.gamma(k, theta, 100000) * test
#
# hist_norm(g1, bins=1000)
# plt.plot(r, gamma_pdf(r, k=k, theta=test*theta))




