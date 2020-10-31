from scipy.optimize import fsolve
import numpy as np
import scipy.integrate as integrate
import scipy.stats as st
import math
import matplotlib.pyplot as plt

def logistic_function(x, k):
    return 1 / (1 + np.exp(- k[0] * (x - k[1])))

def gaussian_mixture(x):
    pi = [0.32, 0.53, 0.14]
    mu = [20.14, 29.35, 34.78]
    sigma = [3.60, 2.96, 1.32]
    pdf = np.sum([pi[i] * st.norm.pdf(x, loc=mu[i], scale=sigma[i]) for i in range(3)])
    return pdf

def integrand(x, k):
    return gaussian_mixture(x) * logistic_function(x, k)

def weighted_fnr_above_d_cens(k, d_cens):
    num = integrate.quad(lambda x: integrand(x, k), d_cens, np.inf)[0]
    denom = integrate.quad(lambda x: gaussian_mixture(x), d_cens, np.inf)[0]
    return num / denom

def solve(k):
    # k[0] = k
    # k[1] = x_0
    d_cens = 35.6
    eqn_1 = logistic_function(d_cens, k) - 0.05
    eqn_2 = weighted_fnr_above_d_cens(k, d_cens) - 0.8
    return [eqn_1, eqn_2]

root = fsolve(solve, [5, 35.6])

xx = np.arange(34, 37, 0.01)
yy = np.zeros(len(xx))
yy_old = np.zeros(len(xx))
for i in range(len(xx)):
    yy[i] = 1/(1+math.exp(-root[0]*(xx[i]-root[1])))
    if xx[i]<=35.6:
        yy_old[i] = 0;
    else:
        yy_old[i] = 0.8
        
l1 = plt.plot(xx, yy, label = 'New model', linestyle = '-', color = 'blue')
#l2 = plt.plot(xx, yy_old, label = 'Old model', linestyle = '-', color = 'red')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.xlabel('$C_t$ value')
plt.ylabel('False negative rate')
plt.title('False negative rate vs. $C_t$ value')
plt.show()


Gamma_new = np.zeros([101,11])
Gamma_new[1:, 1:] = Gamma
plt.matshow(Gamma_new, cmap=plt.get_cmap('OrRd'))
#plt.gca().set_aspect('equal', adjustable='box')
plt.yscale('log')
plt.xlim(1,10)
plt.ylim(1,100)
plt.xlabel('The number of infected in the pool d')
plt.ylabel('The logarithm of pool size N')
plt.title('False negative rate vs. N and d')






