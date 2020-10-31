#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 15:36:31 2020

@author: ryx
"""


import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm
from scipy.stats import binom
import random
import time
import math


pi = [0.33, 0.54, 0.13];
mu = [20.13, 29.41, 34.81];
sigma = [3.60, 3.02, 1.31];
FACT = np.math.factorial
PWR = np.power
np_pmf = lambda d, n, p0: FACT(n) / (FACT(d) * FACT(n - d)) * PWR(p0, d) * PWR(1 - p0, n - d)
gamma_matrix = np.load('F_N_d_round.npy')
eta_matrix = np.load('Eta_N_d_round.npy')
psi_matrix = np.load('Psi_N_d_round.npy')
theta_matrix = np.load('Theta_N_d_round.npy')

#false positive
def gamma(n,d,q=0.001):
    if d == 0:
        return 1 - q
    return gamma_matrix[n-1,d-1]

def eta(n,d):
    return eta_matrix[n-1,d-1]

def psi(n,d1,d2):
    if d1 > d2:
        return psi_matrix[n-1,d1-1,d2-1]
    return psi_matrix[n-1,d2-1,d1-1]

def theta(n,d1,d2):
    if d1 > d2:
        return theta_matrix[n-1,d1-1,d2-1]
    return theta_matrix[n-1,d2-1,d1-1]

def EmL(n,p0):
    return n*sum([(1-gamma(n,d)) * np_pmf(d, n, p0) for d in range(0,min(n+1,10))])

def EfLG(n,p0):
    fn =  sum([d*gamma(n,d) * np_pmf(d, n, p0) for d in range(1,min(n+1,10))])
    return fn

def EfLI(n,p0):
    fn =  sum([d*eta(n,d) * np_pmf(d, n, p0) for d in range(1,min(n+1,10))])
    return fn

def EML(N,n,p0):
    div = N // n
    rem = N % n
    return math.ceil(N / n) + EmL(n, p0)*div + EmL(rem, p0)

def EFL(N,n,p0):
    div = N // n
    rem = N % n
    return (EfLI(n,p0) + EfLG(n,p0))*div + EfLI(rem,p0) + EfLG(rem,p0)

def EmS(n,p0):
    part1 = n**2 * p0 * sum([sum([theta(n,d1,d2) * np_pmf(d1-1, n-1, p0) * np_pmf(d2-1, n-1, p0) for d1 in range(1,min(n+1,10))]) for d2 in range(1,min(n+1,10))])
    part2 = n**2 * (1 - p0) * (sum([(1-gamma(n,d)) * np_pmf(d, n-1, p0) for d in range(0,min(n,10))]))**2
    return part1+part2

def EfSG(n,p0):
    part1 = n**2 * p0 * (1-sum([sum([theta(n,d1,d2) * np_pmf(d1-1, n-1, p0) * np_pmf(d2-1, n-1, p0) for d1 in range(1,min(n+1,10))]) for d2 in range(1,min(n+1,10))]))
    return part1

def EfSI(n,p0):
    part1 = n**2 * p0 * sum([sum([psi(n,d1,d2) * np_pmf(d1-1, n-1, p0) * np_pmf(d2-1, n-1, p0) for d1 in range(1,min(n+1,10))]) for d2 in range(1,min(n+1,10))])
    return part1

def EMS(N,n,p0):
    div = N // (n**2)
    rem = N % (n**2)
    m = EmS(n,p0)
    return (m+ 2*n) * div + rem

def EFS(N,n,p0):
    div = N // (n**2)
    rem = N % (n**2)
    part1 = EfSG(n,p0) + EfSI(n,p0)
    return div * part1 + p0 * rem * gamma(1,1)


def EMLp(n,p0):
    return 1/n + sum([(1-gamma(n,d)) * np_pmf(d, n, p0) for d in range(0,min(n+1,10))])

def EFLp(n,p0):
    fn =  sum([d*(eta(n,d)+gamma(n,d)) * np_pmf(d, n, p0) for d in range(1,min(n+1,10))])
    return fn/n

def EMSp(n,p0):
    part1 = p0 * sum([sum([theta(n,d1,d2) * np_pmf(d1-1, n-1, p0) * np_pmf(d2-1, n-1, p0) for d1 in range(1,min(n+1,10))]) for d2 in range(1,min(n+1,10))])
    part2 = (1 - p0) * (sum([(1-gamma(n,d)) * np_pmf(d, n-1, p0) for d in range(0,min(n,10))]))**2
    return 2/n + part1+part2

def EFSp(n,p0):
    part1 = p0 * (1-sum([sum([(theta(n,d1,d2)-psi(n,d1,d2)) * np_pmf(d1-1, n-1, p0) * np_pmf(d2-1, n-1, p0) for d1 in range(1,min(n+1,10))]) for d2 in range(1,min(n+1,10))]))
    return part1

# p0 = input('p0:')
# N = input('N:')
# C = input('enter daily test number C:')

# p0 = 0.001
# N = 10000
# C = 300

def EFPLp(n,p0):
    fp =  (1-gamma(n,0))/n * sum([(n-d)*(1-gamma(n,d)) * np_pmf(d, n, p0) for d in range(0,min(n+1,10))])
    return fp

def EFPSp(n,p0):
    fp = (1-gamma(n,0)) * (1 - p0) * (sum([(1-gamma(n,d)) * np_pmf(d, n-1, p0) for d in range(0,min(n,10))]))**2
    return fp

def square_array_optimal_n_formulation_1(N, p0, C):
    max_n = int(N**0.5)
    n_optimal = float('inf')
    M_optimal = float('inf')
    F_optimal = float('inf')
    for n in range(2,max_n+1):
        div = N // (n**2)
        rem = N % (n**2)
        if 2 * n * div + rem < C:
            M = EMS(N,n,p0)
            if M <= C:
                F = EFS(N,n,p0)
                if F < F_optimal:
                        n_optimal = n
                        M_optimal = M
                        F_optimal = F
                
    return (n_optimal, M_optimal, F_optimal)

def square_array_optimal_n_formulation_2(N, p0, C):
    c = C/N
    max_n = int(N**0.5)
    n_optimal = float('inf')
    Mp_optimal = float('inf')
    Fp_optimal = float('inf')
    for n in range(2,max_n+1):
        if 2 /n < c:
            Mp = EMSp(n,p0)
            if Mp <= c:
                Fp = EFSp(n,p0)
                if Fp < Fp_optimal:
                        n_optimal = n
                        Mp_optimal = Mp
                        Fp_optimal = Fp
                
    return (n_optimal, Mp_optimal, Fp_optimal)
 
def EM(N,n,p0):
    div_S = N // (n**2)
    rem_S = N % (n**2)
    div_L = rem_S // n
    rem_L = rem_S % n
    return (EmS(n,p0)+ 2*n) * div_S + math.ceil(rem_S / n) + div_L * EmL(n,p0) + EmL(rem_L, p0)

def EF(N,n,p0):
    div_S = N // (n**2)
    rem_S = N % (n**2)
    div_L = rem_S // n
    rem_L = rem_S % n
    return div_S * (EfSG(n,p0) + EfSI(n,p0))+ div_L * (EfLG(n,p0) + EfLI(n,p0)) + (EfLG(rem_L,p0) + EfLI(rem_L,p0))
 

def mixed_array_optimal_n_formulation(N, p0, C):
    max_n = int(N**0.5)
    n_optimal = float('inf')
    M_optimal = float('inf')
    F_optimal = float('inf')
    for n in range(2,max_n+1):
        div = N // (n**2)
        rem = N % (n**2)
        if 2 * n * div + math.ceil(rem / n) < C:
            M = EM(N,n,p0)
            if M <= C:
                F = EF(N,n,p0)
                if F < F_optimal:
                        n_optimal = n
                        M_optimal = M
                        F_optimal = F
                
    return (n_optimal, M_optimal, F_optimal)


p0=0.001
ns= range(1,101)
N = 10000
t_before = time.time()
EMLps = [EMLp(n,p0) for n in ns]
EMSps = [EMSp(n,p0) for n in ns]
EFLps = [EFLp(n,p0) for n in ns]
EFSps = [EFSp(n,p0) for n in ns]
EFPLps = [EFPLp(n,p0) for n in ns]
EFPSps = [EFPSp(n,p0) for n in ns]
plt.plot(ns, EMLps, label='Linear Array')
plt.plot(ns, EMSps, label='Square Array')
plt.xlabel('pool size')
plt.ylabel('expected number of tests per person')
plt.legend()
plt.ticklabel_format(axis='y',style="sci", scilimits= (0,0), useMathText=True)
# plt.title('')
plt.savefig('test_per_person.png', dpi=300, bbox_inches='tight')
plt.show()

plt.plot(ns, EFLps, label='Linear Array')
plt.plot(ns, EFSps, label='Square Array')
plt.xlabel('pool size')
plt.ylabel('expected number of false negatives per person')
plt.legend()
plt.ticklabel_format(axis='y',style="sci", scilimits= (0,0), useMathText=True)
#plt.title('')
plt.savefig('fn_per_person.png', dpi=300, bbox_inches='tight')
plt.show()

plt.plot(ns, EFPLps, label='Linear Array')
plt.plot(ns, EFPSps, label='Square Array')
plt.xlabel('pool size')
plt.ylabel('expected number of false positives per person')
plt.legend()
plt.ticklabel_format(axis='y',style="sci", scilimits= (0,0), useMathText=True)
#plt.title('')
plt.savefig('fp_per_person.png', dpi=300, bbox_inches='tight')
plt.show()

# # EMLs = [EML(N,n,p0) for n in ns]
# # EMSs = [EMS(N,n,p0) for n in ns]
# # EFLs = [EFL(N,n,p0) for n in ns]
# # EFSs = [EFS(N,n,p0) for n in ns]

# # plt.plot(ns, EMLs, label='Linear Array')
# # plt.plot(ns, EMSs, label='Square Array')
# # plt.xlabel('pool size')
# # plt.ylabel('expected number of tests')
# # plt.legend()
# # # plt.title('')
# # plt.savefig('test_number', dpi=300, bbox_inches='tight')
# # plt.show()

# # plt.plot(ns, EFLs, label='Linear Array')
# # plt.plot(ns, EFSs, label='Square Array')
# # plt.xlabel('pool size')
# # plt.ylabel('expected false negative number')
# # plt.legend()
# # # plt.title('')
# # plt.savefig('fn_number.png', dpi=300, bbox_inches='tight')
# # plt.show()

plt.scatter(EMLps, EFLps, label='Linear Array')
plt.scatter(EMSps, EFSps, label='Square Array')
plt.xlabel('expected number of tests per person')
plt.ylabel('expected number of false negatives per person')
plt.legend()
plt.title('Trade-off between number of tests and number of false negatives\n')
plt.ticklabel_format(axis='y',style="sci", scilimits= (0,0), useMathText=True)
plt.axis([0,0.5,0,0.0005])
plt.savefig('fn_frontier.png', dpi=300, bbox_inches='tight')
plt.show()


plt.scatter(EMLps, EFPLps, label='Linear Array')
plt.scatter(EMSps, EFPSps, label='Square Array')
plt.xlabel('expected number of tests per person')
plt.ylabel('expected number of false positives per person')
plt.legend()
plt.title('Trade-off between number of tests and number of false positives\n')
plt.ticklabel_format(axis='y',style="sci", scilimits= (0,0), useMathText=True)
plt.axis([0,0.5,0,0.00006])
plt.savefig('fp_frontier.png', dpi=300, bbox_inches='tight')
plt.show()

t_after = time.time()
runtime = t_after - t_before
print('require %f seconds' % (runtime))


# plt.plot(ns, EFSps, label='FNR')
# plt.plot(ns, EFPSps, label='FPR')
# plt.xlabel('pool size')
# #plt.ylabel('expected number of false positives per person')
# plt.legend()
# plt.title('Square Array\n q = 0.01, p = 0.001')
# plt.show()




