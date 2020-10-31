#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:13:39 2020

@author: ryx
"""

import numpy as np
import time
import math

d_cens = 35.6
pi = [0.32, 0.53, 0.15];
mu = [20.14, 29.35, 34.78];
sigma = [3.60, 2.96, 1.32];

q = 0.01

def FNR(x):
    return 1/(1+math.exp(-12.5*(x-35.8)))

def rv_generation(pi, mu, sigma):
    U = np.random.uniform(0,1)
    if U <= pi[0]:
        X = np.random.normal(mu[0],sigma[0])
    if U > pi[0] and U <= pi[0] + pi[1]:
        X = np.random.normal(mu[1],sigma[1])
    if U > pi[0] + pi[1] and U <= pi[0] + pi[1] + pi[2]:
        X = np.random.normal(mu[2],sigma[2])
    return X

t_before = time.time() 
    
iteration = 10000
T = np.zeros((100,10,10))
T_CI = np.zeros((100,10,10,2))
P = np.zeros((100,10,10))
P_CI = np.zeros((100,10,10,2))
count_3 = 0
for n in range(1,101):
    for d1 in range(1,10):
        for d2 in range(1,d1+1):
            # total 4500
            count_3 += 1
            print('Progress: '+ '{:.1%}'.format(count_3/4500))
            count_1, count_2 = 0, 0
            record1, record2 = [], []            
            for k in range(iteration):
                add1, add2 = 0, 0
                C0 = 2**(-rv_generation(pi, mu, sigma))
                Crow = [2**(-rv_generation(pi, mu, sigma)) for i in range(d1-1)]
                Ccol = [2**(-rv_generation(pi, mu, sigma)) for i in range(d2-1)]
                Crow += [C0]
                Ccol += [C0]
                Crow_pool, Ccol_pool = -math.log(sum(Crow)/n, 2), -math.log(sum(Ccol)/n, 2);
                U = np.random.uniform(0,1)
                V = np.random.uniform(0,1)
                if (not (U<=FNR(Crow_pool))) and (not (V<=FNR(Ccol_pool))):
                    add1 = 1
                    count_1 += add1
                    W = np.random.uniform(0,1)
                    if W<=FNR(-math.log(C0, 2)):
                        add2 = 1
                        count_2 += add2
                record1.append(add1)
                record2.append(add2)
            hw1 = 1.96*np.std(record1)/math.sqrt(iteration)
            hw2 = 1.96*np.std(record2)/math.sqrt(iteration)
            T[n-1,d1-1,d2-1] = count_1/iteration
            P[n-1,d1-1,d2-1] = count_2/iteration
            T_CI[n-1,d1-1,d2-1] = [T[n-1,d1-1,d2-1]-hw1, T[n-1,d1-1,d2-1]+hw1]
            P_CI[n-1,d1-1,d2-1] = [P[n-1,d1-1,d2-1]-hw2, P[n-1,d1-1,d2-1]+hw2]
            
np.save('Theta_N_d.npy', T)
np.save('Psi_N_d.npy', P)
np.save('Theta_N_d_CI.npy', T_CI)
np.save('Psi_N_d_CI.npy', P_CI) 
   
T_round, P_round = np.zeros([100,10,10]), np.zeros([100,10,10])

for i in range(100):
    for j in range(10):
        for k in range(10):
            hw_T = 0.5*(T_CI[i][j][k][1]-T_CI[i][j][k][0])
            if hw_T != 0:
                T_round[i][j][k] = round(T[i][j][k], math.ceil(-math.log(hw_T,10)))
            hw_P = 0.5*(P_CI[i][j][k][1]-P_CI[i][j][k][0])
            if hw_P != 0:
                P_round[i][j][k] = round(P[i][j][k], math.ceil(-math.log(hw_P,10)))

np.save('Theta_N_d_round.npy', T_round)
np.save('Psi_N_d_round.npy', P_round)
            
t_after = time.time()
runtime = t_after - t_before
print('require %f seconds' % (runtime))

