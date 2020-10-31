#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:03:38 2020

@author: ryx
"""

#from closed_form_A2_approximate_10 import *
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm
from scipy.stats import binom
import random
import time
import math


gamma_matrix = np.load('F_N_d_round.npy')
eta_matrix = np.load('Eta_N_d_round.npy')
psi_matrix = np.load('Psi_N_d_round.npy')
theta_matrix = np.load('Theta_N_d_round.npy')

def gamma(n,d,q=0.001):
    if d > 10:
        return 0
    if d == 0:
        return 1 - q
    return gamma_matrix[n-1,d-1]

def eta(n,d):
    if d > 10:
        return gamma(1,1)
    return eta_matrix[n-1,d-1]

def psi(n,d1,d2):
    if d1 > 9 or d1 > 9:
        return gamma(1,1)
    if d1 > d2:
        return psi_matrix[n-1,d1-1,d2-1]
    return psi_matrix[n-1,d2-1,d1-1]

def theta(n,d1,d2):
    if d1 > 9 or d1 > 9:
        return 1
    if d1 > d2:
        return theta_matrix[n-1,d1-1,d2-1]
    return theta_matrix[n-1,d2-1,d1-1]

def single_square_array(n, A, fpr):
    # A is a single square array of n * n with entry 1 if swab infected, 0 otherwise
    A = A.reshape(n,n)
    # total infected
    # D = np.sum(A)
    q, rq, wq, nq = [], [], [], []
    test_time = 2*n
    for i in range(n):
        d_row = int(sum(A[i,:]))
        for j in range(n):
            d_col = int(sum(A[:,j]))
            if A[i,j] == 1:
                U = np.random.uniform(0,1)
                # with probability 1-theta(n,d_row,d_col) test negative
                if U > theta(n,d_row,d_col):
                    nq.append([i,j])
                # with probability psi(n,d_row,d_col) pool test positive and individual test negative
                elif U <= psi(n,d_row,d_col):
                    test_time += 1
                    nq.append([i,j])
                # if neither of the above two cases, this infected will be successfully detected.
                else:
                    test_time += 1
                    q.append([i,j])
                    rq.append([i,j])
            if A[i,j] == 0:
                U = np.random.uniform(0,1)
                # with probability 1-gamma(n,d_row) row test positive
                # with probability 1-gamma(n,d_col) col test positive
                # with probability (1-gamma(n,d_row))*(1-gamma(n,d_col)) pool test positive
                if U < (1-gamma(n,d_row))*(1-gamma(n,d_col)):
                    test_time += 1
                    V = np.random.uniform(0,1)
                    if V < fpr:
                        q.append([i,j])
                        wq.append([i,j])
    return (q, rq, wq, nq, test_time)

def single_linear_array(n, A, fpr):
    # A is a single linear array of n with entry 1 if swab infected, 0 otherwise
    #A = A.reshape(1,n)
    # total infected
    # D = np.sum(A)
    q, rq, wq, nq = [], [], [], []
    test_time = 1
    d = int(sum(A))
    for i in range(n):
        if A[i] == 1:
            U = np.random.uniform(0,1)
            # with probability gamma(n,d) test negative
            if U > 1-gamma(n,d):
                nq.append(i)
            # with probability psi(n,d_row,d_col) pool test positive and individual test negative
            elif U <= eta(n,d):
                test_time += 1
                nq.append(i)
            else:
                test_time += 1
                q.append(i)
                rq.append(i)
        if A[i] == 0:
            U = np.random.uniform(0,1)
            # with probability 1-gamma(n,d) pool test positive
            if U < (1-gamma(n,d)):
                test_time += 1
                V = np.random.uniform(0,1)
                if V < fpr:
                    q.append(i)
                    wq.append(i)
    return (q, rq, wq, nq, test_time)


