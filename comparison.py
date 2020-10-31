#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 01:16:01 2020

@author: ryx
"""

from closed_form_A2_approximate_10 import *
# from benchmark import *
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import ticker

def L_Array_optimal_1000(N, p0, max_test=1000):
    max_n = int(N**0.5)
    list_1000 = []
    for n in range(2,max_n+1):
        div = N // (n)
        if div  < max_test:
            M = EML(N,n,p0)
            if M <= max_test:
                fn = EFL(N,n,p0)
                list_1000 += [[n,M,fn]]
    return list_1000

def L_Array_optimal_p0(N, p0, max_tests):
    l = L_Array_optimal_1000(N, p0, max_test=1000)
    fn_optimals = []
    for max_test in max_tests:
        # n_optimal = float('inf')
        # M_optimal = float('inf')
        fn_optimal = float('inf')
        for n,M,fn in l:
            if M <= max_test and fn < fn_optimal:
                # n_optimal = n
                # M_optimal = M
                fn_optimal = fn
        if fn_optimal == float('inf'):
            fn_optimal = float('nan')
        fn_optimals += [fn_optimal]
    return fn_optimals

def S_Array_optimal_1000(N, p0, max_test=1000):
    max_n = int(N**0.5)
    list_1000 = []
    for n in range(2,max_n+1):
        div = N // (n**2)
        rem = N % (n**2)
        if 2 * n * div + rem <= max_test:
            M = EMS(N,n,p0)
            if M <= max_test:
                fn = EFS(N,n,p0)
                list_1000 += [[n,M,fn]]
    return list_1000

def S_Array_optimal_p0(N, p0, max_tests):
    l = S_Array_optimal_1000(N, p0, max_test=1000)
    fn_optimals = []
    for max_test in max_tests:
        # n_optimal = float('inf')
        # M_optimal = float('inf')
        fn_optimal = float('inf')
        for n,M,fn in l:
            if M <= max_test and fn < fn_optimal:
                # n_optimal = n
                # M_optimal = M
                fn_optimal = fn
        if fn_optimal == float('inf'):
            fn_optimal = float('nan')
        fn_optimals += [fn_optimal]
    return fn_optimals


def L_Array_optimal_per_person_1000(N, p0, max_test=1000):
    c = max_test/N
    max_n = int(N**0.5)
    list_1000 = []
    for n in range(2,max_n+1):
        if 1/n  < c:
            M = EMLp(n,p0)
            if M <= c:
                fn = EFLp(n,p0)
                fp = EFPLp(n,p0)
                list_1000 += [[n,M,fn,fp]]
    return list_1000

def L_Array_optimal_per_person_p0(N, p0, max_tests):
    l = L_Array_optimal_per_person_1000(N, p0, max_test=1000)
    fn_optimals = []
    fp_optimals = []
    for max_test in max_tests:
        c = max_test/N
        # n_optimal = float('inf')
        # M_optimal = float('inf')
        fn_optimal = float('inf')
        fp_optimal = float('inf')
        for n,M,fn,fp in l:
            if M <= c and fn < fn_optimal:
                # n_optimal = n
                # M_optimal = M
                fn_optimal = fn
                fp_optimal = fp
        if fn_optimal == float('inf'):
            fn_optimal = float('nan')
            fp_optimal = float('nan')
        fn_optimals += [fn_optimal]
        fp_optimals += [fp_optimal]
    return (fn_optimals,fp_optimals)

def S_Array_optimal_per_person_1000(N, p0, max_test=1000):
    c = max_test/N
    max_n = int(N**0.5)
    list_1000 = []
    for n in range(2,max_n+1):
        if 2 / n  <= c:
            M = EMSp(n,p0)
            if M <= c:
                fn = EFSp(n,p0)
                fp = EFPSp(n,p0)
                list_1000 += [[n,M,fn,fp]]
    return list_1000

def S_Array_optimal_per_person_p0(N, p0, max_tests):
    l = S_Array_optimal_per_person_1000(N, p0, max_test=1000)
    fn_optimals = []
    fp_optimals = []
    for max_test in max_tests:
        c = max_test/N
        # n_optimal = float('inf')
        # M_optimal = float('inf')
        fn_optimal = float('inf')
        fp_optimal = float('inf')
        for n,M,fn,fp in l:
            if M <= c and fn < fn_optimal:
                # n_optimal = n
                # M_optimal = M
                fn_optimal = fn
                fp_optimal = fp
        if fn_optimal == float('inf'):
            fn_optimal = float('nan')
            fp_optimal = float('nan')
        fn_optimals += [fn_optimal]
        fp_optimals += [fp_optimal]
    return (fn_optimals,fp_optimals)


N = 10000
P_0s = np.arange(0.0005,0.0015,0.00005)
max_tests = np.arange(100,1000,50)
row = len(P_0s)
col = len(max_tests)
# S_fn = []
# L_fn = []
# t_before = time.time()
# for p0 in P_0s:
#     L_fn += [L_Array_optimal_p0(N, p0, max_tests)]
#     S_fn += [S_Array_optimal_p0(N, p0, max_tests)]
# t_after = time.time()
# runtime = t_after - t_before
# print('require %f seconds' % runtime)

# gridx = np.repeat(np.matrix(P_0s),col,0).T
# gridy = np.repeat(np.matrix(max_tests),row,0)

# surf = plt.contourf(gridx, gridy, S_fn, cmap=cm.coolwarm, vmin=0, vmax=30)
# plt.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel('$p$')
# plt.ylabel('$C$')
# plt.title('Square Array $\mathbb{E}[ F^{S}(N,n^*)]$')
# plt.savefig('Square array_2d.png', dpi=300)
# plt.show()


# surf = plt.contourf(gridx, gridy, L_fn, cmap=cm.coolwarm, vmin=0, vmax=30)
# plt.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel('$p$')
# plt.ylabel('$C$')
# plt.title('Linear Array $\mathbb{E}[ F^{L}(N,n^*)]$')
# plt.savefig('Linear array_2d.png', dpi=300)
# plt.show()


S_fn_per_person = []
L_fn_per_person = []
S_fp_per_person = []
L_fp_per_person = []
t_before = time.time()
for p0 in P_0s:
    fn_optimals,fp_optimals = L_Array_optimal_per_person_p0(N, p0, max_tests)
    L_fn_per_person += [fn_optimals]
    L_fp_per_person += [fp_optimals]
    fn_optimals,fp_optimals = S_Array_optimal_per_person_p0(N, p0, max_tests)
    S_fn_per_person += [fn_optimals]
    S_fp_per_person += [fp_optimals]
t_after = time.time()
runtime = t_after - t_before
print('require %f seconds' % runtime)

fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
gridx = np.repeat(np.matrix(P_0s),col,0).T
gridy = np.repeat(np.matrix(max_tests)/N,row,0)

surf = plt.contourf(gridx, gridy, S_fn_per_person, cmap=cm.coolwarm, vmin=0, vmax=0.001)
plt.colorbar(surf, shrink=0.5, aspect=5, format=fmt)
plt.xlabel('Prevalence rate $p$')
plt.ylabel('Test capacity per person $C$')
plt.title('Square Array\nexpected number of false negatives per person')
plt.savefig('Square_array_2d.png', dpi=300)
plt.show()


surf = plt.contourf(gridx, gridy, L_fn_per_person, cmap=cm.coolwarm, vmin=0, vmax=0.001)
plt.colorbar(surf, shrink=0.5, aspect=5, format=fmt)
plt.xlabel('Prevalence rate $p$')
plt.ylabel('Test capacity per person $C$')
plt.title('Linear Array\nexpected number of false negatives per person')
plt.savefig('Linear_array_2d.png', dpi=300)
plt.show()

bench_fn_per_person = []
for p0 in P_0s:
    bench_fn_per_person_row = []
    for C in max_tests:
        c = C/N
        bench_fn_per_person_row += [(1-c)*p0] # This is a lower bound
    
    bench_fn_per_person += [bench_fn_per_person_row]   
    
surf = plt.contourf(gridx, gridy, bench_fn_per_person, cmap=cm.coolwarm, vmin=0, vmax=0.001)
plt.colorbar(surf, shrink=0.5, aspect=5, format=fmt)
plt.xlabel('Prevalence rate $p$')
plt.ylabel('Test capacity per person $C$')
plt.title('Benchmark individual testing\nexpected number of false negatives per person')
plt.savefig('benchmark_2d.png', dpi=300)
plt.show()

surf = plt.contourf(gridx, gridy, S_fp_per_person, cmap=cm.coolwarm, vmin=0, vmax=0.001)
plt.colorbar(surf, shrink=0.5, aspect=5, format=fmt)
plt.xlabel('Prevalence rate $p$')
plt.ylabel('Test capacity per person $C$')
plt.title('Square Array\nexpected number of false positives per person')
plt.savefig('Square_array_2d_fp.png', dpi=300)
plt.show()


surf = plt.contourf(gridx, gridy, L_fp_per_person, cmap=cm.coolwarm, vmin=0, vmax=0.001)
plt.colorbar(surf, shrink=0.5, aspect=5, format=fmt)
plt.xlabel('Prevalence rate $p$')
plt.ylabel('Test capacity per person $C$')
plt.title('Linear Array\nexpected number of false positives per person')
plt.savefig('Linear_array_2d_fp.png', dpi=300)
plt.show()