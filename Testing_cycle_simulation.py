# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:47:07 2020

@author: Jingyuan Wan
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math
import sys
sys.path.append(r"D:\Research\COVID-19 project\101920")
import closed_form_A2_approximate_10 as cf
import pool_simulation as ps
import numpy.random as npr
import os

import copy
from scipy.stats import norm 
from scipy.stats import binom

Gamma = np.load('F_N_d_round.npy')
Eta = np.load('Eta_N_d_round.npy')
Theta = np.load('Theta_N_d_round.npy')
Psi = np.load('Psi_N_d_round.npy')


class Individual(object):
    def __init__(self, ID = 0, pos = 0, tested = 0):
        self.ID = ID;
        self.pos = pos;
        self.tested = tested;
        

def CRN(N, p, seed=123456):
    np.random.seed(seed)
    return [np.random.binomial(N, p) for i in range(iter_time)]

##### Recurrence starts here

start=time.time()
# Parameters
V_0 = 0.001 # initial prevalence rate

t = 2 # period in which a full test on all individuals is conducted, i.e., testing cycle length
T = 100000 # total number of populations across the campus (students, staff, faculty)
alpha = 1.1 # infection rate
C = 3000 # daily test capacity
out_rate = 0.00002 # infection rate induced by outside contacts
fpr = 0.01 #false positive rate 

iter_time, total_length = 100, 14 # number of iteration time and time horizon


final_prvl_rate = []
## These matrices store all data in each day 
prvl_rate_matrix = [] # Prevalence rate

S_matrix = []   # Group size
M_matrix = []   # Test times
Q_matrix = []   # Total quarantined
RQ_matrix = []  # Right quarantined
WQ_matrix = []  # Wrong quarantined
fn_matrix = []  # False negative number

# Individual test or group test
Individual_test = 0

# Set common random numbers

Initial_infected_num = CRN(T, V_0)

for iterative in range(iter_time):
    print("iteration ", iterative)
    ERROR, flag = 0, 0
    # Initialization
    T = 100000 # Total population
    
    TI = np.zeros(total_length+2) # tested, infected, after one day.
    TNI = np.zeros(total_length+2) # tested, not infected, after one day.
    NTI = np.zeros(total_length+2) # not tested, infected, after one day.
    NTNI = np.zeros(total_length+2) # not tested, not infected, after one day.
     
    # inc: increase
    TQ_inc = np.zeros(total_length+2) # tested, quarantined
    TRQ_inc = np.zeros(total_length+2) # tested, infected, and quarantined
    TWQ_inc = np.zeros(total_length+2) # tested, not infected, but quarantined, i.e., those false positives.
    TNQ_inc = np.zeros(total_length+2) # tested, infected, but not quarantined, i,e., those false negatives.
    TNI_inc = np.zeros(total_length+2) # tested, not infected
    QI = np.zeros(total_length+2) # Quarantined, infected
    RQI = np.zeros(total_length+2) # Right Quarantined, infected
    WQI = np.zeros(total_length+2) # Wrong Quarantined, infected
    
    TI_test = np.zeros(total_length+2) # tested, infected, after the test stage.
    NTI_test = np.zeros(total_length+2) # not tested, infected, after the test stage.
    TNI_test = np.zeros(total_length+2) # tested, not infected, after the test stage.
    NTNI_test = np.zeros(total_length+2) # not tested, not infected, after the test stage.
    
    # Response initialization
    M_row, N_row, n_row = [], [], []
    prvl_rate_row, Q_row, RQ_row, WQ_row, fn_row = [V_0], [], [], [], []
    
    TI[0] = 0
    TNI[0] = 0
    NTI[0] = Initial_infected_num[iterative]
    NTNI[0] = T - NTI[0]
#    NTI[0] = T*V_0
#    NTNI[0] = T-T*V_0
    
    # Create and Initialize an array of all individuals
    All_Individuals = [];
    for i in range(T):
        tmp = Individual(0)
        tmp.ID = i+1
        All_Individuals.append(tmp)
    
    # Sub-array for all individuals that not infected
    NI_Individuals = copy.copy(All_Individuals)
    
    Initial_infected = random.sample(All_Individuals, int(NTI[0]))
    for ind in Initial_infected:
        ind.pos = 1
        NI_Individuals.remove(ind)
    
    # Sub-array for individuals that not tested and not quarantined, respectively.
    NT_Individuals = copy.copy(All_Individuals)
    NQ_Individuals = copy.copy(All_Individuals)
    
    # Number of not quarantined, number of not tested, initial prevalence rate, date
    Num_NQ, Num_NT, V_i, i = T, T, V_0, 0
    
    # Recurrence Formulation
    # In this stage, we add random sampling in the selection of tested person and the virus spreading.
    while i <= total_length-1: 
        # Update before starting a new testing cycle
        Num_NT = T
        for ind in NQ_Individuals:
            ind.tested = 0
        NT_Individuals = copy.copy(NQ_Individuals)
        print('i =', i, 'T =', T)
        if i != 0:
            TI[i], TNI[i], NTI[i], NTNI[i] = 0, 0, V_i*T, (1-V_i)*T
        # Determine daily task: how many tests are going to take, how many people get tested, the group size and the false negative rate
        if Individual_test:
            Num_test_today, test_times_today = C, C 
            group_size = -2
            f_n_rate = Gamma[0][0]
        else:
            Num_test_today = int(min(np.ceil(T/t), Num_NT))
            print(Num_test_today)

            # Switch formulaations: use total number or per person
            #group_size = cf.mixed_array_optimal_n_formulation(Num_test_today, V_i, C)[0]
            group_size = cf.square_array_optimal_n_formulation_2(Num_test_today, V_i, C)[0]
            if group_size == float('inf'):
                ERROR = 1
                break
        
        print('optimal pool size is', group_size)
        T0 = np.ceil(T/t)
        # One testing cycle
        while Num_NT > 0:
            i = i + 1 # Day i
            if i == total_length+1:
                flag = 1
                break
            
            Num_test_today = int(min(T0, Num_NT))
            N_row.append(Num_test_today)
            n_row.append(group_size)
            Num_NT = Num_NT - Num_test_today
    
            # Randomly Select N individuals to get tested on the test stage
            not_tested = []
            for ind in NQ_Individuals:
                if ind.tested == 0:
                    not_tested.append(ind)
                    
            test = random.sample(not_tested, int(Num_test_today))
            print('Number of test today is',Num_test_today)
            for ind in test:
                ind.tested = 1

            ## Test Stage
            if not Individual_test:
                # Conduct square array pool test
                test_times_today = 0
                num_pool = int(Num_test_today/group_size**2)
                for k in range(num_pool):
                    pool = np.zeros([group_size, group_size])
                    pool_k = test[k*group_size**2:(k+1)*group_size**2]
                    for l in range(group_size):
                        for j in range(group_size):
                            pool[l][j] = pool_k[l*group_size+j].pos
                    Q_k, RQ_k, WQ_k, NQ_k, time_k = ps.single_square_array(group_size, pool, fpr)
                    TQ_inc[i] += len(Q_k)
                    TRQ_inc[i] += len(RQ_k)
                    TWQ_inc[i] += len(WQ_k)
                    TNQ_inc[i] += len(NQ_k)
                    test_times_today += time_k
                    for p in Q_k:
                        NQ_Individuals.remove(pool_k[p[0]*group_size+p[1]])
                # Conduct linear array pool test
                num_array = math.ceil((Num_test_today-num_pool*group_size**2)/group_size)
                if num_array!=0:
                    for k in range(num_array-1):
                        array = np.zeros(group_size)
                        array_k = test[num_pool*group_size**2+k*group_size:num_pool*group_size**2+(k+1)*group_size]
                        for j in range(group_size):
                            array[j] = array_k[j].pos
                        Q_k, RQ_k, WQ_k, NQ_k, time_k = ps.single_linear_array(group_size, array, fpr)
                        TQ_inc[i] += len(Q_k)
                        TRQ_inc[i] += len(RQ_k)
                        TWQ_inc[i] += len(WQ_k)
                        TNQ_inc[i] += len(NQ_k)
                        test_times_today += time_k
                        for p in Q_k:
                            NQ_Individuals.remove(array_k[p]) 
                    # Deal with the last linear array
                    array_last = test[num_pool*group_size**2+(num_array-1)*group_size:]
                    array = np.zeros(len(array_last))
                    for j in range(len(array_last)):
                        array[j] = array_last[j].pos
                    Q_k, RQ_k, WQ_k, NQ_k, time_k = ps.single_linear_array(len(array_last), array, fpr)
                    TQ_inc[i] += len(Q_k)
                    TRQ_inc[i] += len(RQ_k)
                    TWQ_inc[i] += len(WQ_k)
                    TNQ_inc[i] += len(NQ_k)
                    test_times_today += time_k
                    for p in Q_k:
                        NQ_Individuals.remove(array_last[p])
            else: 
                # Individual test
                for ind in test:
                    if ind.pos == 1:
                        U = np.random.uniform(0,1)
                        if U <= f_n_rate:
                            TNQ_inc[i] += 1
                        else:
                            TQ_inc[i] += 1
                            TRQ_inc[i] += 1
                            NQ_Individuals.remove(ind)
                    else:
                        U = np.random.uniform(0,1)
                        if U <= fpr:
                            TQ_inc[i] += 1
                            TWQ_inc[i] += 1
                            NQ_Individuals.remove(ind)
                            
                
            
            QI[i] = QI[i-1] + TQ_inc[i]
            RQI[i] = RQI[i-1] + TRQ_inc[i]
            WQI[i] = WQI[i-1] + TWQ_inc[i]
            TNI_inc[i] = Num_test_today - TQ_inc[i] - TNQ_inc[i]
                
            Num_NQ = Num_NQ - TQ_inc[i]
            TI_test[i] = TI[i-1] + TNQ_inc[i]
            NTI_test[i] = NTI[i-1] - (TRQ_inc[i] + TNQ_inc[i])   # TQ_inc[i] individuals get quarantined.
            TNI_test[i] = TNI[i-1] + TNI_inc[i]
            NTNI_test[i] = NTNI[i-1] - TNI_inc[i] - TWQ_inc[i]
            #print('TI_test[i]=', TI_test[i], 'TNI_test[i]=', TNI_test[i], 'NTI_test[i]=', NTI_test[i], 'NTNI_test[i]=', NTNI_test[i])
            M_row.append(test_times_today)
        
            ## Spread Stage
            # Randomly Select new infected individuals based on the infected rate: alpha.
            Sus_Individuals = []
            for ind in NQ_Individuals:
                if ind.pos == 0:
                    Sus_Individuals.append(ind)
            new_infected_num = np.random.binomial((TI_test[i] + NTI_test[i]),(alpha - 1))+np.random.binomial(len(Sus_Individuals), out_rate)  # include outside contact
            new_infected = random.sample(Sus_Individuals, new_infected_num)
            #print(new_infected_num)
                
            NTNI[i] = NTNI_test[i]
            TNI[i] = TNI_test[i]
            TI[i] = TI_test[i]
            NTI[i] = NTI_test[i]
              
            for ind in new_infected:  ## Mark all new infected individuals 
                ind.pos = 1
                NI_Individuals.remove(ind)
                if ind.tested == 1:
                    TNI[i] -= 1
                    TI[i] += 1
                else:
                    NTNI[i] -= 1
                    NTI[i] += 1
            
            # Update the prevalence rate
            V_i = (TI[i]+NTI[i])/Num_NQ
            prvl_rate_row.append(V_i)
            Q_row.append(TQ_inc[i])
            RQ_row.append(TRQ_inc[i])
            WQ_row.append(TWQ_inc[i])
            fn_row.append(TNQ_inc[i])
            print('TRQ_inc[i]=', TRQ_inc[i], 'TWQ_inc[i]=', TWQ_inc[i])
            print('TI[i] =', TI[i], 'TNI[i] =', TNI[i], 'NTI[i] =', NTI[i], 'NTNI[i] =', NTNI[i])
            print('V_i =', V_i, 'Number of not quarantined: ', Num_NQ, 'after day',i)
            if V_i == 0:
                break
        if flag == 1 or ERROR == 1:
            break
        # Update the new population that need to get tested in the next cycle
        T = Num_NQ
    if ERROR == 0 or ERROR == 1:
        prvl_rate_matrix.append(prvl_rate_row)
        S_matrix.append(n_row)
        M_matrix.append(M_row)
        Q_matrix.append(Q_row)
        RQ_matrix.append(RQ_row)
        WQ_matrix.append(WQ_row)
        fn_matrix.append(fn_row)
        final_prvl_rate.append(V_i)
    else:
        print("No feasible group size, N =", Num_test_today, "at day ", i)
        
    

final_prvl_rate_estimated = np.mean(final_prvl_rate);
print('The final prevalence rate for t = '+str(t)+' is '+str(final_prvl_rate_estimated))
print('The variance for t = '+str(t)+' is '+str(np.var(final_prvl_rate)))

day_vector = range(total_length+1)
average_prvl_rate, average_size, average_test_times, average_Q, average_fn = np.zeros(total_length+1), np.zeros(total_length), np.zeros(total_length), np.zeros(total_length), np.zeros(total_length);
average_RQ, average_WQ = np.zeros(total_length), np.zeros(total_length)
for i in range(total_length+1):
    for j in range(np.size(prvl_rate_matrix, 0)):
        average_prvl_rate[i] += prvl_rate_matrix[j][i]
    average_prvl_rate[i] = average_prvl_rate[i] / np.size(prvl_rate_matrix, 0)
for i in range(total_length):
    for j in range(np.size(M_matrix, 0)):
        average_size[i] += S_matrix[j][i]
        average_test_times[i] += M_matrix[j][i]
        average_Q[i] += Q_matrix[j][i]
        average_RQ[i] += RQ_matrix[j][i]
        average_WQ[i] += WQ_matrix[j][i]
        average_fn[i] += fn_matrix[j][i]
        
    average_size[i] = average_size[i] / np.size(S_matrix, 0)
    average_test_times[i] = average_test_times[i] / np.size(M_matrix, 0)
    average_Q[i] = average_Q[i] / np.size(Q_matrix, 0)
    average_RQ[i] = average_RQ[i] / np.size(RQ_matrix, 0)
    average_WQ[i] = average_WQ[i] / np.size(WQ_matrix, 0)
    average_fn[i] = average_fn[i] / np.size(fn_matrix, 0)
    
plt.plot(day_vector, average_prvl_rate, 'ro-', label = 'Individual Test')
plt.xlabel('Date')
plt.ylabel('Prevalence rate')
title = "The prevalence rate curve, " + "t = " + str(t)
plt.title(title)

print('The average prvalence rate is', average_prvl_rate)
print('The average optimal group size for each day is', average_size)
print('The average testing times is', average_test_times)
print('The average number of quarantined is', average_Q)
print('The average number of right quarantined is', average_RQ)
print('The average number of wrong quarantined is', average_WQ)
print('The average false negative number for each day is', average_fn)
# Record the time spent of the whole program and simulation results

#Simulation_result = np.array([average_prvl_rate, average_size, average_test_times, average_Q, average_RQ, average_WQ, average_fn])
#file_name = 'N100000_nominal_t='+str(t)+'.npy'
#file_name = 'Asym_N100000_pessimistic_t='+str(t)+'.npy'
#file_name = 'Simulation_results_nominal_t='+str(t)+'fpr='+str(fpr)+'.npy'
#file_name = 'Simulation_results_nominal_t='+str(t)+'C='+str(C)+'.npy'
#file_name = 'Asym_N100000_Simulation_results_nominal_individual.npy'
#np.save(file_name, Simulation_result)

end = time.time()
print('totally cost', end-start)











