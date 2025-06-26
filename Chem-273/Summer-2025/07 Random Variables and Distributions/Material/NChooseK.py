# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:49:45 2024

@author: MMH_user
"""

import math
import numpy as np
import matplotlib.pyplot as plt

n_choose_k = math.comb


def n_choose_k_vs_Stirling(n, k):
    
    exact      = n_choose_k(n,k)
    
    #log for numerical issues
    approx_low = np.exp(k+ k*np.log(n)- k*np.log(k))
    approx_hig = np.exp(   k*np.log(n)- k*np.log(k))
    #approx = ((n/k)**k)/k_fac
    
    return exact, approx_hig, approx_low 


K = [2, 4, 6]
N = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000,\
     int(1e+6)]

N_choose_K_vs_Stirling = lambda N, k: [n_choose_k_vs_Stirling(n, k) for n in N]



for k in K:
    col    = np.random.uniform(0,1,3)
    Result = np.array(N_choose_K_vs_Stirling(N,k))
    plt.plot(N, Result[:,0], '*-', label = 'k = ' + str(k), c = col)
    plt.plot(N, Result[:,1], '--', label = 'k = ' + str(k), c = col)
    plt.plot(N, Result[:,2], '--', label = 'k = ' + str(k), c = col)
    
plt.title(r"$\star$- : exact" +'\n'+ "--: Stirling's approximation" )
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel(r'$\binom{n}{k}$', rotation = 0, fontsize = 18, labelpad = 20)
plt.show()


for k in K:
    col    = np.random.uniform(0,1,3)
    Result = np.array(N_choose_K_vs_Stirling(N,k))
    rel    = (Result[:,1] - Result[:,0])/Result[:,1]
    plt.plot(N, Result[:,0], '*-', label = 'k = ' + str(k), c = col)
    
plt.title(r"relative error" )
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.show()