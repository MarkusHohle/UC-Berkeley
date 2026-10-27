# -*- coding: utf-8 -*-
"""
Created on Thu May  8 01:53:11 2025

@author: MMH_user
"""

from scipy.stats import *
import numpy as np
import matplotlib.pyplot as plt

def FalsePositive(N: int = 10, n: int = 1, alpha: float = 0.05):
    
    #N:     number of measurements
    #n:     at least n p-values
    #alpha: threshold for accepting H1
    
    Ntrial     = np.arange(N) + 1     # number of trials: (1...N)
    Ntrial     = Ntrial[n-1:]         # at least (n....N) trials for at least n FP
    
    P_at_least = np.zeros((N-n+1,))
    
    for i_n in Ntrial:
        P_at_least += binom.pmf(i_n, Ntrial, alpha) # it is easier to sum up pmf instead of using fully vectorized cdf

    
    plt.plot(Ntrial, P_at_least, 'r*-')
    plt.plot([0, N],[1, 1], 'k-')
    plt.plot([0, N],[0.9, 0.9], 'k--')
    plt.plot([0, N],[0.5, 0.5], 'k--')
    plt.plot([0, N],[alpha, alpha], 'k--')
    plt.xlabel('N measurements')
    plt.ylabel('Probability for at least ' + str(n) + ' FP')
    plt.title(r"$\alpha$ = " + str(alpha))