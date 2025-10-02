# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 19:40:57 2025

@author: MMH_user
"""

import matplotlib.pyplot as plt
import numpy as np


def Decay(N: int = 100, tau: float = 1):
    
    R    = np.random.uniform(0, 1, (N,))
    Nvec = -np.arange(0, N) + N
    T    = - np.log(R)/Nvec/tau
    
        
    plt.stairs(Nvec[:-1], np.cumsum(T), color = 'r',\
               baseline = None, linewidth = 3)
        
    plt.xlabel('time')
    plt.ylabel('A(t)')