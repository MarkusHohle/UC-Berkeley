# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:21:23 2023

@author: MMH_user
"""
import numpy as np
import matplotlib.pyplot as plt

def PlotPositionEncoding(seq_len = 100, d = 512, n = 10000):

    denominator = np.power(n, 2*np.arange(int(d/2))/d)
    P = np.zeros((seq_len,d))
    P[:,::2]  = np.sin(np.arange(seq_len)/denominator[:,None]).T
    P[:,1::2] = np.cos(np.arange(seq_len)/denominator[:,None]).T
     
    plt.matshow(P, cmap = 'Blues')
    plt.xlabel('dimension k')
    plt.ylabel('position p')
    plt.show()
    
    return P


        