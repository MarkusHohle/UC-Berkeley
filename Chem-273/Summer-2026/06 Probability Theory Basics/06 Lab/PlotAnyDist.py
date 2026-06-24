# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:31:54 2026

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt

def PlotRandom(dist: str = 'uniform', shape: tuple = (10,), **kwargs):
    
    pdf  = getattr(np.random, dist)
    data = pdf(**kwargs, size = shape)
    
    plt.hist(data, density = True, histtype = 'step', color = 'k')
    