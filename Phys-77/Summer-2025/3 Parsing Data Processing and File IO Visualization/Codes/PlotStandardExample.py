# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 23:16:29 2025

@author: MMH_user
"""

from math import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def PlotStandardExample():

    Data = pd.read_csv('data sets/Molecules.csv')
    
    x    = Data.molecular_weight
    y    = Data.bond_lengths
    
    figSt, axes = plt.subplots(nrows = 2, ncols = 2, layout = "constrained")
    
    axes[1, 1].hist(x,20, density=True, histtype = 'step', facecolor = 'g',
                   alpha = 0.75)
    axes[1, 1].set(title = 'histogramm')
        
    axes[0, 0].scatter(x, y, s = 20, c = 'k', alpha = 0.2, edgecolors = 'none')
    axes[0, 0].set(xlabel = 'X value')
    
    axes[1, 0].pie([24, 11, 11, 10, 5,39], 
                  colors = ['#6495ED', 'blue', [0.9, 0.9, 0.9], '#CCCCFF', '#000080', '#999999'])
    axes[1, 0].set(title = 'TIOBE Feb 2025')
    axes[1, 0].legend(['Python', 'C++', 'Java', 'C', 'C#', 'other'],\
                     bbox_to_anchor =(0.5,-0.5), loc = 'lower center',
                     ncol = 2)
    axes[0, 1].boxplot([x,y])
    axes[0, 1].set(xlabel = 'sample')
    axes[0, 1].set(ylabel = 'values')
    axes[0, 1].set(title = 'box plot')
    

    
    
    figSt.savefig('test.pdf', dpi = 1600)






