# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 23:16:29 2025

@author: MMH_user
"""

#from math import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def PlotMosaicExample():

    Data = pd.read_csv('data sets/Molecules.csv')
    
    x    = Data.molecular_weight
    y    = Data.bond_lengths
    
    figMo, axes = plt.subplot_mosaic([['A', 'B', 'C'],\
                                     ['D', 'D', 'C']], layout = "constrained")
        
    axes['A'].scatter(x, y, s = 20, c = 'k', alpha = 0.2, edgecolors = 'none')
    axes['A'].set(xlabel = 'X value')
    
    axes['B'].pie([24, 11, 11, 10, 5,39], 
                  colors = ['#6495ED', 'blue', [0.9, 0.9, 0.9], '#CCCCFF', '#000080', '#999999'])
    axes['B'].set(title = 'TIOBE Feb 2025')
    axes['B'].legend(['Python', 'C++', 'Java', 'C', 'C#', 'other'],\
                     bbox_to_anchor =(0.5,-0.5), loc = 'lower center',
                     ncol = 2)
    axes['C'].boxplot([x,y])
    axes['C'].set(xlabel = 'sample')
    axes['C'].set(ylabel = 'values')
    axes['C'].set(title = 'box plot')
    
    axes['D'].hist(x,20, density=True, histtype = 'step', facecolor = 'g',
                   alpha = 0.75)
    axes['D'].set(title = 'histogramm')
    
    
    figMo.savefig('test.pdf', dpi = 1600)






