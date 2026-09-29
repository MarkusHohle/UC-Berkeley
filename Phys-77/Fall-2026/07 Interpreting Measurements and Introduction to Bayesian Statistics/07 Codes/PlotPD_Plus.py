# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:41:14 2025

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt

def PlotPD_Plus(P_D: list = [0.0001, 0.001, 0.01, 0.1]):
    
    P_PlusD = np.arange(0.5,0.99,0.01)
    P_PlusH = np.arange(0.001,0.1,0.001)
    
    x, y    = np.meshgrid(P_PlusD, P_PlusH)
    
    fig, axes = plt.subplots(nrows = 1,\
                             ncols = len(P_D), figsize=(10*len(P_D),5))
    
    for i, p_d in enumerate(P_D):
        
        P_D_Plus = 1/(1 + (y/x)*( (1-p_d)/p_d)  )
        
        P_D_Plus = np.log10(P_D_Plus)
        
        #axes[i].pcolormesh(x, y, P_D_Plus,\
        #                   shading='gouraud', cmap = 'Blues')
        axes[i].set_title('for P(D) = ' + str(p_d))
        CS  = axes[i].contourf(x, y, P_D_Plus, cmap = 'Blues')
        CSC = axes[i].contour(CS, levels = CS.levels, colors='k')
        axes[i].set_ylabel('P(+|H)')
        axes[i].set_xlabel('P(+|D)')
        
        cbar = fig.colorbar(CS)
        cbar.ax.set_ylabel("$log_{10}[P(D|+)]$")
        cbar.add_lines(CSC)
    
    plt.savefig('PD_Plus.pdf', dpi = 1800)
    plt.show()
        
        
        
        