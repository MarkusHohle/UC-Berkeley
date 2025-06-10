# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:46:00 2025

@author: MMH_user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:36:35 2025

@author: MMH_user
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Diffusion2D():
    
    def __init__(self):
        
        #defining grids
        Lx = 100
        Ly = 100
        Lt = 2500
                
        C = 3*np.ones((Lx,Ly,Lt))
      

        #adding up to 12 "seeds" at random locations
        N = round(12*np.random.rand())
        for n in range(N):
            C[int(Lx*np.random.rand()), int(Ly*np.random.rand()),0] = 500
            
            
        self.C = C
        
        self.Lx = Lx
        self.Ly = Ly
        self.Lt = Lt
        
        
    def RunSimulation(self, D = 0.08):
        
        C = self.C
        
       
        sns.heatmap(C[:,:,0], cbar = True, cmap="Blues",
                    xticklabels = False, yticklabels = False)
        plt.show()
        


        for k in range(1, self.Lt-1):
            for j in range(self.Ly):
        
                jrun_up   = j
                jrun_down = j
         
                #cyclic BCs----------------------------------------------------
                if j+1> self.Ly-1:
                    jrun_up = -1
                
                if j-1 == -1:
                    jrun_down = self.Ly - 1
                #--------------------------------------------------------------
        
                for i in range(self.Lx):
                    
                    irun_up   =i
                    irun_down =i
                
                    #cyclic BCs----------------------------------------------------
                    if i+1>self.Lx-1:
                        irun_up = -1
                        
                    if i-1 == -1:
                        irun_down = self.Lx - 1
                    #--------------------------------------------------------------
                                                                                                                                                                                                   
                    C[i,j,k] = 2*D*(C[irun_up + 1,j,k-1] + C[irun_down - 1,j,k-1] +\
                                        C[i,jrun_up+1,k-1] - 4*C[i,j,k-1] +\
                                        C[i,jrun_down-1,k-1]) +\
                                        C[i,j,k-1] 
 

            if not k % 250:
                
                sns.heatmap(C[:,:,k], cbar = True, cmap="Blues",
                            xticklabels = False, yticklabels = False)
                plt.show()
            
        
        
