# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 00:09:14 2024

@author: MMH_user
"""
import numpy as np
import matplotlib.pyplot as plt

def IntegrationAccuracy():
    
    N      = [2, 5, 10, 20, 50, 100]
    lN     = len(N)
    
    x_plot = np.arange(0, np.pi, 1/10000)
    y_plot = np.sin(x_plot)
    
    plt.figure(figsize = (30,12))
    plt.subplots_adjust(hspace = 0.5)
    
    for i, n in enumerate(N):
                
        dx = np.pi/(n+1)
        x  = np.arange(0, np.pi, dx)
        y  = np.sin(x)
        
        #simple
        I1   = np.sum(y[:-1]*dx)
        acc1 = I1/2
        
        #average aka trapezoidal
        I2   = np.sum( (y[:-1] + y[1:])*dx/2 )
        acc2 = I2/2
        
        #simpson (quadratic)
        x2 = (x[:-1] + x[1:])/2
        y2 = np.sin(x2)
        
        I3   = np.sum(( y[:-1] + y[1:] + 4*y2) *dx/6)
        acc3 = I3/2
        
        plt.subplot(3, lN, i+1)
        plt.bar(x, y, color = 'w', edgecolor = 'k', linewidth = 1,\
                align = 'center', width = dx)
        plt.plot(x_plot, y_plot, 'r-', linewidth = 5, alpha = 0.5)
        plt.title("N = " + str(n) + f"\n accuracy =  {acc1: .3f}")
        if i ==0:
            plt.ylabel("simple")
        
        plt.subplot(3, lN, i+1 + lN)
        plt.bar(x, y, color = 'w', edgecolor = 'k', linewidth = 1,\
                align = 'center', width = dx)
        plt.plot(x_plot, y_plot, 'r-', linewidth = 4, alpha = 0.5)
        plt.title("N = " + str(n) + f"\n accuracy =  {acc2: .3f}")
        if i ==0:
            plt.ylabel("trapezoidal")
        
        plt.subplot(3, lN, i+1 + 2* lN)
        plt.bar(x, y, color = 'w', edgecolor = 'k', linewidth = 1,\
                align = 'center', width = dx)
        plt.plot(x_plot, y_plot, 'r-', linewidth = 5, alpha = 0.5)
        plt.title("N = " + str(n) + f"\n accuracy =  {acc3: .3f}")
        if i ==0:
            plt.ylabel("simpson n = 2")
        
        