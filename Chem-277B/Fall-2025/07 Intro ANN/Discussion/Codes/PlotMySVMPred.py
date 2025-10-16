# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:54:14 2023

@author: MMH_user
"""
import numpy as np
import matplotlib.pyplot as plt

def PlotMySVMPred(data, model, Ynum, accur):
    
    N = 100

    x1 = np.linspace(data[:,0].min(), data[:,0].max(), N)
    x2 = np.linspace(data[:,1].min(), data[:,1].max(), N)
    
    [xx1, xx2] = np.meshgrid(x1, x2)
    
    P = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    P = P.reshape(xx1.shape)
    
    CM = "BuGn"
    
    plt.contourf(xx1, xx2, P, cmap = CM, alpha = 0.7)
    ax = plt.scatter(data[:, 0], data[:, 1], c = Ynum, cmap = CM, \
                edgecolors = 'grey')
    plt.legend(labels = ['setosa', 'versicolor', 'virginica'], \
               handles = ax.legend_elements()[0], title = "species")
    plt.title(f"accuracy = {accur: .2f}%")
    plt.show()
    
    
    