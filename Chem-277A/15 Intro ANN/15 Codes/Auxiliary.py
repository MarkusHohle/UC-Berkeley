# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 00:27:55 2026

@author: MMH_user
"""

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def FindMyFile(filename: str, ServerHardDiscPath: str = r"c:\Users\MMH_user\Desktop") -> str:
    """
    finds file of name "filename" anywhere in "ServerHardDiscPath" and returns complete path
    """
    for r,d,f in os.walk(ServerHardDiscPath):
        for files in f:
             if files == filename: #for example "MessyFile.xlsx"
                 file_name_and_path =  os.path.join(r,files)
                 return file_name_and_path
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def PlotClasses(X, Y):
    NClasses = np.max(Y)+1
    for c in range(NClasses):
        idx = np.argwhere(Y == c)[:,0]
        plt.scatter(X[idx, 0], X[idx, 1], label = str(c))
    plt.legend()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('spiral data set')
    plt.show()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def Compute_Boundary(X: np.array, model, torch_model = False):
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy       = np.meshgrid(np.linspace(x_min, x_max, 200),
                               np.linspace(y_min, y_max, 200))

    grid         = np.c_[xx.ravel(), yy.ravel()]

    if torch_model:
        with torch.no_grad():
            logits = model(torch.tensor(grid, dtype = torch.float32))
            Z      = torch.argmax(logits, dim = 1).numpy()
    else:
        preds = model.predict(grid, verbose = 0)
        Z = np.argmax(preds, axis = 1)

    return xx, yy, Z.reshape(xx.shape)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def Plot_Boundary_Snapshot(X:  np.array, Y:  np.array, 
                           xx: np.array, yy: np.array, 
                           Z:  np.array, title: str):
    
    plt.figure(figsize = (5,5))
    plt.contourf(xx, yy, Z, alpha = 0.3)
    NClasses = np.max(Y)+1
    for c in range(NClasses):
        idx = np.argwhere(Y == c)[:,0]
        plt.scatter(X[idx, 0], X[idx, 1], label = str(c), s = 10)
    plt.legend()
    plt.title(title)
    plt.show()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def plot_entropy(P, YNum, ClassLabs, ClassLabsNum):
    
    N, Nclass = P.shape
    
    fig, ax = plt.subplots(Nclass, 1, sharex = True)
    fig.set_figheight(Nclass)
    fig.subplots_adjust(hspace = 1.5)
    plt.title('entropy')
    for i, L in enumerate(ClassLabsNum):
        idx      = np.array([j for j, t in enumerate(YNum) if t == L])
        pclass   = P[idx,i]
        (value, where) = np.histogram(pclass,\
                                      bins = np.arange(0,1,0.01),\
                                      density = True)
        w = 0.5*(where[1:] + where[:-1])
        ax[i].plot(w, value, 'k-')
        ax[i].set_ylabel('frequency')
        ax[i].set_title(ClassLabs[i])
    ax[Nclass-1].set_xlabel('probability')
    plt.show()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def plot_confusion(Ypred, Ynum, ClassLabs):
    
       #----------------------------------------------------------------------
        #confusion matrix
        cm               = confusion_matrix(Ynum, Ypred)

        plt.figure(figsize = (8, 6))
        sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',\
                    xticklabels = ClassLabs, yticklabels = ClassLabs)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def my_timer(my_function):
    def get_args(*args,**kwargs):
        t1 = time.monotonic()
        results = my_function(*args,**kwargs)
        t2 = time.monotonic()
        dt = t2 - t1
        print("Total runtime: " + str(dt) + ' seconds')
        return results, dt
    return get_args