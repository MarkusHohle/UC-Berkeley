# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 22:22:11 2025

@author: MMH_user
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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