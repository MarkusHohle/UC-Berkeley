# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 00:27:55 2026

@author: MMH_user
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm   
from sklearn.metrics import confusion_matrix

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


def SVM(X2D, Y):
    
    outlinear   = svm.SVC(kernel = 'linear', C = 1, decision_function_shape = 'ovr')
    Fit         = outlinear.fit(X2D,Y)
    PredY       = outlinear.predict(X2D)
    accurlinear = np.round(100*(PredY == Y).sum()/len(Y))

    outrbf      = svm.SVC(kernel = 'rbf', gamma = 1, C = 1, decision_function_shape = 'ovr')
    Fit         = outrbf.fit(X2D,Y)
    PredY       = outrbf.predict(X2D)
    accurrbf    = np.round(100*(PredY == Y).sum()/len(Y))
    
    outpoly     = svm.SVC(kernel = 'poly', degree = 3, C = 1, decision_function_shape = 'ovr')
    Fit         = outpoly.fit(X2D,Y)
    PredY       = outpoly.predict(X2D)
    accurpoly   = np.round(100*(PredY == Y).sum()/len(Y))
        
        
    outsig      = svm.SVC(kernel = 'sigmoid', C = 1, decision_function_shape = 'ovr')
    Fit         = outsig.fit(X2D,Y)
    PredY       = outsig.predict(X2D)
    accursig    = np.round(100*(PredY == Y).sum()/len(Y))
      
    Acc   = [accurlinear, accurrbf, accurpoly, accursig]
    Model = [outlinear, outrbf, outpoly, outsig]
    
    for acc, m in zip(Acc, Model):
        PlotMySVMPred(X2D, m.fit(X2D, Y), Y, acc)


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

def FindMyFile(filename: str, ServerHardDiscPath: str = r"c:\Users\MMH_user\Desktop") -> str:
    """
    finds file of name "filename" anywhere in "ServerHardDiscPath" and returns complete path
    """
    for r,d,f in os.walk(ServerHardDiscPath):
        for files in f:
             if files == filename: #for example "MessyFile.xlsx"
                 file_name_and_path =  os.path.join(r,files)
                 return file_name_and_path
