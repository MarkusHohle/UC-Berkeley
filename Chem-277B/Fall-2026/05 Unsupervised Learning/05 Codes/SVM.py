# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:14:55 2023

@author: MMH_user
"""
import numpy as np
from sklearn import svm
from  PlotMySVMPred import PlotMySVMPred

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











