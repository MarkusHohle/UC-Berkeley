# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 19:49:24 2026

@author: MMH_user
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



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
def plot_confusion(Y_Test, Ypred, ClassLabs):
    
    cm = confusion_matrix(Y_Test, Ypred)

    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ClassLabs)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def boot_strap_coeff(XS_Train_add, Ydumm, 
                     best_a1: float, best_a2: float, Nboot: int = 1000):
    
    Coefs       = [None]*Nboot
    L, _        = XS_Train_add.shape

    for i, _ in enumerate(range(Nboot)):
        
        idx      = np.random.choice(L, L, replace = True)
        X_boot   = XS_Train_add[idx,:]
        Y_boot   = Ydumm.iloc[idx]
        
        my_model_boot = sm.GLM(Y_boot, X_boot, family = sm.families.Binomial())
        
        fit_boot      = my_model_boot.fit_regularized(method = 'elastic_net', 
                                                      alpha  = best_a1, 
                                                      L1_wt  = best_a2)
        
        Coefs[i] = fit_boot.params

    Coefs = np.array(Coefs)

    # 95% CI
    lower = np.percentile(Coefs, 2.50, axis = 0)
    upper = np.percentile(Coefs, 97.5, axis = 0)
    
    return lower, upper

    # Coefbest = np.array(my_model_best.params)
    # CoefAll  = np.vstack((lower, Coefbest, upper)).transpose()
    # CoefDf   = pd.DataFrame(CoefAll, 
    #                         columns = ['95%/2 lower', 'best', '95%/2 upper'],
    #                         index   = ['const'] + list(X.columns))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def scan_regularization(my_model, 
                        XS_Test,
                        XS_Train,
                        Y_Test,
                        Y_Train,
                        alpha1_start: float = 0, dalpha1: float = 0.005, alpha1_stop: float = 0.3,
                        alpha2_start: float = 0, dalpha2: float =  0.01, alpha2_stop: float = 1.0):


    Alpha1 = np.arange(alpha1_start, alpha1_stop, dalpha1)
    Alpha2 = np.arange(alpha2_start, alpha2_stop, dalpha2)
    
    LA1    = len(Alpha1)
    LA2    = len(Alpha2)
    
    Fit    = np.zeros((LA1, LA2))
    Eval   = np.zeros((LA1, LA2))
    
    LTest,  _ = XS_Test.shape
    LTrain, _ = XS_Train.shape
    
    
    for i, a1 in enumerate(Alpha1):
        for j, a2 in enumerate(Alpha2):
            
            my_model_ElNet = my_model.fit_regularized(method = 'elastic_net', alpha = a1, L1_wt = a2)
            
            Ypred_ElNet    = my_model_ElNet.predict(sm.add_constant(XS_Test))
            ProbsR         = np.round(Ypred_ElNet)
            Ypred          = ['diabetes' if p== 1 else 'healthy'for p in ProbsR]
            
            Eval[i,j]      = sum(Ypred == Y_Test)/LTest
            
            
            Yfit_ElNet     = my_model_ElNet.predict(sm.add_constant(XS_Train))
            ProbsR         = np.round(Yfit_ElNet)
            Yfit           = ['diabetes' if p== 1 else 'healthy'for p in ProbsR]
            
            Fit[i,j]       = sum(Yfit == Y_Train)/LTrain
    
    
    TickA1 = np.arange(0, LA1)
    LabA1  = [f"{a: .3f}" for a in Alpha1]
    
    TickA2 = np.arange(0, LA2)
    LabA2  = [f"{a: .3f}" for a in Alpha2]
    
    sns.heatmap(Fit, cmap = 'Blues', annot = False)
    plt.title('fit accuracy')
    plt.xticks(TickA2[1::3], LabA2[1::3], rotation = 90)
    plt.yticks(TickA1[1::3], LabA1[1::3], rotation = 0)
    plt.xlabel(r"$\alpha_2$")
    plt.ylabel(r"$\alpha_1$")
    plt.show()
    
    
    sns.heatmap(Eval, cmap = 'Blues', annot = False)
    plt.title('evaluation accuracy')
    plt.xticks(TickA2[1::3], LabA2[1::3], rotation = 90)
    plt.yticks(TickA1[1::3], LabA1[1::3], rotation = 0)
    plt.xlabel(r"$\alpha_2$")
    plt.ylabel(r"$\alpha_1$")
    plt.show()
    
    idx = np.argmax(Eval)
    row, col = np.unravel_index(idx, Eval.shape)
    
    BestEvalAcc, BestAlpha1, BestAlpha2 = Eval[row, col], Alpha1[row], Alpha2[col]
    
    Diff = Fit - Eval 
    
    sns.heatmap(Diff, cmap = 'Blues', annot = False)
    plt.title('difference fit - eval accuracy')
    plt.xticks(TickA2[1::3], LabA2[1::3], rotation = 90)
    plt.yticks(TickA1[1::3], LabA1[1::3], rotation = 0)
    plt.xlabel(r"$\alpha_2$")
    plt.ylabel(r"$\alpha_1$")
    plt.show()
    
    
    my_model_best = my_model.fit_regularized(method = 'elastic_net',
                                             alpha  = BestAlpha1,
                                             L1_wt  = BestAlpha2)
    
    return my_model_best, BestEvalAcc, BestAlpha1, BestAlpha2 
    
    
    
    
    
    