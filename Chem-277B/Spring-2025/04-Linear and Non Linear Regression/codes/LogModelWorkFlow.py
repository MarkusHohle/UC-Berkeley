# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:35:00 2023

@author: MMH_user
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#lin model
import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def AnalyzeMoleculesLogReg():
###############################################################################
#1) load and extract the data
    
    Train  = pd.read_csv("molecular_train_gbc_cat.csv")
    Test   = pd.read_csv("molecular_test_gbc_cat.csv")
    
###############################################################################
    
###############################################################################
#2) plotting the data
    
    out = sns.pairplot(Train, kind="kde", \
                       plot_kws={'color':[176/255,224/255,230/255]}, \
                       diag_kws={'color':'black'})
    out.map_offdiag(plt.scatter, color = 'black')
    
###############################################################################
    
###############################################################################
#3) scaling the data
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    TrainS = scaler.fit_transform(Train.drop('label', axis = 1))
    TestS  = scaler.transform(Test.drop('label', axis = 1))
    
    #scaling returns an array, but we need a dataframe for the fit routine
    TrainS = pd.DataFrame(TrainS, columns = Train.columns[:-1])
    TestS  = pd.DataFrame(TestS,  columns = Train.columns[:-1])
    
###############################################################################
#4) performing the fit
    
    X = sm.add_constant(TrainS)       # adding intercept
    Y = pd.get_dummies(Train['label'])# label into dummies
    
    my_model = sm.GLM(Y, X, family = sm.families.Binomial()).fit()
    my_model.summary()
    
###############################################################################
#5) evaluating the fit
    
    predProbs   = my_model.predict(sm.add_constant(TestS))
    Pred        = np.round(predProbs).astype(int) #assigments refer to non-toxic
    predictions = ['Non-Toxic' if i==1 else 'Toxic' for i in Pred]
    
    TestY       = Test['label']
    accuracy    = 100*(TestY == predictions).sum()/len(predictions)
    print(f'accuracy = {accuracy: .2f}%')
    
    #plotting confusion matrix-----------------------------------------------------
    L = ['Non-Toxic', 'Toxic']
    
    cm   = confusion_matrix(TestY, predictions, labels = L, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = L)
    disp.plot(cmap = 'gray')
    plt.show()
    #------------------------------------------------------------------------------
    
    #plotting entropy--------------------------------------------------------------
    PredProbs = np.vstack((predProbs, 1 - predProbs))
    
    fig, ax = plt.subplots(len(L), 1, sharex = True)
    fig.set_figheight(6)
    fig.subplots_adjust(hspace = 0.5)
    fig.suptitle('entropy')
    for i, l in enumerate(L):
        idx = [k for k, y in enumerate(TestY) if y == l]
        idx = np.array(idx)
        (value, where) = np.histogram(PredProbs[i,idx],\
                                      bins = np.arange(0,1,0.01),\
                                      density = True)
        w = 0.5*(where[1:] + where[:-1])
        ax[i].plot(w, value, 'k-')
        ax[i].set_ylabel('frequency')
        ax[i].set_title(l)
    ax[len(L)-1].set_xlabel('probability')
    plt.show()
    #------------------------------------------------------------------------------






