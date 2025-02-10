# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 01:53:24 2024

@author: MMH_user
"""
import warnings 
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets   
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score  #calculating silhouette coefficient


def AnalyzeGMM():
    
    warnings.filterwarnings('ignore') 

    iris   = datasets.load_iris()
    Labels = iris.target_names
    X      = iris.data
    k      = len(Labels)
    
    
    my_model   = GaussianMixture(n_components = k, random_state = 0).fit(X)
    Center     = my_model.means_
    PredLabels = my_model.predict(X)
    
    
    Color      = ["#1B9E77", "#D95F02", "#7570B3"]
    
    for i in range(3):
        X2D      = X[:,i:i+2]
        Center2D = Center[:,i:i+2]
        
        fig, ax = plt.subplots(figsize=(8,8))
        for species_pred, color in zip(np.unique(PredLabels), Color):
            idxs = np.where(np.array(PredLabels) == species_pred)
            ax.scatter(X2D[idxs,0], X2D[idxs,1], label = species_pred,\
                   s = 50, color = color, alpha = 0.7)
        ax.legend()
        ax.scatter(Center2D[:,0],Center2D[:,1], marker = 'x', s = 200,\
                   color = 'k')
        plt.title('cluster assignment after GMM')
        plt.show()
    
    
    Kmax = 15                 #calculate the silhouette coefficient for 2, 3, ... 15 cluster
    S    = np.zeros((Kmax-1)) #preparing matrix for storing the result
    
    
    for k in range(2, Kmax + 1):
        my_model = GaussianMixture(n_components = k, random_state = 0).fit(X)
        Labels   = my_model.predict(X)
        S[k-2]   = silhouette_score(X, Labels)#calculating the silhouette coefficient (1st entry for k=2, but index = 0)
        
        if k == 3:
            Labels = np.sort(Labels)
            diff0  = Labels[:50]    - 0
            diff1  = Labels[50:100] - 1
            diff2  = Labels[100:]   - 2
    
            diff    = np.vstack((diff0, diff1, diff2))
            idx     = np.array(np.where(diff==0))
            _, col  = idx.shape
            accur   = col/150
            
        
    plt.plot(np.arange(2,Kmax+1), S, c = 'k', linestyle = '-.', linewidth = 3)
    plt.xlabel('number of cluster')
    plt.ylabel('mean silhouette')
    plt.show()
    
    print(f'\n accuracy for k = 3 is: {accur:.3f}%')



