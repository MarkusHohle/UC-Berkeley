# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:03:02 2024

@author: MMH_user
"""
#standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pyclustering.utils.metric import *                           #for choosing between different metrics
from nltk.cluster.kmeans import KMeansClusterer                   #performs K-means
from sklearn.metrics import silhouette_samples, silhouette_score  #calculating silhouette coefficient
from sklearn import datasets                                      #we want to work with an internal data set

def PlotSetosa(X, Labels, Color):
    
    
    fig, ax = plt.subplots(figsize = (8,8))
    i = 0
    for species, color in zip(Labels, Color):
        idxs = np.arange(0,50) + 50*i
        i += 1
        ax.scatter(X[idxs,0], X[idxs,1], label = species, s = 50, color = color, alpha = 0.7)
        ax.legend()
    plt.show()


def KMeansExample():
    
    #1) Loading and Inspecting the Data
    iris = datasets.load_iris()
    print(iris.DESCR)
    
    Labels = iris.target_names
    X      = iris.data
    
    
    #2) Plotting the Data
    
    nClust     = 3  #Guessing the number of cluster
    rep        = 25 #The means are assigned randomly. In order to avoid getting stuck in a local minimum, we repeat the procedure 25 times 
                #and store the best result
    dist       = distance_metric(type_metric.EUCLIDEAN) #The features are meassured in cm, i. e. the correct distance to pick here is Euclidean
    Color      = ["#1B9E77", "#D95F02", "#7570B3"]
    
    for i in range(3):
        X2D = X[:,i:i+2]
        PlotSetosa(X2D, Labels, Color)
        
        
        #3) Kmeans

        my_model   = KMeansClusterer(nClust, distance = dist, repeats  = rep,\
                                     avoid_empty_clusters = True)
        PredLabels = my_model.cluster(X2D, assign_clusters = True)
        Center     = my_model.means()    
    
        CenterAr   = np.array(Center)
    
    
        #printing result
        PredLabels = np.array(PredLabels)
        
        fig, ax = plt.subplots(figsize=(8,8))
        for species_pred, color in zip(np.unique(PredLabels), Color):
            idxs = np.where(np.array(PredLabels) == species_pred)
            ax.scatter(X2D[idxs,0], X2D[idxs,1], label = species_pred,\
                   s = 50, color = color, alpha = 0.7)
        ax.legend()
        ax.scatter(CenterAr[:,0],CenterAr[:,1], marker = 'x', s = 200,\
                   color = 'k')
        plt.title('cluster assignment after k-means')
        plt.show()
        
        
        ax = sns.kdeplot(data = pd.DataFrame(X2D, columns = ['x', 'y']),\
                         x = 'x', y = 'y', cmap = 'Blues', fill=True)
        ax.scatter(CenterAr[:,0],CenterAr[:,1], marker = 'x', s = 200, color = 'k')
        plt.title('density plot')
        plt.show()
    
    #full 4D analysis
    
    Kmax = 15                 #calculate the silhouette coefficient for 2, 3, ... 15 cluster
    S    = np.zeros((Kmax-1)) #preparing matrix for storing the result
    
    
    for k in range(2, Kmax + 1):
        my_model = KMeansClusterer(k, distance = dist, repeats = 25, avoid_empty_clusters = True)
        Labels   = my_model.cluster(X, assign_clusters = True)
        S[k-2]   = silhouette_score(X, Labels)#calculating the silhouette coefficient (1st entry for k=2, but index = 0)
        
    plt.plot(np.arange(2,Kmax+1), S, c = 'k', linestyle = '-.', linewidth = 3)
    plt.xlabel('number of cluster')
    plt.ylabel('mean silhouette')
    plt.show()
    
    
    my_model = KMeansClusterer(3, distance = dist, repeats = 25, avoid_empty_clusters = True)
    Labels   = my_model.cluster(X, assign_clusters = True)
    
    Labels   = np.array(Labels)
    
    diff0 = Labels[:50]    - 0
    diff1 = Labels[50:100] - 1
    diff2 = Labels[100:]   - 2

    diff    = np.vstack((diff0, diff1, diff2))
    idx     = np.array(np.where(diff==0))
    _, col  = idx.shape
    accur   = col/150
    print(f'accuracy is:{accur:.3f}%')