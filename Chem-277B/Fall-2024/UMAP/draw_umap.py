# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 00:23:00 2024

@author: MMH_user
"""
import matplotlib.pyplot as plt
import umap.umap_ as umap
import numpy as np



def draw_umap(data, Color, Labels, n_neighbors = 15, min_dist = 0.1,\
              n_components = 2, metric = 'euclidean', title = ''):
    
    fit = umap.UMAP(n_neighbors = n_neighbors, min_dist = min_dist,\
                    n_components = n_components, metric = metric)
        
        
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        
        i = 0
        for species, color in zip(Labels, Color):
            idxs = np.arange(0,50) + 50*i
            i += 1
            ax.scatter(u[idxs,0], range(len(u[idxs])), label = species,\
                       s = 50, color = color, alpha = 0.7)
        ax.legend()
        
        
    if n_components == 2:
        ax = fig.add_subplot(111)
        
        i = 0
        for species, color in zip(Labels, Color):
            idxs = np.arange(0,50) + 50*i
            i += 1
            ax.scatter(u[idxs,0], u[idxs,1], label = species, s = 50, color = color,\
                       alpha = 0.7)
        ax.legend()
        
        
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        
        i = 0
        for species, color in zip(Labels, Color):
            idxs = np.arange(0,50) + 50*i
            i += 1
            ax.scatter(u[idxs,0], u[idxs,1], u[idxs,2], label = species,\
                       s = 50, color = color,\
                       alpha = 0.7)
        ax.legend()
        
    plt.title(title, fontsize=18)