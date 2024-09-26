# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 00:16:39 2024

@author: MMH_user
"""
import numpy as np
import matplotlib.pyplot as plt
#pip install umap-learn
import umap.umap_ as umap
from sklearn import datasets
from draw_umap import draw_umap


iris   = datasets.load_iris()
Labels = iris.target_names
X      = iris.data

Color = ["#1B9E77", "#D95F02", "#7570B3"]

newXY = umap.UMAP().fit_transform(X)

fig, ax = plt.subplots(figsize = (8,8))
i = 0
for species, color in zip(Labels, Color):
    idxs = np.arange(0,50) + 50*i
    i += 1
    ax.scatter(newXY[idxs,0], newXY[idxs,1], label = species, s = 50, color = color, alpha = 0.7)
ax.legend()
plt.show()




for i in range(10):
    nn = i + 5
    draw_umap(X, Color, Labels, n_neighbors = nn, title = str(nn) + ' neighbors')


for i in range(10):
    dist = i/10
    draw_umap(X, Color, Labels, min_dist = dist, title = 'dist = ' + str(dist))
    
    
for i in range(3):
    i += 1
    draw_umap(X, Color, Labels, n_components = i, title = str(i) + ' components')




