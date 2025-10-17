# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 22:21:23 2025

@author: MMH_user
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""
USAGE:
    K = KMeansSegmentation()
    K.Segment()
"""


class KMeansSegmentation():
    
    def __init__(self, image: np.array = '574.jpg', k: int = 3):

        self.I = plt.imread(image)
        self.k = k
        
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        axs[0, 0].imshow(self.I)
        axs[0, 0].set_title('original image')
        
        axs[0, 1].imshow(self.I[:,:,0], cmap = 'gray')
        axs[0, 1].set_title('red color channel')
        
        axs[1, 0].imshow(self.I[:,:,1], cmap = 'gray')
        axs[1, 0].set_title('green color channel')
        
        axs[1, 1].imshow(self.I[:,:,2], cmap = 'gray')
        axs[1, 1].set_title('blue color channel')
        
        plt.tight_layout() # Adjust rect to make space for suptitle
        plt.show()
        
        
    def Segment(self, values: np.array = [0,0,0], label: int = 0):
        
        #flattening the image
        rows, cols, channels = self.I.shape
        x_flat               = self.I.reshape(rows * cols, channels)
        
        #running k-means
        kmeans     = KMeans(n_clusters = self.k)
        pxl_labels = kmeans.fit_predict(x_flat)#labels
        colors     = kmeans.cluster_centers_   #mean colors
        colors     = np.round(colors).astype(int) #for RGB
        
        Irec       = colors[pxl_labels].reshape(rows, cols, channels)
        Iseg       = pxl_labels.reshape(rows, cols)
        
        #any manipulation
        idx                = np.argwhere(pxl_labels == label)
        x_flat_copy        = x_flat.copy()
        x_flat_copy[idx,:] = values
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 5))
        
        axs[0].imshow(Iseg)
        axs[0].set_title('segmented image')
        
        axs[1].imshow(Irec)
        axs[1].set_title('reconstructed image')
        
        axs[2].imshow(x_flat_copy.reshape(rows, cols, channels))
        axs[2].set_title('manipulated image')
        plt.tight_layout()
        plt.show()
        
        
        
        
        
        
        
        
        
        