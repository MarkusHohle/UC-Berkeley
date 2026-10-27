# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 02:42:58 2024

@author: MMH_user
"""

import random          # for picking images randomly 
import numpy as np
import glob as gl      # for ls like in linux
import matplotlib.pyplot as plt

from os import *
from PIL import Image  # for resizing images
from scipy.ndimage import gaussian_filter

#usage:
    #my_path   = r'C:\my_folder\data set'
    #
    #folder_labels = my_path + r'\annotations_prepped_train'
    #folder_images = my_path + r'\images_prepped_train'
    #
    #ScaleAndAugmentMyImages(folder_images, folder_labels, folder_images, folder_labels)

def ScaleAndAugmentMyImages(source_folder_images, source_folder_labels,\
                            target_folder_images, target_folder_labels,\
                            image_format = '.png', remove_old_aug = "Yes"):
    #augments images by
    # - turning them by a random angle
    # - cropping them randomly
    # - rescaling the cropped image
    # - blurring 

    labels = gl.glob(source_folder_labels + r'\\' + '*' + image_format)
    images = gl.glob(source_folder_images + r'\\' + '*' + image_format)
    
    #only processing not augmented images in case folder contains augmented
    #images
    
    labels_ = [i for i in labels if not '_aug' in i]
    images_ = [i for i in images if not '_aug' in i]
    
    labels = labels_
    images = images_
    
    #removing old augmented images
    if remove_old_aug == "Yes":
        
        old_aug_label = gl.glob(target_folder_labels + r'\\' + '*' + image_format)
        old_aug_image = gl.glob(target_folder_images + r'\\' + '*' + image_format)
        
        [remove(i) for i in old_aug_label if '_aug' in i]
        [remove(i) for i in old_aug_image if '_aug' in i]
    
    Nima = len(labels)
    
    percent_old = 0
    
    for num, (i, l) in enumerate(zip(images,labels)):
        
        percent = np.round(100*num/Nima)
        
        #for rotation
        phi   = np.array(random.sample(range(360), 1))
        
        #for blurring
        sigma = np.array(random.sample(range(4), 1))
        
        I = Image.open(i)
        L = Image.open(l)
        I = I.rotate(phi)
        L = L.rotate(phi)
        
        #for cutting
        X  = I.size[0]
        Y  = I.size[1]
        xl = random.sample(range(int(0.1*X)), 1)[0]
        xr = random.sample(range(xl + 50, X), 1)[0]
        yl = random.sample(range(int(0.1*Y)), 1)[0]
        yr = random.sample(range(xl + 50, Y), 1)[0]
        
        #DO NOT apply gauss filter to labels because it changes the values! 
        I = gaussian_filter(I, float(sigma))

        L = np.array(L)
       
        Iscaled = I[xl:xr, yl:yr, :]
        Lscaled = L[xl:xr, yl:yr]
        
        plt.imsave(target_folder_labels + r'\\' + str(num) + '_aug' + image_format, Lscaled)
        plt.imsave(target_folder_images + r'\\' + str(num) + '_aug' + image_format, Iscaled)
        
        if not percent % 10 and percent != percent_old:
            percent_old = percent
            print(str(percent) + r'% of the images have been processed')
            
            
    print('done!')
    #return(labels, images)
    
    
    
    
    
    
    
    
    