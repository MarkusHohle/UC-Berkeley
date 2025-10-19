# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:17:11 2024

@author: MMH_user
"""

#pip install Tensorflow==2.14.0
#pip install keras-segmentation==0.3.0
import numpy as np
np.bool = np.bool_

import tensorflow as tf
import matplotlib.pyplot as plt

from keras_segmentation.models.unet import *
from keras_segmentation.predict     import predict, predict_multiple
from keras_segmentation.train       import verify_segmentation_dataset

#there is a bug in model.train when using internal augmentation. Fixing it via
import collections
collections.Iterable = collections.abc.Iterable 

class SegmentMyImages():
    
    def __init__(self, my_path = r"C:/my_path/data set/",\
                     n_classes = 51, input_height = 400, input_width = 600):
        
        #checkpoints is for saving the weights and recover model
        self.checkpoint_path = my_path + r"checkpoints/"

        #calling a pretrained segmentation CNN
        model = unet(n_classes     = n_classes,\
                         input_height  = input_height,\
                         input_width   = input_height)
            
        self.model     = model #for checkpoints later
        self.n_classes = n_classes
        self.my_path   = my_path


    def Training(self, Nepochs = 10):
        
        model     = self.model
                    
        model.train(
                train_images      = self.my_path + r"images_prepped_train/",
                train_annotations = self.my_path + r"annotations_prepped_train/",
                checkpoints_path  = self.checkpoint_path,
                do_augment              = True,
                gen_use_multiprocessing = True,
                auto_resume_checkpoint  = True,
                epochs = Nepochs)
        
        self.TrainedModel    = model
        
        
    def ApplyTrainedNetwork(self, *image_name):
        
        my_path = self.my_path
        model   = self.TrainedModel
        
        if not image_name:
            image_name = '0016E5_07965.png'
            
        out = model.predict_segmentation(\
              inp = my_path + "images_prepped_test/"  + image_name,\
              out_fname = my_path + "output_" + image_name)
            
        plt.imshow(out)
            
        return out
    
    
    def RecoverFromCheckpoint(self, *image_name):
        
        my_path = self.my_path
        
        #loading untrained CNN
        model = self.model
        
        if not image_name:
            image_name = '0016E5_07965.png'
            
        #calling input from checkpoints
        latest = tf.train.latest_checkpoint(self.checkpoint_path)
        model.load_weights(latest)
        
        out = model.predict_segmentation(\
              inp = my_path + "images_prepped_test/"  + image_name,\
              out_fname = my_path + "output_" + image_name)
        
        plt.imshow(out)

        return out





