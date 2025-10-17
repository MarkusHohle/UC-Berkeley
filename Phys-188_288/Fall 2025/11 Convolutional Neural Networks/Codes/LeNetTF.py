# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:37:10 2023

@author: hohle
"""
#from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D

#from tensorflow.keras import datasets, models
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist


import tensorflow as tf
#import datetime
import random 
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf


####run the module like eg.
#import LeNetTF as L
#M = L.Model()
#M.Run()
#M.Eval()

###############################################################################
# LeNet model as discussed in the lecture
class MyLeNet(Sequential):
    
    def __init__(self, input_shape, nb_classes):
        super().__init__()

#layers of sequential NN
        self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same"))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(120, kernel_size=(5, 5), strides=(3, 3), activation='tanh', padding='valid'))
        self.add(Flatten())
        self.add(Dense(84, activation='tanh'))
        self.add(Dense(nb_classes, activation='softmax'))
        
#setting optimizer and learning schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                      initial_learning_rate = 1e-2,
                      decay_steps           = 10000,
                      decay_rate            = 0.98)

        sgd = tf.keras.optimizers.SGD(learning_rate = lr_schedule,\
                                      momentum = 0.9, nesterov = True)
            
        self.compile(optimizer = sgd,\
                    loss = categorical_crossentropy,\
                    metrics = ['accuracy'])
###############################################################################            

###############################################################################
#calling and preparing data set
class CallData:
    
    def CallMNIST():
        
        #######################################################################
        #loading and preparing training data
        (Train_x, Train_y), (Test_x, Test_y) = mnist.load_data()

        #adding new axis (to fit input shape required for ANN) and normalization (8 bit)
        Train_X3D = Train_x[:, :, :, np.newaxis]/255
        Test_X3D  = Test_x[:, :, :, np.newaxis]/255

        #y has to be turned in actual categorical, otherwise weird error message:
        #Shapes (none, 1) and (none, 10) are incompatible categorical_crossentropy
        #
        #
        
        num_classes = np.max(Test_y) + 1

        Train_y = to_categorical(Train_y, num_classes)
        Test_y  = to_categorical(Test_y, num_classes)

        #check
        Train_X3D.shape

        #######################################################################
        #split eval data 
        N = len(Train_y)


        Train_X3D = np.array(Train_X3D)

        idx     = random.sample(range(N), round(N*0.2))
        val_x   = Train_X3D[idx,:,:,:]
        train_x = Train_X3D[[i for i in range(N) if i not in idx],:,:]#somehow ~idx causes an error
        val_y   = Train_y[idx]
        train_y = Train_y[[i for i in range(N) if i not in idx]]

        #######################################################################

        
        return(num_classes,train_x,val_x,train_y,val_y,Test_X3D,Test_y)
###############################################################################

###############################################################################
    
class Model:
    
    def Run(self, epochs = 20):

        [num_classes,train_x,val_x,train_y,val_y,Test_X3D,Test_y]  = CallData.CallMNIST()
    
        model       = MyLeNet(train_x.shape[1:], num_classes)
        model.summary()

        print('running model...')
        history = model.fit(train_x, train_y,\
                  epochs=epochs, batch_size=512,\
                  validation_data=(val_x, val_y),\
                  verbose=1)

    # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('training results.pdf')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('training loss.pdf')
        plt.show()
        
        
        self.model = model

    
    def Eval(self):
        
        [num_classes,train_x,val_x,train_y,val_y,Test_X3D,Test_y]  = CallData.CallMNIST()
        model                                                      = self.model
        
        predict = model.predict(Test_X3D)

        fig = plt.figure(figsize=(15, 7))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

        #pick 50 images randomly
        N   = 50
        idx = random.sample(range(Test_y.shape[0]), N)

        # plot the images: each image is 28x28 pixels
        for i in range(N):
            ii = idx[i]
            ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
            ax.imshow(Test_X3D[ii,:,:,0].reshape((28,28)),cmap=plt.cm.gray_r, interpolation='nearest')
            
            predclass = np.argmax(predict[ii,:])
            trueclass = np.argmax(Test_y[ii,:])
            
            S = str(predclass)
          
            if predclass == trueclass:
                # label the image with the blue text
                P = str(round(predict[ii,predclass],2))#probability
                ax.text(0, 3, S + ', P = ' + P, color = [0,128/255,0])
            else:
                # label the image with the red text
                ax.text(0, 3, S, color=[178/255,34/255,34/255])
                
        plt.savefig('evaluation results.pdf')
        plt.show()
    
        