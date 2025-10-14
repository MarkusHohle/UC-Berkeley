# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:38:04 2024

@author: MMH_user
"""
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
from MyANN import *

[x, y] = spiral_data(samples = 200, classes = 5)


def RunMyANN(x, y, N_neurons = 64, Nepoch = 10000, learning_rate = 0.2,\
             decay = 0.001, momentum = 0.9, plot = True):
    
    (samples, features) = x.shape
    nclasses            = np.max(y)+1

    dense1        = Layer_Dense(features, N_neurons)
    dense2        = Layer_Dense(N_neurons, nclasses)
    activation1   = Activation_ReLU()
    loss_function = CalcSoftmaxLossGrad()
    optimizer     = Optimizer_SGD(learning_rate, decay, momentum)
    
    N       = Nepoch
    Monitor = np.zeros((N,3))
    
    for epoch in range(N):
        
        dense1.forward(x)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_function.forward(dense2.output, y)
        
        predictions = np.argmax(loss_function.output, axis = 1)
        
        if len(y.shape) == 2:
                y = np.argmax(y,axis = 1)
                
        accuracy = np.mean(predictions == y)
        
        loss_function.backward(loss_function.output, y)
        dense2.backward(loss_function.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
        
        Monitor[epoch,0] = accuracy *100
        Monitor[epoch,1] = loss
        Monitor[epoch,2] = optimizer.current_learning_rate
        
        if not epoch % 100:
            print(f'epoch: {epoch}, ' + f'accuracy: {accuracy: .3f} ' +
                  f'loss: {loss: .3f}')
    
    np.save('weights1.npy', dense1.weights)
    np.save('weights2.npy', dense2.weights)
    
    np.save('bias1.npy', dense1.biases)
    np.save('bias2.npy', dense2.biases)
    

    if plot:

        fig, ax = plt.subplots(3,1, sharex= True)
        ax[0].plot(np.arange(N),Monitor[:,0]) 
        ax[0].set_ylabel('accuracy [%]')
        ax[1].plot(np.arange(N),Monitor[:,1]) 
        ax[1].set_ylabel('loss')
        ax[2].plot(np.arange(N),Monitor[:,2]) 
        ax[2].set_ylabel(r'learning rate $\alpha$')
        ax[2].set_xlabel('epoch')
        plt.xscale('log',base=10)
        plt.show()
        
        for cl in range(nclasses):
            idx  = np.argwhere(y == cl)
            idxp = np.argwhere(predictions == cl)
            
            color = np.random.uniform(0,1,(3,))
            
            plt.scatter(x[idx,0], x[idx,1], facecolors = color)
            plt.scatter(x[idxp,0], x[idxp,1], facecolors = 'none', edgecolors = color)





