# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:22:27 2024

@author: MMH_user
"""

#from MyANN import *
from MyCNN import *
import matplotlib.pyplot as plt
import numpy as np
import random 

#calling the MNIST data set (numbers, black/white)
#pip install keras
#pip install tensorflow
from keras.datasets import mnist

class MyLeNet():
    
    def __init__(self):
        
        #loading and preparing data set
        
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        train_x = train_x.transpose(1,2,0)#turning matrix in right direction, 
        test_x  = test_x.transpose(1,2,0)#such that it fits our ANN
                                
        #our ANN wants 3D images
        S         = train_x.shape
        train_X3D = np.zeros((S[0],S[1],3,S[2]))
        
        for i in range(2):
            train_X3D[:,:,i,:] = train_x

        S         = test_x.shape
        test_X3D = np.zeros((S[0],S[1],3,S[2]))
        
        for i in range(2):
            test_X3D[:,:,i,:] = test_x

        
        self.train_X3D = train_X3D
        self.train_y   = train_y
        
        self.test_X3D  = test_X3D
        self.test_y    = test_y
        
        self.Ntot      = range(len(train_y))
        
        
    def RunTraining(self, minibatch_size = 512, iterations = 10, epochs = 500,\
                    learning_rate = 0.1, decay = 0.001, momentum = 0.5, saved_weights = False):
        
###############################################################################
        #initializing layers
        n_neuron        = 84
        n_class         = len(np.unique(self.train_y))
        n_inputs        = 480
        
        Conv1           = ConvLayer(5,5,6)
        Conv2           = ConvLayer(5,5,16)
        Conv3           = ConvLayer(5,5,120)
    
        AP1             = Average_Pool()
        AP2             = Average_Pool()
    
        T               = [Tanh() for i in range(4)]
        F               = Flat()
        
        dense1          = Layer_Dense(n_inputs, n_neuron)
        dense2          = Layer_Dense(n_neuron, n_class)
        
        loss_activation = CalcSoftmaxLossGrad()
        #initializing optimizer
        optimizer       = Optimizer_SGD(learning_rate, decay, momentum)
        
        if saved_weights:
        #######################################################################
            ###calling weights/biases
            ##if weights:
            Conv1.weights  = np.load('weightsC1.npy')
            Conv2.weights  = np.load('weightsC2.npy')
            Conv3.weights  = np.load('weightsC3.npy')
            
            Conv1.biases   = np.load('biasC1.npy')
            Conv2.biases   = np.load('biasC2.npy')
            Conv3.biases   = np.load('biasC3.npy')
            
            dense1.weights = np.load('weights1.npy')
            dense2.weights = np.load('weights2.npy')
            
            dense1.biases  = np.load('bias1.npy')
            dense2.biases  = np.load('bias2.npy')
        #######################################################################
        
###############################################################################
        
        Ntot = self.Ntot
        
        ie      = iterations*epochs
        Monitor = np.zeros((ie,3))
        ct      = 0
        
        for e in range(epochs):
            
            idx = random.sample(Ntot, minibatch_size)
            M   = self.train_X3D[:,:,:,idx]
            C   = self.train_y[idx]
            
            for it in range(iterations):
                
                Conv1.forward(M,0,1)
                T[0].forward(Conv1.output)
                AP1.forward(T[0].output,2,2)
                
                Conv2.forward(AP1.output,0,1)
                T[1].forward(Conv2.output)
                AP2.forward(T[1].output,2,2)
                
                Conv3.forward(AP2.output,2,3)
                T[2].forward(Conv3.output)

                #flattening
                F.forward(T[2].output)
                x = F.output

                dense1.forward(x)
                T[3].forward(dense1.output)
                dense2.forward(T[3].output)
                
                loss = loss_activation.forward(dense2.output, C)
                 
                predictions = np.argmax(loss_activation.output, axis = 1)
                if len(C.shape) == 2:
                    C = np.argmax(C ,axis = 1)
                accuracy = np.mean(predictions == C)
                   
                #backward passes
                loss_activation.backward(loss_activation.output, C)
                dense2.backward(loss_activation.dinputs)
                T[3].backward(dense2.dinputs)
                dense1.backward(T[3].dinputs)
                    
                F.backward(dense1.dinputs)
                
                T[2].backward(F.dinputs)
                Conv3.backward(T[2].dinputs)

                AP2.backward(Conv3.dinputs)
                T[1].backward(AP2.dinputs)
                Conv2.backward(T[1].dinputs)
                            
                AP1.backward(Conv2.dinputs)
                T[0].backward(AP1.dinputs)
                Conv1.backward(T[0].dinputs)
             
                optimizer.pre_update_params()
                    
                optimizer.update_params(dense1)
                optimizer.update_params(dense2)
                                    
                optimizer.update_params(Conv1)
                optimizer.update_params(Conv2)
                optimizer.update_params(Conv3)
                    
                optimizer.post_update_params()
                    
                Monitor[ct,0] = accuracy
                Monitor[ct,1] = loss
                Monitor[ct,2] = optimizer.current_learning_rate

                ct += 1                
       
                print(f'epoch: {e}, ' +
                     f'iteration: {it}, ' +
                     f'accuracy: {accuracy:.3f}, ' +
                     f'loss: {loss:.3f}, ' +
                     f'current learning rate: {optimizer.current_learning_rate:.5f}')
                
                
            #saving learnables
            ###################################################################
            np.save('weights1.npy', dense1.weights)
            np.save('weights2.npy', dense2.weights)
            
            np.save('bias1.npy', dense1.biases)
            np.save('bias2.npy', dense2.biases)
            
            np.save('weightsC1.npy', Conv1.weights)
            np.save('weightsC2.npy', Conv2.weights)
            np.save('weightsC3.npy', Conv3.weights)
            
            np.save('biasC1.npy', Conv1.biases)
            np.save('biasC2.npy', Conv2.biases)
            np.save('biasC3.npy', Conv3.biases)
            
            np.savetxt('Monitor.txt',Monitor)
            ###################################################################

            
        Vie   = np.arange(ie)
        VieIt = np.arange(epochs) *iterations
        
        fig, ax = plt.subplots(3, 1,sharex=True)
        ax[0].plot(Vie, Monitor[:,0]*100)
        for xh in VieIt:
            ax[0].axvline(x=xh,ymin=0, ymax=100, color = 'black')
        ax[0].set_ylabel('accuracy [%]')
        ax[1].plot(Vie, Monitor[:,1])
        for xh in VieIt:
            ax[1].axvline(x=xh,ymin=0, ymax=100, color = 'black')
        ax[1].set_ylabel('loss')
        ax[2].plot(Vie, Monitor[:,2])
        ax[2].set_ylabel(r'$\alpha$')
        ax[2].set_xlabel('epoch * iterations')
        
        
   
    def EvaluateMyLeNet(self, N = 50):
        
        #selecting test data set for evaluation
        test_X3D       = self.test_X3D
        test_y         = self.test_y
        
        idx            = random.sample(range(len(test_y)),N)
        
        M              = test_X3D[:,:,:,idx]
        C              = test_y[idx]
        
        
        #building LeNet forward part. Note: try super() to save code here : ) 
        n_neuron       = 84
        n_class        = len(np.unique(self.test_y))
        n_inputs       = 480
        
        Conv1          = ConvLayer(5,5,6)
        Conv2          = ConvLayer(5,5,16)
        Conv3          = ConvLayer(5,5,120)
    
        AP1            = Average_Pool()
        AP2            = Average_Pool()
    
        T              = [Tanh() for i in range(4)]
        F              = Flat()
        
        dense1         = Layer_Dense(n_inputs, n_neuron)
        dense2         = Layer_Dense(n_neuron, n_class)
        
        #loading weights
        Conv1.weights  = np.load('weightsC1.npy')
        Conv2.weights  = np.load('weightsC2.npy')
        Conv3.weights  = np.load('weightsC3.npy')
            
        Conv1.biases   = np.load('biasC1.npy')
        Conv2.biases   = np.load('biasC2.npy')
        Conv3.biases   = np.load('biasC3.npy')
            
        dense1.weights = np.load('weights1.npy')
        dense2.weights = np.load('weights2.npy')
            
        dense1.biases  = np.load('bias1.npy')
        dense2.biases  = np.load('bias2.npy')
        
        #run forward part
        Conv1.forward(M,0,1)
        T[0].forward(Conv1.output)
        AP1.forward(T[0].output,2,2)
        
        Conv2.forward(AP1.output,0,1)
        T[1].forward(Conv2.output)
        AP2.forward(T[1].output,2,2)
        
        Conv3.forward(AP2.output,2,3)
        T[2].forward(Conv3.output)

        F.forward(T[2].output)
        x = F.output

        dense1.forward(x)
        T[3].forward(dense1.output)
        dense2.forward(T[3].output)
        
        softmax = Activation_Softmax()
        softmax.forward(dense2.output)
        
        probabilities = softmax.output
        
        
        #######################################################################
        #some plots
        fig = plt.figure(figsize=(15, 7))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05,\
                            wspace=0.05)

        # plot the images: each image is 28x28 pixels
        for i in range(N):
            ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
            ax.imshow(M[:,:,0,i].reshape((28,28)),cmap=plt.cm.gray_r,\
                      interpolation='nearest')
            
            predclass = np.argmax(probabilities[i,:])
            trueclass = np.argmax(C[i])
            
            S = str(predclass)
          
            if predclass == trueclass:
                # label the image with the blue text
                P = str(round(probabilities[i,predclass],2))#probability
                ax.text(0, 3, S + ', P = ' + P, color = [0,128/255,0])
            else:
                # label the image with the red text
                ax.text(0, 3, S, color=[178/255,34/255,34/255])
                
        plt.savefig('evaluation results.pdf')
        plt.show()
        #######################################################################

        
        
        return(predclass, probabilities)