# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:22:57 2024

@author: MMH_user
"""

import numpy as np

class Layer_Dense:
    
    
    def __init__(self, n_inputs,n_neurons):
        
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases  = np.zeros((1,n_neurons))
        
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        self.inputs = inputs
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dinputs  = np.dot(dvalues,self.weights.T)
        self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)
        
        
###############################################################################
###############################################################################

class Activation_ReLU:
    
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)
        self.inputs  = inputs
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0#!!!
        
        
###############################################################################
###############################################################################

class Activation_Softmax:
    
    def forward(self, inputs):
        
        exp_values    = np.exp(inputs - np.max(inputs))
        probabilities = exp_values/np.sum(exp_values, axis = 1, \
                                          keepdims = True)
        self.output   = probabilities
        
    def backward(self, dvalues):
        
        self.dinputs = np.empty_like(dvalues)
        
        for i, (single_output, single_dvalues) in enumerate(zip(self.output,dvalues)):
            
            single_output = single_output.reshape(-1,1)
            
            jacobMatr = np.diagflat(single_output) - \
                        np.dot(single_output,single_output.T)
            
            self.dinputs[i] = np.dot(jacobMatr,single_dvalues)
        
###############################################################################
###############################################################################

class Loss:
    
    def calculate(self, output, y):
        #output is the output from the softmax layer
        #vector/matrix of the actual class from the training data
        
        sample_losses = self.forward(output,y)
        data_loss     = np.mean(sample_losses)
        
        return(data_loss)
        
###############################################################################
###############################################################################

class Loss_CategoricalCrossEntropy(Loss):
    
    def forward(self, prob_pred, y_true):
        
        Nsamples = len(prob_pred)
        y_pred_clipped = np.clip(prob_pred, 1e-7, 1-1e-7)
        
        #checking if we have one hot or sparse
        if len(y_true.shape) == 1:#sparse [0,2,3,1,1,0]
            correct_confidences = y_pred_clipped[range(Nsamples),y_true]
            
        elif len(y_true.shape) == 2:# one hot [[0,1,0], [1,0,0]]
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        
        return(negative_log_likelihoods)
    
    
    def backward(self, dvalues, y_true):
        
        Nsamples = len(dvalues)
        
        if len(y_true.shape) == 1:
            Nlabels = len(dvalues[0])
            y_true  = np.eye(Nlabels)[y_true]
        
        self.dinputs = - y_true/dvalues/Nsamples
        
###############################################################################
###############################################################################

class CalcSoftmaxLossGrad:
    
    def __init__(self):
        
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossEntropy()
        
    def forward(self, inputs, y_true):
        #inputs comes from the last dense layer (output) --> needed for 
        #softmax
        #y_true: vector/matrix of the true classes/labels
        self.activation.forward(inputs)
        self.output = self.activation.output
        
        return(self.loss.calculate(self.output, y_true))
    
    def backward(self, dvalues, y_true):
        #output from softmax layer
        Nsamples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        
        self.dinputs = dvalues.copy()
        #calculating the gradient
        self.dinputs[range(Nsamples),y_true] -= 1
        self.dinputs = self.dinputs/Nsamples
        
###############################################################################
###############################################################################

class Optimizer_SGD:
    
    def __init__(self, learning_rate = 0.1, decay = 0, momentum = 0):
        self.learning_rate         = learning_rate
        self.decay                 = decay
        self.current_learning_rate = learning_rate
        self.iterations            = 0
        self.momentum              = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate* \
                (1 / (1 + self.iterations * self.decay))
        
    def update_params(self, layer):
        
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentums -\
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums -\
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        
        else: 
            weight_updates = -self.current_learning_rate* layer.dweights
            bias_updates   = -self.current_learning_rate* layer.dbiases
        
        layer.weights += weight_updates
        layer.biases  += bias_updates

    def post_update_params(self):
        self.iterations += 1








