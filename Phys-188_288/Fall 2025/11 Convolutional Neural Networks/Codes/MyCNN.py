# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:05:00 2022

@author: MMH_user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:44:41 2022

@author: hohle
"""
import random          # for picking images randomly 
import numpy as np
import glob as gl      # for ls like in linux
from PIL import Image  # for resizing images



class Read_Scale_Imds:
    
    def __init__(self, minibatch_size , ANN_size):
        
        self.xANN = ANN_size[0]
        self.yANN = ANN_size[1]
        
        self.minibatch_size = minibatch_size
        
        my_path   = r'C:\yourdata\pet pics'
        form      = '.jpg'

        path_dogs = my_path + '\Dog\\' + '*' + form
        path_cats = my_path + '\Cat\\' + '*' + form

        Cats      = gl.glob(path_cats)
        Dogs      = gl.glob(path_dogs)
        
        self.Ld   = len(Dogs)
        self.Lc   = len(Cats)
        
        D         = np.zeros((self.Ld))
        C         = np.zeros((self.Lc)) + 1
        
        self.All_classes = np.array((np.hstack((D,C))))
        self.All_imds    = np.hstack((Dogs,Cats))
        
    
    def Read_Scale(self):
        
        xANN = self.xANN
        yANN = self.yANN
        
        minibatch_size = self.minibatch_size
        
        idx  = random.sample(range(self.Ld + self.Lc), minibatch_size)
        
        classes = self.All_classes[idx]
        imds    = self.All_imds[idx]
        
        
        ImdsMatrix = np.zeros((xANN, yANN, 3, minibatch_size))
        
        for i in range(minibatch_size):
            I                   = Image.open(imds[i])
            Ire                 = I.resize((xANN, yANN))
            Iar                 = np.array(Ire)
            
            if len(Iar.shape) != 3:
                I3D        = np.zeros((xANN, yANN, 3))
                I3D[:,:,0] = Iar
                I3D[:,:,1] = Iar
                I3D[:,:,2] = Iar
                Iar        = I3D
            
            ImdsMatrix[:,:,:,i] = Iar
            
        ImdsMatrix.astype(np.uint8)
        
        return(ImdsMatrix, classes)
            
###############################################################################
# convolution
###############################################################################

class ConvLayer:
    #code inspired by Fisseha Berhane
    def __init__(self, xKernShape = 3, yKernShape = 3, Kernnumber = 10):
                    
            #kernel sizes
            self.xKernShape = xKernShape
            self.yKernShape = yKernShape
            self.Kernnumber = Kernnumber
        
            #creating kernels
            self.weights    = 0.1*np.random.randn(xKernShape,yKernShape,Kernnumber)
            self.biases     = np.zeros((1, self.Kernnumber))
            
            #L2 regularization penalizing large weights
            L2 = 1e-2
            self.weights_L2 = L2
            self.biases_L2  = L2
            
            
    def forward(self, M, padding = 0, stride = 1):
    
    #getting shapes and channels
    #note: numChans referes to incomming matrix, Kernnumber to output matrix

        [xImgShape, yImgShape, numChan, numImds] = M.shape
    
        xK         = self.xKernShape
        yK         = self.yKernShape
        NK         = self.Kernnumber
        
        b          = self.biases
            
        #storage for backprop
        self.padding = padding
        self.stride  = stride
            
        xOutput = int(((xImgShape - xK + 2 * padding)/stride) + 1)
        yOutput = int(((yImgShape - yK + 2 * padding)/stride) + 1)
            
        #------------------------------------------------------------------
        W = np.nan_to_num(self.weights)
        #initializing empty output matrix
        output = np.zeros((xOutput,yOutput,numChan,NK,numImds))

        imagePadded = np.zeros(((xImgShape + padding*2),\
                                (yImgShape + padding*2),numChan,NK,numImds))
                        
        for k in range(NK):
            imagePadded[int(padding):int(padding + xImgShape),\
                        int(padding):int(padding + yImgShape),:,k,:]= M
                
        #######################################################################
        #according to http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
        #not every of the six feature maps of C1 are connected to each of the
        #16 feature maps in C2 --> need a filter
        
        if numChan == 6 & NK == 16:
            filt = np.array([[1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1],\
                             [1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1],\
                             [1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1],\
                             [0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1],\
                             [0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1],\
                             [0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1]])
        else:
            filt = np.ones([numChan,NK])
            
        self.filt = filt


        for i in range(numImds):# loop over number of images
            currentIm_pad = imagePadded[:,:,:,:,i]# Select ith padded image
            for c in range(numChan):# loop over channels
                for k in range(NK): #loop over filter
                
                  if filt[c,k] == 1:
                
                    for y in range(yOutput):# loop over vert axis of output
                        for x in range(yOutput):# loop over hor axis of output

                               # finding corners of the current "slice" 
                               y_start = y*stride
                               y_end   = y_start + yK
                               x_start = x*stride 
                               x_end   = x_start + xK
                    
                               #selecting the current part of the image
                               current_slice = currentIm_pad[x_start:x_end,\
                                                         y_start:y_end,c,k]
                                
                                   #the actual conv part
                               s = np.multiply(current_slice, W[:,:,k])
                                    
                               output[x, y, c, k, i] = np.sum(s) +\
                                                          b[0,k].astype(float)
                                                          

        output = output.sum(2)                                        


                                                 
        self.output = np.nan_to_num(output)
        self.input  = M
        self.impad  = imagePadded
            
            
            
    def backward(self, dvalues):
        
        filt       = self.filt
        
        biases     = self.biases
        weights    = self.weights
        
        stride     = self.stride
        padding    = self.padding
    
        xK         = self.xKernShape
        yK         = self.yKernShape
        NK         = self.Kernnumber
        
        imagePadded = self.impad
        
        S = imagePadded.shape
        
        dinputs    = np.zeros((S[0],S[1],S[2],S[4]))
        dbiases    = np.zeros(biases.shape)
        dweights   = np.zeros(weights.shape)
        
        xd         = dvalues.shape[0]
        yd         = dvalues.shape[1]
        numChan    = S[2]
        numImds    = dvalues.shape[3]
        
        #defining matrix for dinputs: note: has do be de-padded at the end
        imagePadded = imagePadded[:,:,:,0,:]#was the same input for all the k
            
        for i in range(numImds):# loop over number of images
            currentIm_pad = imagePadded[:,:,:,i]# Select ith padded image
            for k in range(NK):# loop over kernels (= #filters)
               for c in range(numChan):# loop over channels of incomming data
                  
                   if filt[c,k] ==1:
                 
                    for y in range(yd):# loop over axis of output
                         for x in range(xd):# loop over axis of output
                        
                         
                            # finding corners of the current "slice" (≈4 lines)
                            y_start = y*stride
                            y_end   = y_start + yK
                            x_start = x*stride 
                            x_end   = x_start + xK
                                
                            sx      = slice(x_start,x_end)
                            sy      = slice(y_start,y_end)
                        
                            current_slice = currentIm_pad[sx,sy,c]
    
                            dweights[:,:,k]    += current_slice * dvalues[x,y,k,i]
                            dinputs[sx,sy,c,i] += weights[:,:,k]* dvalues[x,y,k,i]
                        
                
               dbiases[0,k] += np.sum(np.sum(dvalues[:,:,k,i],axis=0),axis=0)
        
        dinputs       = dinputs[padding:S[0]-padding,padding:S[1]-padding,:,:]
        self.dinputs  = dinputs
        
        #derivatives from L2 regularization leads to multiplication with 2
        self.dbiases  = dbiases  + 2* self.biases_L2 *self.biases
        self.dweights = dweights + 2* self.weights_L2 *self.weights
        
###############################################################################
# different pooling options
###############################################################################
class Average_Pool:
    
    def forward(self, M, stride = 1, KernShape = 2):
        
        [xImgShape, yImgShape, numChan, numImds] = M.shape
        
        self.inputs = M
        
        #maxpool usually over nxn matrix
        xK = KernShape
        yK = KernShape
        
        xOutput = int(((xImgShape - xK) / stride) + 1)
        yOutput = int(((yImgShape - yK) / stride) + 1)
        
        imagePadded = M
        #output matrix after max pool
        output = np.zeros((xOutput,yOutput,numChan,numImds))
        
        for i in range(numImds):# loop over number of images
            currentIm_pad = imagePadded[:,:,:,i]# select ith padded image
            for y in range(yOutput):# loop over vert axis of output
                for x in range(xOutput):# loop over hor axis of output
                    for c in range(numChan):# loop over channels (= #filters)
                    
                    # finding corners of the current "slice" 
                        y_start = y*stride
                        y_end   = y_start + yK
                        x_start = x*stride 
                        x_end   = x_start + xK
                        
                        sx      = slice(x_start,x_end)
                        sy      = slice(y_start,y_end)
                    
                        current_slice = currentIm_pad[sx,sy,c]
                        
                        #actual average pool
                        slice_mean         = float(current_slice.mean())
                        output[x, y, c, i] = slice_mean
                        
        #storing info, also for backpropagation
        self.xKernShape = xK
        self.yKernShape = yK
        self.output     = output
        self.impad      = imagePadded
        self.stride     = stride
    
    def backward(self, dvalues):
        
        xd = dvalues.shape[0]
        yd = dvalues.shape[1]
        
        numChan = dvalues.shape[2]
        numImds = dvalues.shape[3]
        
        imagePadded = self.impad
        dinputs     = np.zeros(imagePadded.shape)
        Ones        = np.ones(imagePadded.shape)#for backprop
        
        stride  = self.stride
        xK      = self.xKernShape
        yK      = self.yKernShape
        
        Ones    = Ones/xK/yK # normalization that came from average pool
        
        #assigning dvalues to corresponding max values. 
        for i in range(numImds):# loop over number of images
            for y in range(yd):# loop over vert axis of output
                for x in range(xd):# loop over hor axis of output
                    for c in range(numChan):# loop over channels (= #filters)
                    
                        # finding corners of the current "slice" 
                        y_start = y*stride
                        y_end   = y_start + yK
                        x_start = x*stride 
                        x_end   = x_start + xK
                            
                        sx      = slice(x_start,x_end)
                        sy      = slice(y_start,y_end)
                            
                        dinputs[sx,sy,c,i]  += Ones[sx,sy,c,i]*dvalues[x,y,c,i]
                            
        
        self.dinputs = dinputs

###############################################################################
###############################################################################
class Min_Pool:
    
    def forward(self, M, stride = 1, KernShape = 2):
        
        [xImgShape, yImgShape, numChan, numImds] = M.shape
        
        self.inputs = M
        
        xK = KernShape
        yK = KernShape

        xOutput          = int((xImgShape - xK)/stride + 1)
        yOutput          = int((yImgShape - yK)/stride + 1)
        
        imagePadded      = M
        
        output           = np.zeros((xOutput, yOutput, numChan, numImds))
        
        StoreMin         = np.zeros((xK*yK, 2, xOutput*yOutput, numChan, numImds))
        StoreNMin        = np.zeros((xOutput*yOutput, numChan, numImds))
        
        imagePadded_copy = imagePadded.copy()*0
        
        for i in range(numImds):
            currentIm_pad = imagePadded[:,:,:,i]
            for c in range(numChan):
                ct = 0
                for y in range(yOutput):
                    for x in range(xOutput):
                        ct += 1
                        
                        #finding corners of current slice
                        y_start = y*stride
                        y_end   = y_start + yK
                        x_start = x*stride
                        x_end   = x_start + xK
                        
                        sx = slice(x_start, x_end)
                        sy = slice(y_start, y_end)
                        
                        current_slice      = currentIm_pad[sx,sy,c]
                        slice_min          = float(current_slice.min())
                        output[x, y, c, i] = slice_min
                        
                        (xsm, ysm) = np.where(current_slice == slice_min)
                        
                        #more than one pixel can be the max in a slice
                        for ii, (xx, yy) in enumerate(zip(xsm, ysm)):
                             StoreMin[ii,0,ct-1,c,i] = int(xx)
                             StoreMin[ii,1,ct-1,c,i] = int(yy)
                        StoreNMin[ct-1,c,i]       = ii+1
                        
                             
                        imagePadded_copy[sx, sy, c, i] += np.equal(\
                                    currentIm_pad[sx,sy,c], slice_min).astype(float)
                            
        mask            = imagePadded_copy
                        
                        
        self.xKernShape = xK
        self.yKernShape = yK
        self.output     = output
        self.mask       = mask
        self.stride     = stride
        self.StoreMin   = StoreMin
        self.StoreNMin  = StoreNMin
        
        
    def backward(self, dvalues):
        
        [xd, yd, numChan, numImds] = dvalues.shape

        StoreMin  = self.StoreMin
        StoreNMin = self.StoreNMin
        stride    = self.stride
        
        dinputs = np.zeros(self.inputs.shape)
        
        for i in range(numImds):
            for c in range(numChan):
                ct = 0
                for y in range(yd):
                    for x in range(xd):
                        ct += 1
                        
                        #finding corners of current slice
                        y_start = y*stride
                        x_start = x*stride
                        
                        Nmin = int(StoreNMin[ct-1, c, i])
                        
                        for ii in range(Nmin):
                            xm = int(StoreMin[ii,0, ct-1, c, i])
                            ym = int(StoreMin[ii,1, ct-1, c, i])
                            
                            dinputs[x_start + xm, y_start + ym, c, i] =\
                                dvalues[x,y,c,i]
  
        self.dinputs = dinputs
        
###############################################################################
###############################################################################
class Max_Pool:
    
    def forward(self, M, stride = 1, KernShape = 2):
        
        [xImgShape, yImgShape, numChan, numImds] = M.shape
        
        self.inputs = M
        
        xK = KernShape
        yK = KernShape

        xOutput          = int((xImgShape - xK)/stride + 1)
        yOutput          = int((yImgShape - yK)/stride + 1)
        
        imagePadded      = M
        
        output           = np.zeros((xOutput, yOutput, numChan, numImds))
        
        StoreMax         = np.zeros((xK*yK, 2, xOutput*yOutput, numChan, numImds))
        StoreNMax        = np.zeros((xOutput*yOutput, numChan, numImds))
        
        imagePadded_copy = imagePadded.copy() * 0
        
        for i in range(numImds):
            currentIm_pad = imagePadded[:,:,:,i]
            for c in range(numChan):
                ct = 0
                for y in range(yOutput):
                    for x in range(xOutput):
                        ct += 1
                        
                        #finding corners of current slice
                        y_start = y*stride
                        y_end   = y_start + yK
                        x_start = x*stride
                        x_end   = x_start + xK
                        
                        sx = slice(x_start, x_end)
                        sy = slice(y_start, y_end)
                        
                        current_slice      = currentIm_pad[sx,sy,c]
                        slice_max          = float(current_slice.max())
                        output[x, y, c, i] = slice_max
                        
                        (xsm, ysm) = np.where(current_slice == slice_max)
                        
                        #more than one pixel can be the max in a slice
                        for ii, (xx, yy) in enumerate(zip(xsm, ysm)):
                             StoreMax[ii,0,ct-1,c,i] = int(xx)
                             StoreMax[ii,1,ct-1,c,i] = int(yy)
                        StoreNMax[ct-1,c,i]       = ii+1
                        
                             
                        imagePadded_copy[sx, sy, c, i] += np.equal(\
                                    currentIm_pad[sx,sy,c], slice_max).astype(float)
                            
        mask      = imagePadded_copy
                        
                        
        self.xKernShape = xK
        self.yKernShape = yK
        self.output     = output
        self.mask       = mask
        self.stride     = stride
        self.StoreMax   = StoreMax
        self.StoreNMax  = StoreNMax
        
        
    def backward(self, dvalues):
        
        [xd, yd, numChan, numImds] = dvalues.shape

        StoreMax  = self.StoreMax
        StoreNMax = self.StoreNMax
        stride    = self.stride
        
        dinputs = np.zeros(self.inputs.shape)
        
        for i in range(numImds):
            for c in range(numChan):
                ct = 0
                for y in range(yd):
                    for x in range(xd):
                        ct += 1
                        
                        #finding corners of current slice
                        y_start = y*stride
                        x_start = x*stride
                        
                        Nmax = int(StoreNMax[ct-1, c, i])
                        
                        for ii in range(Nmax):
                            xm = int(StoreMax[ii,0, ct-1, c, i])
                            ym = int(StoreMax[ii,1, ct-1, c, i])
                            
                            dinputs[x_start + xm, y_start + ym, c, i] =\
                                dvalues[x,y,c,i]
  
        self.dinputs = dinputs
        
###############################################################################
# different activation options
###############################################################################
class Sigmoid:
        
    def forward(self, M):
        
        sigm        = np.clip(1/(1 + np.exp(-M)), 1e-7, 1 - 1e-7)
        self.output = sigm
        self.inputs = sigm #needed for back prop
            
    def backward(self, dvalues):
        
        sigm         = self.inputs
        deriv        = np.multiply(sigm, (1 - sigm))
        self.dinputs = np.multiply(deriv, dvalues)
        

###############################################################################
###############################################################################
class Tanh:
    
    def forward(self, M):
        tanh        = np.tanh(M)
        self.output = tanh
        self.inputs = M
    
    def backward(self, dvalues):
        deriv        = 1 - self.output**2
        deriv        = np.nan_to_num(deriv)
        self.dinputs = np.multiply(deriv, dvalues)
###############################################################################
#
###############################################################################

class Activation_ReLU:
    
    def forward(self, inputs):
        self.output  = np.maximum(0,inputs)
        self.inputs  = inputs
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0#ReLU derivative

###############################################################################
# Flattening
###############################################################################
class Flat:
        
    def forward(self, M):
        
        self.inputs = M
            
        [xImgShape, yImgShape, numChan, numImds] = M.shape
            
        #turning each image into a vector of length L
        L = xImgShape*yImgShape*numChan
        
        #not sure if reshape keeps the order between(!) different images
        output = np.zeros((numImds,L))
        for i in range(numImds):
            output[i,:] = M[:,:,:,i].reshape((1,L))
            
        self.output  = output
        
    def backward(self, dvalues):
        
        [xImgShape, yImgShape, numChan, numImds] = np.shape(self.inputs)
        dinputs = np.zeros((xImgShape, yImgShape,numChan, numImds))
        
        for i in range(numImds):
            dinputs[:,:,:,i] = dvalues[i,:].reshape((xImgShape, yImgShape,\
                                                     numChan))
        
        self.dinputs = dinputs

###############################################################################
# dense layer
###############################################################################

class Layer_Dense():
    
    def __init__(self, n_inputs, n_neurons):
        #note: we are using randn here in order to see if neg values are 
        #clipped by the ReLU
        #import numpy as np
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
        
        L2 = 1e-2
        self.weights_L2 = L2
        self.biases_L2  = L2
        
#passing on the dot product as input for the next layer, as before
    def forward(self, inputs):
        self.output  = np.dot(inputs, self.weights) + self.biases
        self.inputs  = inputs#we're gonna need for backprop
        
    def backward(self, dvalues):
        #gradients
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)
        self.dinputs  = np.dot(dvalues, self.weights.T)
        
        self.dbiases  = self.dbiases  + 2* self.biases_L2 *self.biases
        self.dweights = self.dweights + 2* self.weights_L2 *self.weights
     
###############################################################################
#softmax for predicted probabilities
###############################################################################
        
class Activation_Softmax:
  
    def forward(self,inputs):
        self.inputs = inputs
        exp_values  = np.exp(inputs - np.max(inputs, axis = 1,\
                                      keepdims = True))#max in order to 
                                                       #prevent overflow
        #normalizing probs
        probabilities = exp_values/np.sum(exp_values, axis = 1,\
                                      keepdims = True)  
        self.output   = probabilities                                                
    
    def backward(self, dvalues):
        #just initializing a matrix
        self.dinputs = np.empty_like(dvalues)
        
        for i, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
            
            single_output   = single_output.reshape(-1,1)
            jacobMatr       = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobMatr, single_dvalues)
            
###############################################################################
# loss and gradient
###############################################################################
class Loss:
     
    
    def calculate(self, output, y):
         
         sample_losses = self.forward(output, y)
         data_loss     = np.mean(sample_losses)
         return(data_loss)
    
    
class Loss_CategoricalCrossEntropy(Loss): 
                       #y_pred is not the predicted y, it is its 
                       #probability!!
     def forward(self, y_pred, y_true):
         samples = len(y_pred)
         #removing vals close to zero and one bco log and accuracy
         y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
         
         #now, depending on how classes are coded, we need to get the probs
         if len(y_true.shape) == 1:#classes are encoded as [[1],[2],[2],[4]]
             correct_confidences = y_pred_clipped[range(samples), y_true]
         elif len(y_true.shape) == 2:#classes are encoded as
                                    #[[1,0,0], [0,1,0], [0,1,0]]
             correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
         #now: calculating actual losses
         negative_log_likelihoods = -np.log(correct_confidences)
         return(negative_log_likelihoods)
         
     def backward(self, dvalues, y_true):
         Nsamples = len(dvalues)
         Nlabels  = len(dvalues[0])
         #turning labels into one-hot i. e. [[1,0,0], [0,1,0], [0,1,0]], if
         #they are not
         if len(y_true.shape) == 1:
            #"eye" turns it into a diag matrix, then indexing via the label
            #itself
            y_true = np.eye(Nlabels)[y_true]
         #normalized gradient
         self.dinputs = -y_true/dvalues/Nsamples
        
        
class CalcSoftmaxLossGrad():
    
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossEntropy()
        
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output#the probabilities
        #calculates and returns mean loss
        return(self.loss.calculate(self.output, y_true))
        
    def backward(self, dvalues, y_true):
        Nsamples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        self.dinputs = dvalues.copy()
        #calculating normalized gradient
        self.dinputs[range(Nsamples), y_true] -= 1
        self.dinputs = self.dinputs/Nsamples
        
        
class Optimizer_SGD:
    #initializing with a default learning rate of 0.1
    def __init__(self, learning_rate = 1, decay = 0, momentum = 0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0
        self.momentum              = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1/ (1 + self.decay*self.iterations))
        
    def update_params(self, layer):
        
        #if we use momentum
        if self.momentum:
            
            #check if layer has attribute "momentum"
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)
                
            #now the momentum parts
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        else:
            
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates   = -self.current_learning_rate * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases  += bias_updates
        
    def post_update_params(self):
        self.iterations += 1
