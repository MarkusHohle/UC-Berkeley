# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 01:05:02 2024

@author: MMH_user
"""
import matplotlib.pyplot as plt
from MyANN import *

read_and_scale = Read_Scale_Imds(5, [250,250])
[M, C]         = read_and_scale.Read_Scale()

MP1   = Max_Pool()
MP2   = Max_Pool()
MP3   = Max_Pool()
RL1   = Activation_ReLU()
RL2   = Activation_ReLU()
RL3   = Activation_ReLU()
Conv1 = ConvLayer(3,3,5)
Conv2 = ConvLayer(5,5,4)
Conv3 = ConvLayer(2,2,4)


Conv1.forward(M,0,1)
RL1.forward(Conv1.output)
MP1.forward(RL1.output,2,3)

Conv2.forward(MP1.output,2,3)
RL2.forward(Conv2.output)
MP2.forward(RL2.output,3,3)

Conv3.forward(MP2.output,2,3)
RL3.forward(Conv3.output)
MP3.forward(RL3.output,1,2)

plt.imshow(Conv1.output[:,:,0,4]) 
plt.show()
plt.imshow(MP1.output[:,:,0,4])
plt.show()
plt.imshow(Conv2.output[:,:,0,4])
plt.show()
plt.imshow(MP2.output[:,:,0,4])
plt.show()
plt.imshow(Conv3.output[:,:,0,4])
plt.show()
plt.imshow(MP3.output[:,:,0,4])
plt.show()

##########################################

MP3.backward(MP3.output)
RL3.backward(MP3.dinputs)
Conv3.backward(RL3.dinputs)

MP2.backward(Conv3.dinputs)
RL2.backward(MP2.dinputs)
Conv2.backward(RL2.dinputs)

MP1.backward(Conv2.dinputs)
RL1.backward(MP1.dinputs)
Conv1.backward(RL1.dinputs)

plt.imshow(Conv1.dinputs[:,:,0,4]) 
plt.show()
plt.imshow(MP1.dinputs[:,:,0,4])
plt.show()
plt.imshow(Conv2.dinputs[:,:,0,4])
plt.show()
plt.imshow(MP2.dinputs[:,:,0,4])
plt.show()
plt.imshow(Conv3.dinputs[:,:,0,4])
plt.show()
plt.imshow(MP3.dinputs[:,:,0,4])
plt.show()
