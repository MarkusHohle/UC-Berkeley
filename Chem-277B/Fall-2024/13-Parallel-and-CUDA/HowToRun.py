# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:25:35 2024

@author: MMH_user
"""

n_neurons  = 100
n_epochs   = 200
dt_past    = 30
dt_futu    = 10
n_features = 1
n_sample   = 1

n_stack    = 3# number of LSTMs

from prepare_data import prepare_data


from Keras import *

t, Y_tnorm = GenerateData()
[X, Y]     = prepare_data(Y_tnorm, dt_past, dt_futu)
cut        = int(np.round(0.6*Y_tnorm.shape[0]))

TrainX, TrainY = X[:cut], Y[:cut]
TestX,   TestY = X[cut:], Y[cut:]

M1 = KerasLSTM(dt_past, n_features, dt_futu, n_neurons)
M1.Run(TrainX, TrainY, n_epochs)#66 seconds
M1.Evaluate(TestX, t, Y_tnorm)


#using Torch
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#for now:
device = 'cpu'

from Torch import *
M2 = TorchLSTM(device, dt_past, n_stack, n_features, dt_futu, n_neurons)
M2.Run(TrainX, TrainY, n_epochs)#13sec
M2.Evaluate(TestX, t, Y_tnorm)

#now: same with CUDA:
M3 = TorchLSTM('cuda', dt_past, n_stack, n_features, dt_futu, n_neurons)
M3.Run(TrainX, TrainY, n_epochs)#4.5sec
M3.Evaluate(TestX, t, Y_tnorm)


