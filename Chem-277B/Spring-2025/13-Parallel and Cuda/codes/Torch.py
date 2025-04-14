# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:34:39 2024

@author: MMH_user
"""
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

########usage##################################################################
# from Torch import *

# t, Y_tnorm = GenerateData()

# n_neurons  = 100
# n_epochs   = 200
# dt_past    = 30
# dt_futu    = 10
# n_features = 1
# n_sample   = 1

# n_stack    = 3

# cut  = 180

# [X, Y]         = prepare_data(Y_tnorm, dt_past, dt_futu)
# TrainX, TrainY = X[:cut], Y[:cut]
# TestX,   TestY = X[cut:], Y[cut:]


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# #for now:
# device = 'cpu'

# from Torch import *
# M2 = TorchLSTM(device, dt_past, n_stack, n_features, dt_futu, n_neurons)
# M2.Run(TrainX, TrainY, n_epochs)#13sec
# M2.Evaluate(TestX, t, Y_tnorm)

# #now: same with CUDA:
# M3 = TorchLSTM('cuda', dt_past, n_stack, n_features, dt_futu, n_neurons)
# M3.Run(TrainX, TrainY, n_epochs)#4.5sec
# M3.Evaluate(TestX, t, Y_tnorm)
###############################################################################

#############helper functions##################################################
def my_timer(my_function):
        def get_args(*args,**kwargs):
            t1 = time.monotonic()
            results = my_function(*args,**kwargs)
            t2 = time.monotonic()
            dt = t2 - t1
            print("Total runtime: " + str(dt) + ' seconds')
            return results
        return get_args
###############################################################################

###############################################################################
def GenerateData():
    
    t_start = -50
    t_end   = 20
    incr    = 0.25
    
    t       = np.arange(t_start, t_end, incr)
    t       = t.reshape(len(t), 1)
    Y_t     = np.sin(t) + 0.1*np.random.randn(len(t), 1) + np.exp((t + 20)*0.05)
    
    plt.plot(t, Y_t)
    plt.title('complete series')
    plt.show()
    
    scaler  = MinMaxScaler(feature_range = (0, 1))
    Y_tnorm = scaler.fit_transform(Y_t)
    
    return t, Y_tnorm
###############################################################################

###############################################################################
def prepare_data(data, n_past, n_future):
    x, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        x.append(data[i - n_past:i])
        y.append(data[i:i + n_future])
    return np.array(x), np.array(y)
###############################################################################

#######actual code: the model##################################################
class LSTMModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        
        super(LSTMModel, self).__init__()
        
        # LSTM layer
        lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,\
                            batch_first = True)
        # Fully connected layer
        fc = nn.Linear(hidden_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        self.layer_dim  = layer_dim
        
        self.lstm       = lstm.to(device)
        self.fc         = fc.to(device)
        self.device     = device

    def forward(self, x):
        
        device = self.device
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        h0 = h0.to(device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = c0.to(device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Reshaping the outputs for the fully connected layer
        out = self.fc(out[:, -1, :])
        
        return out


###############################################################################


#######actual code: main#######################################################
class TorchLSTM():
    
    def __init__(self, device, dt_past, n_stack, n_features, dt_futu,\
                 n_neurons = 100):
    
        model     = LSTMModel(input_dim = n_features, hidden_dim = n_neurons,\
                              layer_dim = n_stack, output_dim = dt_futu,\
                              device = device)
        model     = model.to(device)
        
        print(model)
        
        self.model  = model
        self.device = device
        
        
    @my_timer    
    def Run(self, TrainX, TrainY, n_epochs = 100):
        
        model = self.model
        
        #reshaping so that PyTorch understands the shapes
        TrainY = TrainY[:,0]
        
        TrainX = TrainX.reshape((TrainX.shape[0], TrainX.shape[1]))
        TrainY = TrainY.reshape((TrainY.shape[0]))

        TrainX = torch.tensor(TrainX[:, :, None], dtype=torch.float32)
        TrainY = torch.tensor(TrainY[:, None], dtype=torch.float32)

        TrainX = TrainX.to(self.device)
        TrainY = TrainY.to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
        
        Loss   = np.zeros((n_epochs)) 
    

        torch.cuda.synchronize()

        # Training loop
        for epoch in range(n_epochs):
            
            outputs = model(TrainX)
            optimizer.zero_grad()
            loss = criterion(outputs, TrainY)
            loss.backward()
            optimizer.step()
        
            Loss[epoch] = loss.item()
        
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))
        
        torch.cuda.synchronize()
        
        
        self.model = model
        self.Loss  = Loss
        
    def Evaluate(self, TestX, t, Y_tnorm):
        
        TestX = TestX.reshape((TestX.shape[0], TestX.shape[1]))
        TestX = torch.tensor(TestX[:, :, None], dtype=torch.float32)
        TestX = TestX.to(self.device)

        
        model = self.model
        
        #plotting #############################################################
        plt.plot(self.Loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        #######################################################################
    
        # Predicted outputs
        PredY = model(TestX).detach().to('cpu').numpy()
        back  = PredY.shape[0]
        
        plt.plot(t, Y_tnorm, linewidth = 5)
        plt.plot(t[-back:],PredY[:,0])
        plt.legend(['actual data', 'prediction'])
        plt.fill_between([t[-back,0], t[-1,0]],\
                          0, 1, color = 'k', alpha = 0.1)
        plt.plot([t[-back,0], t[-back,0]], [0, 1],'k-',linewidth = 3)
        plt.show()
