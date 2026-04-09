# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:21:25 2024

@author: MMH_user
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

#######usage###################################################################
# from Keras import *

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


# K = KerasLSTM(dt_past, n_features, dt_futu, n_neurons)
# K.Run(TrainX, TrainY)
# K.Evaluate(TestX, t, Y_tnorm)
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

#######actual code#############################################################
class KerasLSTM():
    
    def __init__(self, dt_past, n_features, dt_futu, n_neurons = 100):
        
        model = Sequential()

        model.add(LSTM(n_neurons, activation = 'tanh', return_sequences = True, input_shape = (dt_past, n_features)))
        model.add(LSTM(n_neurons, activation = 'relu', return_sequences = True))
        model.add(LSTM(n_neurons, activation = 'relu'))
        model.add(Dense(dt_futu))

        opt = optimizers.Adam()
        model.compile(loss = 'mean_squared_error', optimizer = opt)

        model.summary()
        
        self.model      = model
        self.dt_past    = dt_past
        self.dt_futu    = dt_futu
        self.n_features = n_features
        
        
    @my_timer
    def Run(self, TrainX, TrainY, n_epochs = 100):
        
        model = self.model
        
        self.out = model.fit(TrainX, TrainY, epochs = n_epochs,\
                        validation_split = 0.2, verbose = 2, shuffle = False)
    
    
    def Evaluate(self, TestX, t, Y_tnorm):
        
        out = self.out
        
        #plotting #############################################################
        plt.plot(out.history['loss'])
        plt.plot(out.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc = 'upper left')
        plt.savefig('training loss.pdf')
        plt.show()
        #######################################################################
        
        PredY = self.model.predict(TestX)
        back  = PredY.shape[0]
        
        plt.plot(t, Y_tnorm, linewidth = 3)
        plt.plot(t[-back:], PredY[:, self.dt_futu-1])
        plt.legend(['actual data', 'prediction'])
        plt.fill_between([t[-back,0], t[-1,0]], 0, 1, color = 'k', alpha = 0.1)
        plt.plot([t[-back,0], t[-back,0]], [0, 1], 'k-', linewidth = 3)
        plt.show()