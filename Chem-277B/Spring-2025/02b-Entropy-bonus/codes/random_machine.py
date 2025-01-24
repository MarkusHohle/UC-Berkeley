# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 02:55:48 2024

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt

#function illustrates the increase of entropy by rolling ONE die among N 
#dice each of M time steps by starting with a homogenious set and evolving
#with a poissonian time keeper (Gillespie alg)

def random_machine(N, M):
    
    I = 6 #number of states: I = 6 for a die

    #N dice, all set to state "three" for t=1
    Dice    = 3*np.ones((N,M))
    Entropy = np.zeros((M))
    
    Emax    = np.log(I)#max entropy
    e_calc  = np.zeros((I))
    
    #SETTING TIME
    #generate M random numbers between 0 and 1 (uniformly dist)
    R      =  np.random.uniform(0,1,(M,1))
    Tau    = -np.log(R)/N
    T      = np.cumsum(Tau)
    
    #generate M random numbers between 0 and N-1 (uniformly dist)
    #for choosing die for every time step
    R_dice = np.random.randint(0,N,(M))
    
    #SETTING STATE of one randomly choosen die
    #generate M random numbers between 1 and I (uniformly dist)
    R_state = np.random.randint(1,I+1,(M))
    #choose die
    for i in range(M):#over time
        idx_die           = R_dice[i]
        Dice[idx_die, i:] = R_state[i]
        
        #calculating entropy
        for j in range(I):
            e_calc[j] = (Dice[:,i] == j+1).sum()/N + 1e-300 #avoiding log(0)
            
            
        Entropy[i] =  -np.dot(e_calc,np.log(e_calc))
        
    
    #plotting histogram of states
    labels, counts = np.unique(Dice[:,-1], return_counts = True)
    plt.bar(labels, counts, align = 'center', width = 0.1, color = 'k')
    plt.gca().set_xticks(labels)
    plt.xlabel('states')
    plt.ylabel('#')
    plt.title('states after ' + str(M) + ' rolls with ' + str(N) + ' dice')
    plt.show()
    
    #plotting evolution of entropy
    plt.plot(T, Entropy, '-*', linewidth = 2, color = [0.8, 0.1, 0.2],\
             label = '$S(t)$')
    plt.plot([0, T[-1]], [Emax, Emax], 'k-', linewidth = 3, label = '$S_{max}$')
    plt.xlabel('time')
    plt.ylabel('total entropy')
    plt.title('after ' + str(M) + ' rolls with ' + str(N) + ' dice')
    plt.legend()
    plt.show()

    return Dice







