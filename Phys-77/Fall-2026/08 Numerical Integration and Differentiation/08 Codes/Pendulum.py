# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 22:56:36 2025

@author: MMH_user
"""
import numpy as np
import matplotlib.pyplot as plt
### Your code here
def Pendulum(x: float = 0, v: float = 1, N: int = 20000, dt: float = 0.001, k: float = 1, m: float = 1, gamma: float = 1):
    
    M  = np.zeros((N,3))

    a = - 4*(k/m)*x**3 - gamma*v 
    
    for n in range(N):
        x += v*dt + 0.5*a*dt**2
        v += a*dt
        a  = - 4*(k/m)*x**3 - gamma*v 
        
        M[n,:] = [x, v, a]

        if not n%100:
            
            if n > 600:
                plt.scatter(M[n-600,0], M[n-600,1], 30, marker = 'o', color = [0.8, 0.8, 0.8])
                plt.scatter(M[n-300,0], M[n-300,1], 30, marker = 'o', color = [0.5, 0.5, 0.5])
                plt.scatter(M[n-100,0], M[n-100,1], 30, marker = 'o', color = [0.2, 0.2, 0.2])
            plt.scatter(x, v, 30, marker = 'o', color = 'black')
            plt.title('after N = ' + str(n) + ' iterations')
            plt.xlabel('location')
            plt.ylabel('velocity')
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            plt.show()
            
    plt.plot(M[:,0], M[:,1], linewidth = 3, alpha = 0.2, color = 'black')
    plt.xlabel('location')
    plt.ylabel('speed') 
    plt.show()