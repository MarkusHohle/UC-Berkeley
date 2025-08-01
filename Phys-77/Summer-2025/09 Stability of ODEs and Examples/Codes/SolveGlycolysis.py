# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 19:07:04 2024

@author: MMH_user
"""
import numpy as np
import matplotlib.pyplot as plt

#from scipy.integrate import odeint, solve_ivp
from Glycolysis import Glycolysis
from ode_solver import ode_solver

def SolveGlycolysis(Init, t_span, a, b):
    

    XY     = ode_solver(Glycolysis, Init, t_span, method = 'RK45',\
                        a = a, b = b)
        
    t = XY.t
    X = XY.y[0,:]
    Y = XY.y[1,:]
        
    #####################plotting result#######################################
    plt.plot(t,XY.y.transpose())
    plt.legend(['X = [ADP]', 'Y = [F6P]'])
    plt.xlabel('time')
    plt.show()
        
    plt.plot(X, Y)
    plt.plot(X[-1], Y[-1], 'rx', markersize=20)
    plt.title('phase diagram')
    plt.xlabel('[ADP]')
    plt.ylabel('[F6P]')
    
    return XY