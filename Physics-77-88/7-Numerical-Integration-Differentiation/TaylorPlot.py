# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:35:32 2024

@author: MMH_user
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, diff

#usage:
#    PlotTaylorSeries(n_order = 10, my_fun = 'sin(x)', x_eval = 0)


def PlotTaylorSeries(my_fun: str = 'cos(x)', n_order: int = 5,\
                     x_eval: float = 0):
    
    expr   = sympify(my_fun)#for example "cos(x)"
    x      = symbols('x')
    
    #plotting the actual function
    x_plot = np.arange(x_eval - 7, x_eval + 7, 1/1000)
    y_plot = np.array([float(expr.subs({x: xx})) for xx in x_plot])
    
    plt.plot(x_plot, y_plot, 'k-', linewidth = 3, alpha = 0.5)
    
    #zeroth order
    y_eval  = float(expr.subs({x: x_eval}))
    y_plotn = y_eval*np.ones(x_plot.shape)
    
    #all other orders
    for n in range(n_order - 1):
        
        fxdx     = diff(expr,x)                 #performing derivative
        y        = float(fxdx.subs({x: x_eval}))#evaluating derivative at
                                                #x = x_eval
        expr     = fxdx
        
        y_plotn += (1/math.factorial(n+1)) * y * (x_plot - x_eval)**(n+1)
        
        #plotting
        transp = (n+1)/n_order
        plt.plot(x_plot, y_plotn, 'r-', linewidth = 3, alpha = transp,\
                 label = 'Order = ' + str(n+1))
    
    plt.xlabel('x')
    plt.ylim((np.min(y_plot) - 1.2*abs(np.min(y_plot)),\
              np.max(y_plot) + 1.2*abs(np.max(y_plot))))
    plt.ylabel(my_fun)
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5),\
               ncol = np.ceil(n_order/15))
    plt.scatter(x_eval, y_eval, marker = '*', s = 120, color = [0, 0, 0])
    plt.show()
    





