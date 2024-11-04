# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:15:31 2024

@author: MMH_user
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, diff

def Bisection(my_fun: str = 'cos(x)-x', accur: float = 1/5000,\
              x_left: float = 0, x_right: float = 3):
    
    expr     = sympify(my_fun)#for example "cos(x)"
    x        = symbols('x')
    
    x_center = (x_right + x_left)/2
    
    y_left   = float(expr.subs({x: x_left}))
    y_center = float(expr.subs({x: x_center}))
    
    print(x_center)
    
    if abs(y_center) > accur:
        
        prod = y_left * y_center
        
        #1) zero lies in between: x_right --> x_center
        if prod < 0:
           x_center = Bisection(my_fun, accur, x_left, x_center)
            
        #2) zero is at the right: x_left --> x_center
        if prod > 0:
            x_center = Bisection(my_fun, accur, x_center, x_right)
            
    return(x_center)
    
    
    
    
    
    
    
    
    
    
    
    
    