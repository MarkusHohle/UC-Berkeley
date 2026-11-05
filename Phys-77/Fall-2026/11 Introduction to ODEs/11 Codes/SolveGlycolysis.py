# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 19:07:04 2024

@author: MMH_user
"""
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
    USAGE:
        Init   = [1,1]
        t_span = [0, 100]
        a      = 0.04
        b      = 0.4
        SolveGlycolysis(Init, t_span, a, b)

"""


###############################################################################
def ode_solver(ode_func, Init, t_span, method = 'RK45', **params):

    #avaiable integration methods: 
    #('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA').

    result = solve_ivp(fun = lambda t, y: ode_func(y, t, **params),\
                       t_span = t_span, y0 = Init, method = method,\
                       rtol = 1e-9, atol = 1e-9, max_step = 0.01)

    if result.success:
        print("Integration was successful.")
    else:
        print("Integration failed.")

    return result
###############################################################################

###############################################################################
def Glycolysis(Init, t, a, b):
    
# =============================================================================
#     #reaction rates (limit cycle) 
#     #a=0.05;
#     #b=0.5;
# 
#     #reaction rates (stationary state)  
#     #a=0.05;
#     #b=0.25;
# =============================================================================
    x = Init[0]
    y = Init[1]

    #The glycolysis ODEs
    dx = -x + a*y + (x**2)*y

    dy = b -a*y - (x**2)*y

    D = [dx, dy]
    
    return D
###############################################################################

###############################################################################
def SolveGlycolysis(Init, t_span, a, b, my_ODEs = Glycolysis):
    
    #function solves Glycolysis ODEs
    

    XY     = ode_solver(my_ODEs, Init, t_span, method = 'RK45',\
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