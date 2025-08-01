# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 19:51:26 2024

@author: MMH_user
"""
from scipy.integrate import solve_ivp


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