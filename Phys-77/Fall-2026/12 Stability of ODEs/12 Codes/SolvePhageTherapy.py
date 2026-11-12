# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 19:07:04 2024

@author: MMH_user
"""
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


from mpl_toolkits.mplot3d import Axes3D

"""
    # USAGE:
    # %%%%eg generating Fig 3c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # t_span  = [0, 100]
    # B0      = 1e3
    # P0      = 1e5
    # I0      = 1
    # 
    # Init = [B0, P0, I0]
    # 
    # rates = np.zeros((10))
    # 
    # rates[0] = 1       # growth rate of beacteria at low densities
    # rates[1] = 100     # burst size phages
    # rates[2] = 5e-8    # adsorption rate phages
    # rates[3] = 1       # decay rate phages
    # rates[4] = 1e-6    # killing rate param for immune response
    # rates[5] = 1       # max growth rate of immune response
    # rates[6] = 1e9     # carrying capacity of bacteria
    # rates[7] = 1.5e6   # carrying capacity immune response
    # rates[8] = 1e4     # bact conc, when immune response is half its max
    # rates[9] = 1e8     # bact conc, at which immune resp is half as effective 
    # =============================================================================
    
    # =============================================================================
    # #also interesting:
    # rates[4] = 0     # no immune action on bacteria, Fig 3a
    # rates[2] = 2e-10 # generates Figs 4 a/b (dashed) 
    # 
    # #generates Figs 4 a/b (solid) 
    # Init[2]  = rates[7] 
    # rates[4] = 1e-6
    # Init[0]  = (rates[6]-rates[9])/2 + np.sqrt((rates[6]-rates[9])**2/4\
    # - rates[6]*rates[9]*rates[7]*rates[4]/rates[0])
    # 
    # also interesting:
    # rates[4] = 0     # no immune action on bacteria, Fig a
    # rates[2] = 3e-11 # generates Figs 4 c/d (dashed) 
    # 
    # generates Figs 4 c/d (solid) 
    # rates[4] = 1e-6
    # Init[0]  = (rates[6]-rates[9])/2 + np.sqrt((rates[6]-rates[9])**2/4\
    # - rates[6]*rates[9]*rates[7]*rates[4]/rates[0])
    #%%%%
    # =============================================================================
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
def PhageTherapy(Init, t, rates):
    
    B, P, I = Init
    #B: bacteria
    #P: phages
    #I: immune response
    
    #reaction rates 
    [r, beta, phi, w, epsilon, alpha, Kc, KI, KN, KD] = rates
# =============================================================================
#     r       = rates[1]  # growth rate of beacteria at low densities
#     beta    = rates[2]  # burst size phages
#     phi     = rates[3]  # adsorption rate phages
#     w       = rates[4]  # decay rate phages
#     epsilon = rates[5]  # killing rate param for immune response
#     alpha   = rates[6]  # max growth rate of immune response
#     
#     Kc      = rates[7]  # carrying capacity of bacteria
#     KI      = rates[8]  # carrying capacity immune response
#     KN      = rates[9]  # bact conc, when immune response is half its max
#     KD      = rates[10] # bact conc, at which immune resp is half as effective 
# =============================================================================

    
    
    #The model ODEs
    dB = r*B*(1-B/Kc) - phi*B*P - epsilon*I*B/(1+B/KD)

    dP = beta*phi*B*P - w*P
                                   # adding immune decay after infection
    dI = alpha*I*(1-I/KI)*B/(B+KN) # - 0.01 * I/(B**0.2 + 0.1)

    D = [dB, dP, dI]
    
    return D
###############################################################################

###############################################################################
def SolvePhageTherapy(Init, t_span, rates, my_ODEs = PhageTherapy):

    
    BPI = ode_solver(my_ODEs, Init, t_span, method = 'RK45',\
                        rates = rates)
        
    t   = BPI.t
    B   = BPI.y[0,:] #bacteria conc
    P   = BPI.y[1,:] #phage conc
    I   = BPI.y[2,:] #immune resp
        
    #####################plotting result#######################################
    plt.plot(t, B, 'g')
    plt.plot(t, P, 'b')
    plt.plot(t, I, 'k')
    plt.legend(['bacteria','phages','immune response'])
    plt.xlabel('time [hrs]')
    plt.ylabel('concentration [$ml^{-1}$]')
    plt.yscale('log')
    plt.ylim([0.7, 1.7*BPI.y.max()])
    plt.show()
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(B, P, I)
    ax.plot(B[-1], P[-1], I[-1], 'rx', markersize=20)
    plt.xlabel('bacteria [$ml^{-1}$]')
    plt.ylabel('phages [$ml^{-1}$]')
    ax.set_zlabel('immune response [$ml^{-1}$]')
    plt.title('phase diagram')
    plt.show()

    
    return BPI