# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 21:26:58 2024

@author: MMH_user
"""

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
    dI = alpha*I*(1-I/KI)*B/(B+KN) # - 0.001 * I/(B + eps);

    D = [dB, dP, dI]
    
    return D