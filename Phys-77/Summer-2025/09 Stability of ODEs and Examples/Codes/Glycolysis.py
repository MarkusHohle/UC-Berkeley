# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 18:59:38 2024

@author: MMH_user
"""

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