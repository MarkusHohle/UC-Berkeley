# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:20:32 2023

@author: MMH_user
"""

# =============================================================================
# usage:
#     
#     import numpy as np
#     import matplotlib.pyplot as plt
# 
#     x    = np.arange(1,10,1)
#     xint = np.arange(1,10,0.1)
# 
#     y = np.sin(x)
#     yint = Interpolate(x,xint,y,1)
# 
# 
#     plt.scatter(x,y)
#     plt.plot(xint,yint, 'k-')
# =============================================================================



import numpy as np

def Interpolate(x, xint, y, sigma):
    
    Dx    = np.tile(x.transpose(), (len(xint), 1))
    Dxint = np.tile(xint.transpose(), (len(x), 1))
    
    D     = Dx.transpose() - Dxint
    W     = np.exp(-(D**2)/(sigma**2))
    W     = W/np.sum(W)
    yint  = np.dot(W.transpose(), y)
    Scale = np.max(y)/np.max(yint)
    
    return yint*Scale