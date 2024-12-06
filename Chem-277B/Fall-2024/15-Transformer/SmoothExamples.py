# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 21:02:15 2024

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy import interpolate

x     = np.random.uniform(-5, 5, (145))
noise = np.random.normal(0,2,(145))
y     = -x*np.sin(x) + x

#plt.scatter(x, y, marker = 'x', c = 'k', label = 'data', alpha = 0.3)
plt.scatter(x, y + noise, marker = '.', c = 'k', label = 'data plus noise')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('raw data')
plt.show()

###############################################################################
#own interpolation
from SmoothGaussKernel import Interpolate as OwnSmooth

sigma   = [0.2, 0.5, 1, 2]
xsample = np.arange(-5,5,0.001)


for s in sigma:
    
    ySmooth = OwnSmooth(x, xsample, y + noise, s)
    plt.plot(xsample, ySmooth, '-', linewidth = 3, alpha = 0.3,\
             label = '$\sigma$ = ' + str(s))

plt.scatter(x, y, marker = 'x', c = 'k', label = 'data', alpha = 0.3)
plt.scatter(x, y + noise, marker = '.', c = 'k', label = 'data plus noise')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('smoothed')
plt.show()
