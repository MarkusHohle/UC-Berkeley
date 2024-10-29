# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 21:02:15 2024

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

x     = np.random.uniform(-5, 5, (45))
left  = np.min(x)
right = np.max(x)
y     = -x*np.cos(4*x)*np.sin(x) + x 

plt.scatter(x, y, marker = 'x', c = 'k', label = 'actual data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('raw data')
plt.show()

###############################################################################
#polynomial interpolation
I    = interpolate.interp1d(x, y)

xint = np.arange(left, right, 0.1)#points we need to interpolate. 
                               #Note, poly interpolation only works with ref 
                               #points on both sides --> we cannot use the full
                               #interval.
yint = I(xint)

plt.plot(xint, yint, c = 'r', linewidth = 3, alpha = 0.3, label = 'interpolation')
plt.scatter(x, y, marker = 'x', c = 'k', label = 'actual data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('linear interpolation')
plt.show()


I    = interpolate.interp1d(x, y, kind = 2)
xint = np.arange(left, right, 0.1)#points we need to interpolate. 
                               #Note, poly interpolation only works with ref 
                               #points on both sides --> we cannot use the full
                               #interval.
yint = I(xint)

plt.plot(xint, yint, c = 'r', linewidth = 3, alpha = 0.3, label = 'interpolation')
plt.scatter(x, y, marker = 'x', c = 'k', label = 'actual data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('quadratic interpolation')
plt.show()


I    = interpolate.interp1d(x, y, kind = 3)
xint = np.arange(left, right, 0.1)#points we need to interpolate. 
                               #Note, poly interpolation only works with ref 
                               #points on both sides --> we cannot use the full
                               #interval.
yint = I(xint)

plt.plot(xint, yint, c = 'r', linewidth = 3, alpha = 0.3, label = 'interpolation')
plt.scatter(x, y, marker = 'x', c = 'k', label = 'actual data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('qubic interpolation')
plt.show()


###############################################################################
#spline interpolation
xint               = np.arange(-5, 5, 0.1)
#x needs to be sorted:
sorted_pairs       = sorted(zip(x, y))
x_sorted, y_sorted = zip(*sorted_pairs)

I    = interpolate.CubicSpline(x_sorted, y_sorted, extrapolate = 'periodic')
yint = I(xint)
plt.plot(xint, yint, c = 'r', linewidth = 3, alpha = 0.3, label = 'interpolation')
plt.scatter(x, y, marker = 'x', c = 'k', label = 'actual data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('qubic spline interpolation with periodic extrapolation')
plt.show()


I    = interpolate.CubicSpline(x_sorted, y_sorted, extrapolate = True)
yint = I(xint)
plt.plot(xint, yint, c = 'r', linewidth = 3, alpha = 0.3, label = 'interpolation')
plt.scatter(x, y, marker = 'x', c = 'k', label = 'actual data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('qubic spline interpolation with extrapolation')
plt.show()