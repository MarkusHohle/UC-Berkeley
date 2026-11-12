# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 03:22:46 2025

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma

"""
usage:
data = np.random.normal(3, 1, (50,))
Var_Bayes_Example(data)
"""

def Var_Bayes_Example(data):
    """
    Applies variational Bayes (VB) to estimate the posterior pdf
    of the mean (mu) and variance (sigma^2) of the input data.

    Parameters
    ----------
    data : array-like
        Input data vector.

    Returns
    -------
    Mean : ndarray
        2D array with grid of mu values and their posterior density.
    Var : ndarray
        2D array with grid of variance values and their posterior density.
    """

    CI = 0.68

    data   = np.asarray(data)
    N      = len(data)
    xbar   = np.mean(data)
    x2bar  = np.mean(data**2)
    vardat = np.var(data, ddof=1)

    accur  = 400.0

    # hyperparameters (uninform priors)
    lamb0  = np.finfo(float).eps
    mu0    = np.finfo(float).eps
    a0     = np.finfo(float).eps
    b0     = np.finfo(float).eps
    lambN  = np.random.rand()
    bN     = np.random.rand()

    aN  = a0 + (N + 1) / 2
    muN = (lamb0 * mu0 + N * xbar) / (lamb0 + N)

    Tracker = []
    Accur   = []
    i = 0

    while accur > 1e-7 and i < 1000:
        Tracker.append([lambN, bN])
        Accur.append(accur)

        bN_new = (b0 + 0.5 * ((lamb0 + N) * (1/lambN + muN**2)
                  - 2 * (lamb0 * mu0 + N * xbar) * muN
                  + N * x2bar + lamb0 * mu0**2))

        lambN_new = (lamb0 + N) * aN / bN_new

        accur = np.mean([ abs(bN - bN_new) / abs(bN), abs(lambN - lambN_new) / abs(lambN)])

        lambN, bN = lambN_new, bN_new
        
        i += 1

    Tracker = np.array(Tracker)
    Accur   = np.array(Accur)


    # ----- posterior for variance -----
    dv = vardat / N / 1000
    V = np.arange(np.finfo(float).eps, vardat*2, dv)

    gammaV = (1/gamma(aN)) * (bN**aN) * (1/V)**(aN-1) * np.exp(-bN/V)

    # remove NaNs and normalize
    mask    = ~np.isnan(gammaV)
    V       = V[mask]
    gammaV  = gammaV[mask]
    gammaVN = gammaV / np.trapezoid(gammaV, V)

    whereg  = np.argmax(gammaVN)#getting index and value of most likely q and Pu
    maxv    = V[whereg]
    
    
    ###########################################################################
    #determining errors from actual integral
    #right tail
    d1     = 0
    intv1  = 0
    d2     = 1
    intv2  = 0
    
    sumintv = intv1 + intv2
    
    while (sumintv < CI*0.999): #I leave a margin here in order to account for 
                                #numerical inacurracy
        if (whereg+d1<len(V)):
            
            d1     = d1 + 1
            indi   = np.arange(whereg, whereg+d1,1)
            intv1  = sum(gammaVN[indi])*dv
    #left tail
        if (whereg+d2>0):
            
            d2      = d2 - 1
            indi    = np.arange(whereg+d2,whereg,1)
            intv2   = sum(gammaVN[indi])*dv
            
        sumintv = intv1 + intv2

    idxU    = whereg + d1
    VUP     = V[idxU]
    idxL    = whereg+d2
    VLO     = V[idxL]
    diff_up = VUP - maxv
    diff_do = maxv - VLO
    
    VarianceResult = [VLO, maxv, VUP, diff_do, diff_up]
    
    
    #plt.plot(V,gammaVN)
    
###############################################################################

    # ----- posterior for mean -----
    Mudat = xbar
    dm = abs(Mudat) / N / 1000
    Mu = np.arange(np.min(data), np.max(data), dm)

    normM = norm.pdf(Mu, loc=muN, scale=np.sqrt(1/lambN))
    
    wherem = np.argmax(normM)
    maxm   = Mu[wherem]
        
   
    ###########################################################################
    #determining errors from actual integral
    #right tail
    d1     = 0
    intv1  = 0
    d2     = 1
    intv2  = 0
    
    sumintv = intv1 + intv2
    
    while (sumintv < CI*0.999): #I leave a margin here in order to account for 
                                #numerical inacurracy
        if (wherem+d1<len(Mu)):
            
            d1     = d1 + 1
            indi   = np.arange(wherem, wherem+d1,1)
            intv1  = sum(normM[indi])*dm
    #left tail
        if (wherem+d2>0):
            
            d2      = d2 - 1
            indi    = np.arange(wherem+d2,wherem,1)
            intv2   = sum(normM[indi])*dm
            
        sumintv = intv1 + intv2

    idxU    = wherem + d1
    MUP     = Mu[idxU]
    idxL    = wherem+d2
    MLO     = Mu[idxL]
    diff_up = MUP - maxm
    diff_do = maxm - MLO
    
    MeanResult = [MLO, maxm, MUP, diff_do, diff_up]
    
    #plt.plot(Mu,normM)
###############################################################################



    return MeanResult, VarianceResult