# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 02:36:01 2025

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
usage:
data = np.random.normal(3, 1, (50,))
Gibbs_NormGamma_Example(data)

compare to
Var_Bayes_Example(data)
"""


def Gibbs_NormGamma_Example(data: np.array, Ntot: int = 5000, Nrecord: int = 4000):
    
    """
    Function performs Gibbs sampling for a Normal likelihood with unknown
    mean and variance.
    Prior: Normal-Gamma (conjugate) 
    dataset: data, 1D np.array
    
    in order to derive the distribution of the mean, given data and the 
    variance given data (compare to Var_Bayes_Example.py)
    The code performs Ntot total sampling steps, but means and variances will 
    be stored from the last Nrecord samples to make sure that the iterative
    sampling process has converged (after "burn-in")
    """
    
    N = Ntot - Nrecord
    
    n = len(data)
    s = np.sum(data)
    
    x_bar     = np.mean(data)
    sigm_head = np.var(data)
    
    # hyperparameters (uninform priors)
    lamb0 = np.finfo(float).eps
    mu0   = np.finfo(float).eps
    a0    = np.finfo(float).eps
    b0    = np.finfo(float).eps

    
    #initial guesses for precission lambda and mean mu
    lam       = 1
    mu        = 1
    
    #Tracker   = np.zeros((Ntot,2))
    #Accur     = np.zeros((Ntot,))
    
    Means     = np.zeros((Nrecord,))
    Vars      = np.zeros((Nrecord,))
    
    for i in range(Ntot):
        
        L_lam = (lamb0 + n)*lam
        M_lam = (lamb0*mu0 + s)/(lamb0 + n)
        
        mu    = np.random.normal(M_lam,np.sqrt(1/L_lam))
        
        S     = np.sum((data - mu)**2)
        A_mu  = a0 + n/2
        B_mu  = b0 + 0.5*S
        
        lam   = np.random.gamma(A_mu, 1/B_mu)
        
        if i >= N:
            Means[i-N] = mu
            Vars[i-N]  = 1/lam
        
    valm, wherem = np.histogram(Means)
    valv, wherev = np.histogram(Vars)
    
    MEst = wherem[np.argmax(valm)]
    VEst = wherev[np.argmax(valv)]
    
    plt.figure(figsize=(7, 8))
    plt.subplot(2,1,1)
    sns.histplot(Vars, bins=30, kde=True, stat='density', color='skyblue', edgecolor='skyblue')
    plt.axvline(sigm_head, color="k", lw=2)
    plt.xlabel(r"$\sigma^2$")
    plt.ylabel(r"$P(\sigma^2|data)$")
    plt.title(f"Estimated variance: {VEst:.3f}")
    plt.legend(["sampled posterior for variance", f"sample variance (N={n})"])
    
    plt.subplot(2,1,2)
    sns.histplot(Means, bins=30, kde=True, stat='density', color='skyblue', edgecolor='skyblue')
    plt.axvline(x_bar, color="k", lw=2)
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$P(\mu|data)$")
    plt.title(f"Estimated mean: {MEst:.2f}")
    plt.legend(["sampled posterior for mean", f"sample mean (N={n})"])
    plt.tight_layout()
   



    