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

    # ----- convergence plots -----
    fig, axs = plt.subplots(3, 1, figsize=(7, 10))

    axs[0].plot(range(1, len(Tracker)+1), Tracker[:, 0], '-x', lw=2, color=(0.8, 0.4, 0))
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel(r"$\lambda_N$")
    axs[0].set_title("Convergence of $λ_N$")

    axs[1].plot(range(1, len(Tracker)+1), Tracker[:, 1], '-x', lw=2, color=(0.8, 0.4, 0))
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel(r"$b_N$")
    axs[1].set_title("Convergence of $b_N$")

    axs[2].plot(range(1, len(Accur)+1), 100*Accur, '-x', lw=2, color=(0.8, 0.4, 0))
    axs[2].set_yscale("log")
    axs[2].set_xlabel("iteration")
    axs[2].set_ylabel("mean error [%]")
    axs[2].set_ylim([100*Accur.min()*0.8, 100*Accur.max()*1.2])
    plt.tight_layout()

    # ----- posterior for variance -----
    dv = vardat / N / 1000
    V = np.arange(np.finfo(float).eps, vardat*2, dv)

    gammaV = (1/gamma(aN)) * (bN**aN) * (1/V)**(aN-1) * np.exp(-bN/V)

    # remove NaNs and normalize
    mask    = ~np.isnan(gammaV)
    V       = V[mask]
    gammaV  = gammaV[mask]
    gammaVN = gammaV / np.trapz(gammaV, V)

    VarEst  = V[np.argmax(gammaVN)]

    plt.figure(figsize=(7, 8))
    plt.subplot(3,1,1)
    plt.plot(V, gammaVN, '-', lw=2, color=(0.8, 0.4, 0))
    plt.axvline(vardat, color="k", lw=2)
    plt.xlabel(r"$\sigma^2$")
    plt.ylabel(r"$P(\sigma^2|data)$")
    plt.title(f"Estimated variance: {VarEst:.3f}")
    plt.legend(["posterior for variance", f"sample variance (N={N})"])

    # ----- posterior for mean -----
    Mudat = xbar
    dm = abs(Mudat) / N / 1000
    Mu = np.arange(np.min(data), np.max(data), dm)

    normM = norm.pdf(Mu, loc=muN, scale=np.sqrt(1/lambN))
    MEst = Mu[np.argmax(normM)]

    if N < 15:
        counts, bins = np.histogram(data, bins='auto')
    else:
        counts, bins = np.histogram(data, bins=round(N/4))

    plt.subplot(3,1,2)
    plt.bar((bins[:-1] + bins[1:])/2, counts, width=np.diff(bins), edgecolor="k", fill=False)
    plt.ylabel("#")
    plt.legend(["data"])

    plt.subplot(3,1,3)
    plt.plot(Mu, normM, '-', lw=2, color=(0.8, 0.4, 0))
    plt.axvline(Mudat, color="k", lw=2)
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$P(\mu|data)$")
    plt.title(f"Estimated mean: {MEst:.2f}")
    plt.legend(["posterior for mean", f"sample mean (N={N})"])
    plt.tight_layout()

    # outputs
    #Mean = np.column_stack([Mu, normM])
    #Var  = np.column_stack([V, gammaVN])

    return None