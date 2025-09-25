# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 22:46:22 2025

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt

def EM__Example(Nit: int = 10, N_0: int = 100, mu_0: float = -3, sigma_0: float = 2,\
                               N_1: int = 200 , mu_1: float = 3, sigma_1: float = 1):
    # Generate data
    G0 = np.random.normal(mu_0, sigma_0, (N_0,))
    G1 = np.random.normal(mu_1, sigma_1, (N_1,))
    G  = np.hstack((G0, G1)).flatten()
    lG = len(G)

    # Init: random state assignment
    State = np.random.randint(0, 2, lG)

    idx0  = np.where(State == 0)[0]
    idx1  = np.where(State == 1)[0]

    x0    = G[idx0]
    x1    = G[idx1]

    # Initial probabilities
    P = np.zeros((lG, 2))
    P[idx0, 0] = 1
    P[idx1, 1] = 1

    P0 = len(idx0) / lG
    P1 = len(idx1) / lG

    # Initial means and stds
    mu0  = np.mean(x0)
    mu1  = np.mean(x1)
    sig0 = np.std(x0)
    sig1 = np.std(x1)

    M = []
    
    plt.figure()
    plt.hist(G0, bins=15, density=True, edgecolor="red", facecolor="none")
    plt.hist(G1, bins=15, density=True, edgecolor="blue", facecolor="none")
    plt.hist(x0, bins=15, density=True, color="red", alpha=0.2)
    plt.hist(x1, bins=15, density=True, color="blue", alpha=0.2)
    plt.title("initialization")
    plt.show()

    for i in range(Nit):

        # E-step (responsibilities)
        factor00 = P0 * np.exp(-((x0 - mu0) ** 2) / (2 * sig0 ** 2)) / np.sqrt(2 * np.pi * sig0 ** 2)
        factor01 = P1 * np.exp(-((x0 - mu1) ** 2) / (2 * sig1 ** 2)) / np.sqrt(2 * np.pi * sig1 ** 2)

        factor11 = P1 * np.exp(-((x1 - mu1) ** 2) / (2 * sig1 ** 2)) / np.sqrt(2 * np.pi * sig1 ** 2)
        factor10 = P0 * np.exp(-((x1 - mu0) ** 2) / (2 * sig0 ** 2)) / np.sqrt(2 * np.pi * sig0 ** 2)

        N0 = factor00 / (factor00 + factor01)
        N1 = factor11 / (factor11 + factor10)

        idx00 = np.where(N0 > 0.5)[0]   # state 0 (and assigned 0)
        idx01 = np.where(N0 <= 0.5)[0]  # state 1 (and assigned 0)
        idx10 = np.where(N1 <= 0.5)[0]  # state 0 (and assigned 1)
        idx11 = np.where(N1 > 0.5)[0]   # state 1 (and assigned 1)

        # Update indices
        idx0new = np.concatenate((idx0[idx00], idx1[idx10]))
        idx1new = np.concatenate((idx1[idx11], idx0[idx01]))

        Pnew = P.copy()
        Pnew[idx0new, 0] = 1
        Pnew[idx0new, 1] = 0
        Pnew[idx1new, 0] = 0
        Pnew[idx1new, 1] = 1

        # Update sets
        idx0 = idx0new
        idx1 = idx1new

        x0 = G[idx0]
        x1 = G[idx1]

        P0new = len(x0) / lG
        P1new = len(x1) / lG

        mu0new  = np.mean(x0) if len(x0) > 0 else mu0
        mu1new  = np.mean(x1) if len(x1) > 0 else mu1
        sig0new = np.std(x0) if len(x0) > 0 else sig0
        sig1new = np.std(x1) if len(x1) > 0 else sig1

        M.append([mu0new, mu0, mu1new, mu1, sig0new, sig0, sig1new, sig1, P0new, P0, P1new, P1])

        mu0, mu1 = mu0new, mu1new
        sig0, sig1 = sig0new, sig1new
        P0, P1 = P0new, P1new
        P = Pnew
        
        # Plotting
        plt.figure()
        plt.hist(G0, bins=15, density=True, edgecolor="red", facecolor="none")
        plt.hist(G1, bins=15, density=True, edgecolor="blue", facecolor="none")
        plt.hist(x0, bins=15, density=True, color="red", alpha=0.2)
        plt.hist(x1, bins=15, density=True, color="blue", alpha=0.2)
        plt.title("iteration " + str(i+1))
        plt.show()

    return None