# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 08:58:07 2022

@author: MMH_user
"""
"""
function performs Bayesian Model Selection (Chapt 4.3, D. S. Sivia, Data 
Analysis) with D_1 vs D_2 example

input: i = 0, 2, ... 5 refering to the data set we wanna look at

output:
rho    = P(data comes from ONE population regarding mean and variance|data)/
        P(data comes from TWO different populations regarding mean and 
        variance|data)

p      = ordinary p-val from two-sample, two-tailed, unpooled t-test 
     (as comparison)

Pmeans = P(mean_D_1>mean_D_2|data)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from scipy import stats #for standard t-test
from scipy.integrate import trapezoid, quad

def my_gauss(x,v,m):
    return  np.exp(-0.5 * ((x-m)**2)/v)


class mean_var_analysis_example:
    
    def __init__(self, N_1: int = 10, N_2: int = 20, mu_1: float = 3.0,\
                       mu_2: float = 1.0, sigma_1: float = 2,\
                       sigma_2: float = 1):
        
        #generating randomnormal dist data
        
        self.D_1 = np.random.normal(mu_1, sigma_1, (N_1,))
        self.D_2 = np.random.normal(mu_2, sigma_2, (N_2,))

        #creating a histogram
        sns.histplot(self.D_1, color = "skyblue", label = "data set 1", kde = True)
        sns.histplot(self.D_2, color = "red",     label = "data set 2", kde = True)
        plt.legend()
        plt.show()
        
        #needed for analysis:
        self.N_1 = N_1
        self.N_2 = N_2
        self.N   = N_1 + N_2
        self.D   = np.hstack((self.D_1, self.D_2))
        
        #plausible prior constrains about means and var, here just the data range
        self.mu_max_D  = np.max(self.D)
        self.mu_min_D  = np.min(self.D)
        self.std_max_D = self.mu_max_D - self.mu_min_D
        self.std_min_D = 0
            
        self.mu_max_D_1 = self.mu_max_D
        self.mu_min_D_1 = self.mu_min_D
        
        self.mu_max_D_2 = self.mu_max_D
        self.mu_min_D_2 = self.mu_min_D
        
        self.std_max_D_1 = self.std_max_D
        self.std_min_D_1 = 0
        
        self.std_max_D_2 = self.std_max_D
        self.std_min_D_2 = 0
        
    ###########################################################################

    ###########################################################################
    
    def T_Test(self, alternative: str = 'two-sided'):
        # alternative{'two-sided', 'less', 'greater'}
        
 
        #1) ordinary t-test
        [s, p_val] = stats.ttest_ind(self.D_1, self.D_2, equal_var = False, alternative = 'two-sided')
        
        return s, p_val
    
    ###########################################################################

    ###########################################################################

    def ModelSelection(self):
        #calculates odds raio rho = P(same dist)/P(different dists) 

        Nstep  = 200
    
        x      = np.linspace(self.mu_min_D, self.mu_max_D, Nstep+1)
        y      = np.linspace(self.std_min_D, self.std_max_D, Nstep+1)
    
        xx, yy = np.meshgrid(x, y)
    
        #integral over likelihood function
        #we need to run a loop here, since there is the operation sum(All -xx) for
        #all D, to be improved
        zz = np.zeros(np.shape(xx))
    
        for i in range(len(x)):
            for j in range(len(y)):
                zz[i,j] = np.exp(-0.5*np.sum(((self.D - x[i])/(y[j] + 1e-16))**2))/ \
                          ((np.sqrt(2*math.pi*y[j]**2)**self.N) + 1e-16)
        
        #actual integration
        ID = trapezoid(trapezoid(zz,xx),yy[:,1])
        
        #now:same for D1 and D2
        
        #D1
        x        = np.linspace(self.mu_min_D_1, self.mu_max_D_1, Nstep+1)
        y        = np.linspace(self.std_min_D_1, self.std_max_D_1, Nstep+1)
    
        [xx, yy] = np.meshgrid(x, y)
        zz       = np.zeros(np.shape(xx))
    
        for i in range(len(x)):
            for j in range(len(y)):
                zz[i,j] = np.exp(-0.5*np.sum(((self.D_1 - x[i])/(y[j] + 1e-16))**2))/ \
                    ((np.sqrt(2*math.pi*y[j]**2)**self.N_1) + 1e-16)
        
        ID1 = trapezoid(trapezoid(zz,xx),yy[:,1])
    
        #D2
        x        = np.linspace(self.mu_min_D_2, self.mu_max_D_2, Nstep+1)
        y        = np.linspace(self.std_min_D_2, self.std_max_D_2, Nstep+1)
    
        [xx, yy] = np.meshgrid(x, y)
    
        zz       = np.zeros(np.shape(xx))
        
        for i in range(len(x)):
            for j in range(len(y)):
                zz[i,j] = np.exp(-0.5*np.sum(((self.D_2 - x[i])/(y[j] + 1e-16))**2))/ \
                    ((np.sqrt(2*math.pi*y[j]**2)**self.N_2) + 1e-16)
            
        ID2 = trapezoid(trapezoid(zz,xx),yy[:,1])


        rho = ID/((self.mu_max_D - self.mu_min_D)*(self.std_max_D - self.std_min_D))
        
        rho /= ID1/((self.mu_max_D_1 - self.mu_min_D_1)*(self.mu_max_D_1 - self.mu_min_D_1))
        
        rho /= ID2/((self.mu_max_D_2 - self.mu_min_D_2)*(self.mu_max_D_2 - self.mu_min_D_2))

  
        return rho

    ###########################################################################

    ###########################################################################
    
    def P_Means(self):
        #calculates P(mean_D1>mean_D2|data)
        S_D1      = np.var(self.D_1)
        S_D2      = np.var(self.D_2)
    
        Stot2     = S_D1/self.N_1 + S_D2/self.N_2
        z_hat     = np.mean(self.D_1) - np.mean(self.D_2)
        z_hat_alt = - z_hat #in order to check if probs are consitent: change D_1 with D_2

        prefac = 1/np.sqrt(2*math.pi*Stot2)
    
        #Pmeans can be well approximated by a gaussian integral (page 97, Eq 4.34)
        Imeans = quad(my_gauss, 0, np.inf, args = (Stot2, z_hat))[0]
        Pmeans = prefac*Imeans
        
        Imeans_alt    = quad(my_gauss, 0, np.inf, args = (Stot2, z_hat_alt))[0]
        One_minusPmeans = prefac*Imeans_alt
    ###########################################################################


        return Pmeans, One_minusPmeans
