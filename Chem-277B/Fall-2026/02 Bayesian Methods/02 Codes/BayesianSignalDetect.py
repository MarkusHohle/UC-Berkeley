# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:25:03 2024

@author: MMH_user
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

#from datetime import datetime
#import functools
from os import *
from multiprocessing import cpu_count
from scipy.integrate import simpson
from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman
from scipy.signal import find_peaks

###############################################################################
#decorator for measuring time
###############################################################################

def my_timer(my_function):
        def get_args(*args,**kwargs):
            t1 = time.monotonic()
            results = my_function(*args,**kwargs)
            t2 = time.monotonic()
            dt = t2 - t1
            print("Total runtime: " + str(dt) + ' seconds')
            return results
        return get_args


###############################################################################
#creates test signal
###############################################################################

def CreateSignal(N = 5000, w = 0.1, noiseratio = 0.1):
    
    t = np.arange(N).reshape((N,1))
    #y = np.cos(w*t+3)*np.sin(w*t) + noiseratio*np.random.randn(len(t),1)
    y = np.sin(w*t) + noiseratio*np.random.randn(len(t),1)
    
    y = y + abs(y.min())
    P = y/max(y)
    S = [np.random.binomial(1, p) for p in P]
    S = np.array(S)
    S = np.argwhere(S==1)# photon time of arrival (ToA)
    T = S[:,0]
    
    Phase    = T*w
    PhaseN   = Phase/(2*np.pi)
    PhaseN   = PhaseN % 1
    [n, phi] = np.histogram(PhaseN, int(np.round(N/20)))
    
    nn  = np.hstack((n, n))
    Phi = np.hstack((phi, phi[:-1] + 1))
    
    x    = w*t/(2*np.pi)
    idx  = [i for i, xx in enumerate(x) if xx <= 2]
    ycut = y[idx]# phase is lost for positive photon counts
    ycut = ycut/ycut.max()
    
    
    plt.stairs(nn/nn.max(), Phi, baseline=None, lw = 5, alpha = 0.25)
    plt.plot(x[idx], ycut, 'k-', alpha = 0.5)
    plt.xlabel('phase/2$\pi$')
    plt.title('original signal $\omega$ = ' + str(w))
    plt.legend(['phase binned from TOA', 'actual signal'], loc = 'lower left')
    plt.show()
    
    X_plot = np.arange(T.max()+1)
    Y_plot = np.zeros(len(X_plot))
    Y_plot[T] = 1
    
    plt.figure(figsize=(40,1))
    plt.stairs(Y_plot[:39], X_plot[:40], baseline=None, lw = 3, color = 'k')
    plt.xlabel('time of arrival (ToA) [s]')
    plt.title('counts')
    plt.show()
    
    
    return(T)


###############################################################################
#oridinary FFT
###############################################################################
@my_timer
def FFT(T):
    
    N       = len(T) 
    dt      = np.min(T[1:] - T[:-1])
    stretch = 10
    Dt      = T[-1] - T[0]

    w       = blackman(N)
    
    ywf     = fft(np.array(T)*w)
    yplot   = np.abs(ywf)[:N//2] #absolute value and only positive frequencies
    xf      = fftfreq(N, dt/stretch)[:N//2]
    
    m       = np.mean(yplot)
    M       = np.max(yplot)
    
    xf      = xf/stretch
    
    [idx, height] = find_peaks(yplot, prominence = m)
    
    freqdetect = xf[idx]
    ampdetect  = yplot[idx]
    
    ###################################################################
    #plotting                   
    Llegend = [str(round(i,3)) for i in freqdetect]
    
    plt.loglog(xf, yplot, '-r')
    plt.xlim([2/Dt, 1/(2*dt)])
    plt.ylim([min(yplot), M*5])
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.scatter(freqdetect, ampdetect, s = 60*np.ones((len(idx),1)),
                c = np.random.uniform(0,1,(len(idx),3)), edgecolors='k')
    for i, label in enumerate(Llegend):
        plt.text(freqdetect[i], ampdetect[i]*3, label, fontsize=12, ha='right')
    plt.grid()
    plt.show()


###############################################################################
#plotting periodogram
###############################################################################

def PlotPvsW(Omega, Pw, T):
    
    
    idPw     = Pw.argmax()
    wBest    = Omega[idPw]
    M        = Pw.max()
    m        = Pw.min()
    
    #plotting peaks
    [idx, height] = find_peaks(Pw, height = 0.995* np.mean(Pw))
    
    freqdetect    = Omega[idx]
    ampdetect     = Pw[idx]
    
    
    Llegend       = [str(round(i,3)) for i in freqdetect]
    
    plt.grid()
    plt.plot(Omega,Pw)
    for i, label in enumerate(Llegend):
        plt.text(freqdetect[i], M, label, fontsize=12, ha='right')
        plt.plot([freqdetect[i], freqdetect[i]],[m, M], 'k--')
    plt.scatter(freqdetect, ampdetect, s = 60*np.ones((len(idx),1)),
                c = np.random.uniform(0,1,(len(idx),3)), edgecolors='k')
    plt.xscale('log')
    plt.xlabel('frequency $\omega$ [Hz]')
    plt.ylabel('ln[P($\omega$|D)]')
    plt.show()
    
    Phase    = T*wBest
    PhaseN   = Phase/(2*np.pi)
    PhaseN   = PhaseN % 1
    [n, phi] = np.histogram(PhaseN, 20)
    
    N   = np.hstack((n, n))
    Phi = np.hstack((phi, phi[:-1] + 1))
   
    wBestR = round(wBest*1000)/1000
    
    plt.stairs(N, Phi, baseline=None, lw = 3)
    plt.xlabel('phase/2$\pi$')
    plt.title('reconstructed light curve for $\omega_{best}$ = ' + str(wBestR))
    plt.grid()
    plt.show()
    
    
###############################################################################
#calculates probabilities for frequencies depending on phase and bin
###############################################################################
def CalculateProbs(Omega, Phi, M, T, P, DT, dt, Pr_omega, Pr_M, N):
    
    
    for i, w in enumerate(Omega):#loop over frequencies
        for j, phi in enumerate(Phi):#loop over phase 
            Phase  = T*w + phi
            PhaseN = Phase/(2*np.pi)
            PhaseN = PhaseN % 1
            for k, m in enumerate(M):#loop over phase bins
                [n, _] = np.histogram(PhaseN, m)
                r      = n*m/DT   #rate
                A      = np.mean(r)
                Am     = A*m
                f      = r/(Am) #fraction of events per bin
                f     += 1e-88
                
                #log of P(D|w,phi,f,Mm)
                logP_D = N*( np.log(dt) + np.log(Am) ) +\
                         np.dot(n, np.log(f)) - A*DT +\
                         np.log(Pr_M[k]) + np.log(Pr_omega[i])
                         
                P[i,j,k] = logP_D
                
    return P


# =============================================================================
# ### same function as before, but completely refractored and full comprehension
# 
# ####
# def OverM(m, PhaseN, DT, N):
#     
#     [n, _] = np.histogram(PhaseN, m)
#     r      = n*m/DT   #rate
#     A      = np.mean(r)
#     Am     = A*m
#     f      = r/(Am) #fraction of events per bin
#     f     += 1e-88
#     
#     #log of P(D|w,phi,f,Mm)
#     logP_D = N*(np.log(Am) ) + np.dot(n, np.log(f)) - A*DT 
# 
#     return logP_D
# 
# ####
# 
# ####
# def OverPhi(phi, T, w, M, DT, N, Pr_M):
#     
#     Phase    = T*w + phi
#     PhaseN   = Phase/(2*np.pi)
#     PhaseN   = PhaseN % 1
#     
#     logP_D = [OverM(m, PhaseN , DT, N) for m in M] + np.log(Pr_M)
#     
#     return logP_D
# 
# ####
# 
# 
# ####
# def OverOmega(omega, Phi, M, T, P, DT, dt, pr_omega, Pr_M, N):
#     
#     logP_D = [OverPhi(phi, T, omega, M, DT, N, Pr_M) for phi in Phi]
#     
#     return logP_D + np.log(pr_omega)
# 
# ####
# def CalculateProbs2(Omega, Phi, M, T, P, DT, dt, Pr_omega, Pr_M, N):
#     
#     P = [OverOmega(omega, Phi, M, T, P, DT, dt, pr_omega, Pr_M, N)\
#          for omega, pr_omega in zip(Omega, Pr_omega)]
# 
#     P = P + N*np.log(dt)
#                 
#     return P
# =============================================================================

###############################################################################
#actual signal detection
###############################################################################

class SignalDetect():
    #Detects signal of any shape and phase using Bayesian Signal detection,
    #see Gregory & Loredo, 1992
    #Signal has to be an array T containing photon time of arrival (ToA).
    #Assumption: photons have been emitted by poissonian process.
    #
    #Code is being executed parallel for n cpus
    
    def __init__(self, T, Range_phi = [0, 2*np.pi], dphi = 0.01, MaxM = 20,\
                    **Opts):
        
        #cleaning folder from previous data
        L = listdir(getcwd())
        [remove(i) for i in L if '.npy' in i]

        
        #T: Signal
        #interval for priors:
        #Range_phi  :   phase
        #dphi       :   increment for phase
        #MaxM       :   max number of bins for light curve
        #
        #
        #Opts:
        #w_start    :   start value for frequency in grid search
        #w_end      :   stop value for frequency in grid search
        #dw         :   increment for (angular) frequency
        #dt         :   time resolution of detector
        
        n         = cpu_count() -1
        
        DT        = T.max() - T.min() #total time span
        N         = len(T)
        diff      = T[1:] - T[:-1]
        phi_start = Range_phi[0]
        phi_end   = Range_phi[1]
        
        
        #######################################################################
        #Opts:
        
        if 'w_start' in Opts:
            w_start = Opts['w_start']
        else:
            #Niquist limit
            w_start = 20/DT
            
        if 'w_end' in Opts:
            w_end = Opts['w_end']
        else:
            #Niquist limit
            w_end   = 1/(2*diff.min())
        
        Range_omega = [w_start, w_end]
        
            
        if 'dw' in Opts:
            dw  = Opts['dw']
        else:
            dw  = 20/DT

        w_start = Range_omega[0]
        w_end   = Range_omega[1]
        
        
        if 'dt' in Opts:
            dt  = Opts['dt']
        else:
            dt  = diff.min()
            
            
        if w_start < 20/DT:
            print('Warning: \u03C9_start = ' + str(w_start) +\
                  ' is below theoretical limit of ' + str(20/DT) + '.\n' +
                  'Results might be inaccurate. Choose a different \u03C9_start!')
                
        if w_end > 1/(2*diff.min()):
            print('Warning: \u03C9_end = ' + str(w_end) +\
                  ' is above theoretical limit of ' + str(1/(2*diff.min())) + '.\n' +
                  'Results might be inaccurate. Choose a different \u03C9_end!')
            
        #######################################################################
        
        Omega    = np.arange(w_start, w_end, dw)
        
        #preparing grid search for parallel processing
        
        #creating list of Omega according to number of cpus
        L = []
        k = int(np.floor(len(Omega)/n))
        for i in range(n):
            L += [Omega[i*k: k*(i+1)]]
        L += [Omega[n*k:]] 
        
        #Prior omega (MaxEnt):
        Pr_Omega = 1/np.log(w_end/w_start)/Omega
        #creating list of Prior of Omega according to number of cpus
        LL = []
        for i in range(n):
            LL += [Pr_Omega[i*k: k*(i+1)]]
        LL += [Pr_Omega[n*k:]] 
        
        
        self.Phi       = np.arange(phi_start, phi_end, dphi)
        self.phi_start = phi_start
        self.phi_end   = phi_end
        self.dphi      = dphi
        self.M         = np.arange(2, MaxM)
        self.dt        = dt
        self.N         = N
        self.DT        = DT
        self.T         = T
        self.n         = n
        self.LOmega    = L
        self.POmega    = LL
        self.Omega     = Omega
        
        M_             = self.M - 1
        
        
        self.Pr_omega = 1/np.log(w_end/w_start)/Omega
        
        #constant factors. we will calculate odds ratios anyways
        #Pr_phi   = 1/(phi_end - phi_start)
        #Pr_A     = 1/(N/DT)
        self.Pr_M     = [math.factorial(m) for m in M_]
        
    #@functools.lru_cache()
    def RunAnalysis(self, Omega, Pr_omega, count):
        
        T         = self.T
        Phi       = self.Phi
        M         = self.M
        Pr_M      = self.Pr_M
        dt        = self.dt
        N         = self.N
        DT        = self.DT
        phi_start = self.phi_start
        phi_end   = self.phi_end
        #dphi      = self.dphi
        
        P         = np.zeros((len(Omega),len(Phi),len(M)))
        
# =============================================================================
        P = CalculateProbs(Omega, Phi, M, T, P, DT, dt, Pr_omega, Pr_M, N)
# =============================================================================

        #integration and plots
        Y     = np.linspace(phi_start, phi_end, len(Phi))
        Z     = np.linspace(M.min()  , M.max(), len(M))
        
        margM = simpson(P, x = Z)#marginalization over bins
        Pw    = simpson(margM, x = Y, axis = 1)#best omega marginalization over bins 
                                        #& phase
                                        
        np.save('Pw' + str(count), Pw)

    @my_timer
    def FindFrequency(self):
        
        LOmega = self.LOmega #list of arrays for omega
        PriorO = self.POmega #list of arrays for 1/np.log(w_end/w_start)/Omega
        
        Processes = [mp.Process(target = self.RunAnalysis,\
                                args = (omega, prior, i,))\
                     for i, (omega, prior) in enumerate(zip(LOmega, PriorO))]
        
        for p in Processes:
            p.start()
        for p in Processes:
            p.join()

        P = []
        for i in range(len(LOmega)):
            P = np.hstack((P,np.load('Pw' + str(i) + '.npy')))
            

        Omega  = self.Omega
        
        PlotPvsW(Omega, P, self.T)
        
        return(Omega, P)
    
