# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:21:53 2020

@author: hohle
"""
#function estimates lambda and its confidence interval "CI" of a data set 
#"data" that has been generated by a poissonian process using BPE. The 
#different entries in "data" must have been generated by equally spaced time 
#steps
#
#
#usage:
#
#data          = [1, 0, 2, 1, 1, 0] #or any other count data
#sigma         = 0.68 #confidence interval, here 1sigma
#
#run the function:
#Out = poissfit(data,sigma)
#
#Out[0]: lower bound of lambda
#Out[1]: most likely lambda (just mean(data))
#Out[2]: upper bound lambda
import statistics
import numpy as np
import matplotlib.pyplot as plt
  
def poissfit(data,CI = 0.68):

    #import pylab as p
    
    eps      = 2.220446049250313e-16
    t        = len(data)
    n        = np.sum(data)
    lamGuess = np.mean(data)
    
    lamMax   = 5*(lamGuess + 1)
    dlam     = lamMax/1e4
    lamRange = np.linspace(0,lamMax, int(1e4))
      
    #-----------------------------------------------------------------------------
    if n<10:
        n_fac  = statistics.math.factorial(n)
        exp    = np.exp(-lamRange*t) 
        pdflam = (((lamRange*t)**n)/n_fac)*exp
      
    else:
        pdflamlog = n*np.log(lamRange*t + eps)- \
        sum(np.log(np.linspace(1,n,n)))- \
        lamRange*t #log P(lam|data)
        pdflam    = np.exp(pdflamlog) #P(lam|data)
        
    C         = np.trapz(pdflam,lamRange)#first y, then x 
    pdflamN   = pdflam/C #normalizing pdf for conf int
    
###############################################################################
#evaluation:
    
    idx      = np.argmax(pdflamN)
    pdfmax   = np.amax(pdflamN)
    
#right tail
    d1     = 0
    intv1  = 0
    d2     = 1
    intv2  = 0
    
    sumintv = intv1 + intv2
    
    while (sumintv < CI*0.999): #I leave a margin here in order to account for numerical inacurracy
        if (idx+d1<len(lamRange)):
            
            d1     = d1 + 1
            indi   = np.arange(idx,idx+d1,1)
            intv1  = sum(pdflamN[indi])*dlam
#left tail
        if (idx+d2>0):
            
            d2      = d2 - 1
            indi    = np.arange(idx+d2,idx,1)
            intv2   = sum(pdflamN[indi])*dlam
            
        sumintv = intv1 + intv2

    idxU    = idx + d1
    lamUP   = lamRange[idxU]
    idxL    = idx+d2
    lamLO   = lamRange[idxL]
    diff_up = lamUP - lamGuess
    diff_do = lamGuess - lamLO
###############################################################################
#####the ploting part##########################################################


    xtofill     = lamRange[np.arange(idxL,idxU,1)]
    ytofill     = pdflamN[np.arange(idxL,idxU,1)]
    ytofill[0]  = 0
    ytofill[-1] = 0
        
    plt.plot(lamRange,pdflamN)
    plt.xlabel('estimated rate $\lambda$')
    plt.ylabel('P($\lambda$|data)')
    plt.fill(xtofill, ytofill, facecolor = 'black', alpha = 0.2)
    plt.plot([lamLO, lamLO],[0,pdflamN[idxL]],'k-')
    plt.plot([lamUP, lamUP],[0,pdflamN[idxU]],'k-')
    plt.plot([lamGuess, lamGuess],[0,pdfmax],'k--')
    plt.title('$\lambda = %1.2f ^{+ %1.2f}_{-%1.2f}$' % (lamGuess, \
                  diff_up, diff_do))
        
    plt.savefig('poiss.pdf',orientation = 'landscape')

    return([lamLO, lamGuess, lamUP])
