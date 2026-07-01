# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:52:48 2022

@author: hohle
"""

# =============================================================================
# #usage:
# #    n1 = 4
# #    k1 = np.random.binomial(n1, 0.2)
# #    [q1,b,Prior] = bayesian_bino(n1,k1)
# #    print(q1)
#     
# #    n2 = 5
# #    k2 = np.random.binomial(n2, 0.2)
# #    [q2,b,Prior] = bayesian_bino(n2,k2, Prior = Prior)
# #    print(q2)
#     
# #    n = 4
# #    q = 0.2
# #    k = np.random.binomial(n, q)
# #    [q,b,Prior] = bayesian_bino(n,k)
# #    for i in range(10):
# #        k = np.random.binomial(n, q)
# #        [q,b,Prior] = bayesian_bino(n,k, Prior = Prior)
# #        print(q)
# 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def bayesian_bino(n, k, CI = 0.68, **Prior):


    
    dq     = 1/(1000*n)
    q      = np.arange(0,1,dq)
    #qGuess = k/n
    
    if Prior:
        P        = Prior['Prior']
        [nr, nc] = P.shape
        if nr < nc:
            P = np.transpose(P)
        yint = np.interp(q, P[:,0], P[:,1])
    else:
        yint = 1
    
    #uniform prior
    Pu     = (q**k)*(1 - q)**(n - k) * yint
    C      = np.trapz(Pu,q)#for normalization: first y, then x 
    Pu     = Pu/C #normalizing pdf for conf int
    whereq = Pu.argmax()#getting index and value of most likely q and Pu
    maxPu  = Pu.max()
    maxq   = q[whereq]
    
    #determining errors (gauss approx of pdf)
    sig_q  = (maxq*(1-maxq)/n)**0.5
    
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
        if (whereq+d1<len(q)):
            
            d1     = d1 + 1
            indi   = np.arange(whereq, whereq+d1,1)
            intv1  = sum(Pu[indi])*dq
    #left tail
        if (whereq+d2>0):
            
            d2      = d2 - 1
            indi    = np.arange(whereq+d2,whereq,1)
            intv2   = sum(Pu[indi])*dq
            
        sumintv = intv1 + intv2

    idxU    = whereq + d1
    qUP     = q[idxU]
    idxL    = whereq+d2
    qLO     = q[idxL]
    diff_up = qUP - maxq
    diff_do = maxq - qLO
    
    
###############################################################################
#####the ploting part##########################################################

        
    xtofill     = q[np.arange(idxL,idxU,1)]
    ytofill     = Pu[np.arange(idxL,idxU,1)]
    ytofill[0]  = 0
    ytofill[-1] = 0
        
    plt.plot(q,Pu, color = 'black')
    plt.xlabel('estimated q')
    plt.ylabel('P(q|data)')
    plt.fill(xtofill, ytofill, facecolor = 'black', alpha = 0.02)
    plt.plot([qLO, qLO],[0,Pu[idxL]],'k-')
    plt.plot([qUP, qUP],[0,Pu[idxU]],'k-')
    plt.plot([maxq, maxq],[0,maxPu],'k--')
    plt.title('$q = %1.2f ^{+ %1.2f}_{-%1.2f}$' % (maxq, \
                  diff_up, diff_do))
    plt.show()
        
    plt.savefig('bino.pdf', orientation = 'landscape')

    return([qLO, maxq, qUP], sig_q, np.array([q,Pu]))