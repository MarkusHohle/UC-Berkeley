# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:40:25 2023

@author: MMH_user
"""

#creating n data sets of length len(t) in m files

import numpy as np
import pandas as pd

n  = 100#50#100
m  = 5
t  = np.arange(0,100,0.001)
lt = len(t)

cols = ['time'] + ['sample ' + str(i) for i in range(n)]

for i in range(m):
    S = np.zeros((lt,n))
    for j in range(n):
        nf  = np.random.uniform(5,10)
        nf  = int(nf)
        
        amp = np.random.uniform(0.1, 10,(nf,1))
        fre = 10**np.random.uniform(-1, 2,(nf,1))
        pha = np.random.uniform(0, 2*np.pi,(nf,1))
        off = np.random.uniform(0.1, 10,(nf,1))
        
        for k in range(4):
            S[:,j] += amp[k] * np.sin(2*np.pi*fre[k]*t + np.pi*pha[k]) + off[k]
            
    St  = np.hstack((t.reshape(lt,1),S))
    Sdf = pd.DataFrame(St, columns = cols)
    
    Sdf.to_excel('Data_set_' + str(i) + '.xlsx')
    Sdf.to_csv('Data_set_' + str(i) + '.csv')

