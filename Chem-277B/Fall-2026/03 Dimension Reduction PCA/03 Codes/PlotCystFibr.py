# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 20:27:43 2026

@author: MMH_user
"""


import pandas as pd
import seaborn as sns

data  = pd.read_csv('../03 Datasets/Cystfibr.txt', delimiter = '\t')

Pair = sns.pairplot(data, kind = 'kde')
Pair.map_lower(sns.scatterplot, c = 'k')
Pair.map_upper(sns.scatterplot, c = 'k')
Pair.map_diag(sns.histplot, stat = 'density')