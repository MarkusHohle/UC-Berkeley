# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:34:00 2024

@author: MMH_user
"""
import numpy as np

def prepare_data(data, n_past, n_future):
    x, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        x.append(data[i - n_past:i])
        y.append(data[i:i + n_future])
    return np.array(x), np.array(y)