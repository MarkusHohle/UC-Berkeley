# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:34:00 2024

@author: MMH_user
"""
import numpy as np

def prepare_data(data, n_past, n_future):
    n_samples = len(data) - n_past - n_future + 1
    
    x = np.empty((n_samples, n_past) + data.shape[1:], dtype=data.dtype)
    y = np.empty((n_samples, n_future) + data.shape[1:], dtype=data.dtype)
    
    for i in range(n_samples):
        x[i] = data[i : i + n_past]
        y[i] = data[i + n_past : i + n_past + n_future]
    
    return x, y