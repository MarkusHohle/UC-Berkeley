# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 19:53:15 2026

@author: MMH_user
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ViolinePlotStatesAD():
    
    data      = pd.read_excel('../Datasets/AD_data.xlsx', sheet_name = 'Summary Form')
    variables = data.select_dtypes(include='number').columns.drop('state')
    data['x'] = 'all'
    
    n_vars = len(variables)
    n_cols = 3                           # adjust layout if needed
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes      = axes.flatten()
    palette   = {0: '#008080', 1: '#b22222'}
    
    
    for ax, var in zip(axes, variables):
        sns.violinplot(data  = data, x = 'x', y = var, ax = ax, alpha = 0.7,\
                       split = True, hue = 'state', palette = palette)
        ax.set_title(var)
        ax.set_xlabel('')
        ax.set_xticklabels('')
    for ax in axes[len(variables):]:#remove empty subplots
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


