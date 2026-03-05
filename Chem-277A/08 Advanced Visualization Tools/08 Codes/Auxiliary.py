# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 00:18:16 2026

@author: MMH_user
"""
import os  
import numpy as np
import pandas as pd
import polars as pl
import umap.umap_ as umap
import dask.dataframe as dd
import matplotlib.pyplot as plt

###############################################################################
def FindMyFile(filename: str, ServerHardDiscPath: str = r"c:\Users\MMH_user\Desktop") -> str:
    """
    finds file of name "filename" anywhere in 
    "ServerHardDiscPath" and returns complete path
    """
    for r,d,f in os.walk(ServerHardDiscPath):
        for files in f:
             if files == filename: #for example "AD_data.xlsx"
                 file_name_and_path =  os.path.join(r,files)
                 return file_name_and_path
###############################################################################

###############################################################################
def ReadWithAnyToolAnyMethod(filename:  str = 'Data_set_0.csv', 
                             my_tool:   str = 'pd', 
                             my_method: str = 'read_csv', **kwargs) -> pd.DataFrame:
    tool   = globals()[my_tool]
    method = getattr(tool, my_method)
    return method(filename,**kwargs)
###############################################################################

###############################################################################
def DrawUMAP(data, Y,  n_neighbors:  int = 15, min_dist: float = 0.1,
                       n_components: int = 2,  metric:   str   = 'euclidean',
                       title:        str = ''):
    
    U = umap.UMAP(n_neighbors  = n_neighbors,  min_dist = min_dist,
                  n_components = n_components, metric   = metric)
    
    XYZnew = U.fit_transform(data)
    
    coords = [XYZnew[:, i] for i in range(n_components)]
    
    
    #plotting
    fig = plt.figure()

    # choose projection dynamically
    projection = '3d' if n_components == 3 else None
    ax         = fig.add_subplot(111, projection = projection)

    

    if n_components == 1:
        coords.append(range(len(XYZnew)))

    #trick: coords are *args
    ax.scatter(*coords, c = Y, marker = '.', cmap = 'Spectral', alpha = 0.3
               if n_components == 3 else None)

    ax.set_title(title, fontsize = 18)
    plt.show()
