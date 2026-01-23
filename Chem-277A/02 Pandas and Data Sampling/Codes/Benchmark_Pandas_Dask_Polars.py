# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 01:58:29 2025

@author: MMH_user
"""


import time
import pandas as pd
import dask.dataframe as dd
import polars as pl
#run pip install dask and/or pip install polars if needed!

"""
function benchmarks different data reading methods (pandas, polars and dask)
for different data formats (.excel, .csv, .txt)
USAGE:
    
    dfPandasCSV = ReadWithAnyToolAnyMethod()                                                                                #pandas as default, read csv
    dfPandasEXL = ReadWithAnyToolAnyMethod(filename = '../Datasets/Data_Set.xlsx', my_method = 'read_excel')                #pandas as default, read xlsx

    dfDaskCSV   = ReadWithAnyToolAnyMethod(my_tool = 'dd')                                                                  #dask

    dfPolarsCSV = ReadWithAnyToolAnyMethod(my_tool = 'pl')                                                                  #polars
    dfPolarsEXL = ReadWithAnyToolAnyMethod(filename = '../Datasets/Data_Set.xlsx', my_tool = 'pl', my_method = 'read_excel')#polars
    
"""


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
        return results, dt
    return get_args

@my_timer
def ReadWithAnyToolAnyMethod(filename: str = '../Datasets/Data_Set.csv', my_tool: str = 'pd', my_method: str = 'read_csv') -> pd.DataFrame:
    tool   = globals()[my_tool]
    method = getattr(tool, my_method)
    return method(filename)