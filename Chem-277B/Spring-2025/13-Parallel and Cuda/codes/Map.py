# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:25:53 2024

@author: MMH_user
"""
import time

import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocess import cpu_count



###usage#######################################################################
# from Map import *

# globals()['MyFun'] = my_timer(MyFun)

# MAP  = Parallel_MAP()

# out  = MyFun(list_filenames[0])#25sec

# out1 = MAP.Serial(list_filenames)#150sec
# out2 = MAP.Parallel(list_filenames)#60sec
###############################################################################

list_filenames =\
    ['Data_set_0.xlsx',\
     'Data_set_1.xlsx',\
     'Data_set_2.xlsx',\
     'Data_set_3.xlsx',\
     'Data_set_4.xlsx',\
     'Data_set_5.xlsx']

#------------------------------------------------------------------------------
def my_timer(my_function):
        def get_args(*args,**kwargs):
            t1 = time.monotonic()
            results = my_function(*args,**kwargs)
            t2 = time.monotonic()
            dt = t2 - t1
            print("Total runtime: " + str(dt) + ' seconds')
            return results
        return get_args
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#@my_timer
def MyFun(filename: str) -> np.array:
    #reads huge data set
    data = pd.read_excel(filename)
    return np.array(data.time)
#------------------------------------------------------------------------------

class Parallel_MAP():
    
    
#------------------------------------------------------------------------------
    @my_timer
    def Serial(self, list_filenames: list) -> list:
        
        All = [None]*len(list_filenames)
        
        for i, List in enumerate(list_filenames):
            All[i] = MyFun(List)
            
        return All
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
    @my_timer
    def Parallel(self, list_filenames: list) -> list:
        
        with mp.Pool(processes = cpu_count() - 1) as pool:
            All = pool.map(MyFun, list_filenames)
            
        return All
    
    
