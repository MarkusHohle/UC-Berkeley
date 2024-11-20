# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:24:53 2024

@author: MMH_user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:25:53 2024

@author: MMH_user
"""
import time
import glob
#import datetime as DT

import os
import pandas as pd
import multiprocessing as mp

#from multiprocessing import Pool

###usage#######################################################################
# from Process import *

# globals()['MyFun'] = my_timer(MyFun)

# out  = MyFun(list_filenames[0])#25sec

# Pro  = Parallel_Process()
# Pro.Serial(list_filenames)#180sec
# Pro.Parallel(list_filenames)#68sec
###usage#######################################################################

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
def MyFun(filename: str):
    #reads & saves huge data set
    data = pd.read_excel(filename)
    data.to_csv('Data_set_' + str(time.monotonic()).replace('.','-') + '.csv')
#------------------------------------------------------------------------------

class Parallel_Process():
    
    def __init__(self):
        list_filenames = glob.glob('*.csv')
        for files in list_filenames:
            os.remove(files) 
        
        
#------------------------------------------------------------------------------
    @my_timer
    def Serial(self, list_filenames: list) -> list:
                
        for List in list_filenames:
            MyFun(List)

#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
    @my_timer
    def Parallel(self, list_filenames: list) -> list:
        
        Processes = [mp.Process(target = MyFun, args = (List,))\
                     for List in list_filenames]
         
        for p in Processes:
            p.start()
            
        for p in Processes:
            p.join()

