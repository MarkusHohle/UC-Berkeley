# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:14:31 2024

@author: MMH_user
"""

import time

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
        return results
    return get_args