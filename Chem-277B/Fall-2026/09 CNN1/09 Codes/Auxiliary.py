# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 19:49:24 2026

@author: MMH_user
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def FindMyFile(filename: str, ServerHardDiscPath: str = r"c:\Users\MMH_user\Desktop") -> str:
    """
    finds file of name "filename" anywhere in "ServerHardDiscPath" and returns complete path
    """
    for r,d,f in os.walk(ServerHardDiscPath):
        for files in f:
             if files == filename: #for example "MessyFile.xlsx"
                 file_name_and_path =  os.path.join(r,files)
                 return file_name_and_path
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
