# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 22:48:58 2025

@author: MMH_user
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#a comment

NT   = ['A', 'C', 'G', 'T']
Code = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]] 

Dict = {key: value for key, value in zip(NT,Code)}

Encoder = lambda Sequence: [Dict[s] for s in Sequence]

def MyPlotRoutine1(E, Seq):
    sns.heatmap(E, cmap = "Blues", xticklabels = list(Seq),\
                yticklabels = [None]*4)
    plt.show()
    
    
def MyPlotRoutine2(E, Seq):
    plt.imshow(E, cmap = 'gray')
    plt.xticks(ticks = range(len(Seq)), labels = list(Seq))
    plt.show()


class EncodeMySeq():
    
    def __init__(self, Seq):
        
        self.E   = np.array(Encoder(Seq)).transpose()
        self.Seq = Seq
        
    def PlotMySeq1(self):
        MyPlotRoutine1(self.E, self.Seq)
        
    def PlotMySeq2(self):
        MyPlotRoutine2(self.E, self.Seq)