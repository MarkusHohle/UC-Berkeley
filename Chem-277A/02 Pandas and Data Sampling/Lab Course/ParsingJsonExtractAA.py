# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 18:19:21 2026

@author: MMH_user
"""
import re
import json
import numpy as np
import pandas as pd

from rapidfuzz import fuzz


"""
USAGE:
    from ParsingJsonExtractAA import *
    
    P   = ParsingJsonExtractAA()
    SEQ = P.ExtractSeq()
"""

###############################################################################
AA3_TO_AA1_Dict    = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D",
                      "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
                      "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
                      "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
                      "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}

Convert_AA3_TO_AA1 = lambda AA: [AA3_TO_AA1_Dict[aa] for aa in AA]

aa_seq_regex       = re.compile(r"(\d+)(([A-Z][a-z]{2}\s+)+[A-Z][a-z]{2})(\d+)")

#fuzzy search methods
Methods            = ['ratio', 'partial_ratio',
                      'token_sort_ratio', 'token_set_ratio']
###############################################################################

###############################################################################
def FuzzyMatch(series, search_string: str, method: str = 'partial_ratio', threshold: float = 80):
    """
    returns rows idx where fuzz search found match > threshold
    methods: ratio
             partial_ratio
             token_sort_ratio
             token_set_ratio
    """
    M         = getattr(fuzz, method)  # runs fuzz match
    Mfun      = lambda x: M(x.lower(), search_string.lower()) >= threshold # defines function runs fuzz match and 
                                                                           # returns True | False if match is above threshols 
    TrueFalse = series.fillna("").astype(str).apply(Mfun) # applies Mfun to df column (which is a series)
    
    return np.argwhere(TrueFalse == True).reshape(-1)
###############################################################################

###############################################################################
class ParsingJsonExtractAA():
    
    def __init__(self, JsonFileName: str = 'US11591390B2.json',\
                       Search_Str:   str = 'sequence'):
        #load the file
        with open(JsonFileName, 'r') as file:
            data = json.load(file)
            
        #it is a dictionary:
        #print(type(data))


        Keys = pd.Series(data.keys())


        for m in Methods:
            idx = FuzzyMatch(Keys, Search_Str, method = m)
            print('method: '+ m + '\n')
            print('index and match: \n')
            print(Keys[idx])
            print('----------------------------------------------------\n')
            print('----------------------------------------------------\n')
            
            
        self.data = data
        
        
    def ExtractSeq(self, key: str = 'sequenceListNewRules') -> list:

        Seq      = self.data[key]
        #SeqSplit = Seq.split(' ')

        AA3    = aa_seq_regex.findall(Seq)
        AA_Seq = [None]*len(AA3)

        for i, aa3 in enumerate(AA3):
            AA_Seq[i] = ''.join(Convert_AA3_TO_AA1(aa3[1].split(' ')))
            
        return AA_Seq
    



