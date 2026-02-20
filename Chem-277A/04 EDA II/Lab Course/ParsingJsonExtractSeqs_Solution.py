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
    
    P             = ParsingJsonExtractSeq()
    Sum, ID, INFO = P.ExtractSeq()
"""

###############################################################################
AA3_TO_AA1_Dict    = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D",
                      "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
                      "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
                      "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
                      "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}

Convert_AA3_TO_AA1 = lambda AA: [AA3_TO_AA1_Dict[aa] for aa in AA]

Seq_regex          = re.compile(r'(\d+(?:PRT|DNA))') #finding PRT for prot or DNA
                                                     #without capturing it ?:
                                                         
# ok for sample solution: returns first part of AA sequence
AA_seq_regex       = re.compile(r"(?<=\d)([A-Z][a-z]{2}(?:\s+[A-Z][a-z]{2})*)(?=\d)")
                                # [A-Z][a-z]{2} looking for one cap, followed by two lower case letters
                                # \s+           one or more spaces
                                # (?:           non-capturing --> for later use: .findall, search etc
                                # *             like wildcard, more AAs can follow
                                # <=\d          must be preceded (followed) by digit
                                

# one alternative: explixit AA --> best result
AA_seq_regexAlt    = re.compile(r'(?:\d+)?\b(Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)\b') 
DNA_seq_regex      = re.compile(r"(?<=\d)((?:[acgt]+\s*\d*\s*)+)", re.IGNORECASE)

#fuzzy search methods
Methods            = ['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio']
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
class ParsingJsonExtractSeq():
    
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
        self.Seq = Seq

        Records  = Seq_regex.split(Seq)
        
        Records = [r.strip() for r in Records if r.strip()] #removing blanks
        
        Info    = Records[1::2]
        ID      = Records[0::2]
        
        Summary = [None]*len(ID) 
        ctDNA   = 0
        ctAA    = 0  
        
        for i, (i_d, info) in enumerate(zip(ID, Info)):
            
            if 'PRT' in i_d:
                
                #print(i)
                
                S1 = AA_seq_regex.search(info) #search returns match as group and
                S2 = AA_seq_regexAlt.findall(info) #search returns match as group and
                
                if S1:
                    ctAA +=1 
                    startIdx   = S1.span()[0]               #start/end index as span
                    #AA1        =''.join(Convert_AA3_TO_AA1(S1.group().split(' ')))
                    AA1        =''.join(Convert_AA3_TO_AA1(S2))
                    Text       = info[:startIdx]
                else:
                    AA1  = info
                    Text = f'Sequence at location {i} could not be identified!\nCheck manually!'
                    print(Text)
                
                Summary[i] = {"type"     : "AA",
                              "Sequence" : AA1,
                              "ID"       : i_d,
                              "Info"     : Text
                              } 
                
            if 'DNA' in i_d:
                
                S = DNA_seq_regex.search(info)
                
                if S:
                    ctDNA +=1 
                    startIdx   = S.span()[0]
                    Text       = info[:startIdx]   
                    DNA        = S.group().upper().split(' ')
                else:
                    DNA  = info
                    Text = f'Sequence at location {i} could not be identified!\nCheck manually!'
                    print(Text)
                
                Summary[i] = {"type"     : "DNA",
                              "Sequence" : DNA,
                              "ID"       : i_d,
                              "Info"     : Text
                              } 
        
        print(f"Among {len(ID)} records\n{ctAA} AA sequences and {ctDNA} DNA sequences were found.")
        
        return Summary, ID, Info
    



