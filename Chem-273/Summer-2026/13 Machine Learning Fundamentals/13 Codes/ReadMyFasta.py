# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:52:41 2023

@author: hohle
"""

import pandas as pd
import os
from Bio import AlignIO, SeqIO, Seq

def ReadMyFasta(filename,separator = '>'):
    
    #reads fasta file 'filename' (string) with separator (string, ususally
    #'>')
    
    #returns
    # - df of sample names (Names)      - for later use (eg, labeling, Matlab)
    # - df of actual sequences (MySeq)  - for later use (eg, labeling, Matlab)
    # - an alignment object (ToAlign)-  - so that Python can use it
    
    data  = pd.read_csv(filename, header = None)
    
    
    #1) reading names and sequence
    
    #finding sample names by seperator and saving them in the df 'Names' and
    #removing the seperator sign
    Names    = data[data[0].str.contains(separator) == True]
    Names[0] = Names[0].map(lambda x: x.lstrip(separator))
    
    Idx  = Names.index
    L    = len(Names)
    
    #preallocating empty list (L is the maximum size, depending on formatting)
    MySeq   = [0]*L
    
    for i in range(L-1):
        idx1     = Idx[i]
        idx2     = Idx[i+1]
        MySeq[i] =''.join(data.iloc[idx1+1:idx2,0])
    
    #the last sequence
    MySeq[L-1] = ''.join(data.iloc[idx2+1:-1,0])
    
    
    #creates alignment object 
    #(= solving error message "Sequences must all be the same length"), 
    #code from Jeroen Vangoey
    #
    records = SeqIO.parse(filename, 'fasta')
    records = list(records) # make a copy, otherwise our generator
                            # is exhausted after calculating maxlen
    maxlen = max(len(record.seq) for record in records)

    # pad sequences so that they all have the same length
    for record in records:
        if len(record.seq) != maxlen:
            sequence = str(record.seq).ljust(maxlen, '.')
            record.seq = Seq.Seq(sequence)
    assert all(len(record.seq) == maxlen for record in records)

    # write to temporary file and do alignment
    output_file = '{}_padded.fasta'.format(os.path.splitext(filename)[0])
    with open(output_file, 'w') as f:
        SeqIO.write(records, f, 'fasta')
    ToAlign = AlignIO.read(output_file, "fasta")
    
    
    
    return(Names, MySeq, ToAlign)
        
