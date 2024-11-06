# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 01:30:51 2024

@author: MMH_user
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functools import lru_cache

from Bio import AlignIO, SeqIO, Seq
from keras import optimizers

from keras.layers import LSTM
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D

from sklearn.metrics import confusion_matrix


#Package uses barcodes for Classification
#
#Usage

# A = Analyzer()            #reads fasta file
# A.RunCNN()                #runs CNN --> takes a few hours
# A.EvalModel()             #plots confusion map and entropy
#
#or
#
# A = Analyzer(Ncut = 4000) #reads fasta file
# A.RunLSTM(Nepochs = 40)   #runs simple LSTM for reduced dataset (computational reasons)
# A.EvalModel()             #plots confusion map and entropy

###helper functions############################################################
###############################################################################

###############################################################################
@lru_cache(maxsize = None)
def ReadMyFasta(filename, separator = '>'):
    
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
###############################################################################

###############################################################################

Value   = np.eye(4)
Key     = 'ACGT'

Dict    = {k: v for k,v in zip(Key, Value)} 
Dict.update({'-': [0, 0, 0, 0]})
Dict.update({'B': [0, 0, 0, 0]})
Dict.update({'D': [0, 0, 0, 0]})
Dict.update({'E': [0, 0, 0, 0]})
Dict.update({'F': [0, 0, 0, 0]})
Dict.update({'H': [0, 0, 0, 0]})
Dict.update({'I': [0, 0, 0, 0]})
Dict.update({'J': [0, 0, 0, 0]})
Dict.update({'K': [0, 0, 0, 0]})
Dict.update({'L': [0, 0, 0, 0]})
Dict.update({'M': [0, 0, 0, 0]})
Dict.update({'N': [0, 0, 0, 0]})
Dict.update({'O': [0, 0, 0, 0]})
Dict.update({'P': [0, 0, 0, 0]})
Dict.update({'Q': [0, 0, 0, 0]})
Dict.update({'R': [0, 0, 0, 0]})
Dict.update({'S': [0, 0, 0, 0]})
Dict.update({'U': [0, 0, 0, 0]})
Dict.update({'V': [0, 0, 0, 0]})
Dict.update({'W': [0, 0, 0, 0]})
Dict.update({'X': [0, 0, 0, 0]})
Dict.update({'Y': [0, 0, 0, 0]})
Dict.update({'Z': [0, 0, 0, 0]})
Dict.update({'.': [0, 0, 0, 0]})

Encode = lambda S: [Dict[s] for s in S]

###############################################################################
def SequenceEncoder(NTSequence):
    
    Encoded = []
    
    for b in NTSequence:
        out = np.array(list(map(Encode, [b])))
        #out = out[0,:,:]
        #out = out[:-2,:]#removing end of barcode
        Encoded  += [out.transpose()]

    return Encoded
###############################################################################

###############################################################################
def LabelEncoder(LabelList):
    
    Labels = []
    
    for y in LabelList[0]:
        idx    = [i for i, letter in enumerate(y) if letter == '|']
        idx    = np.array(idx)
        Labels += [ y[idx[0]+1:idx[1]] ]
    
    LabelsUnique = set(Labels)
    Nclass       = len(LabelsUnique)
    
    
    ValueLabel   = np.eye(Nclass)

    DictLabel    = {k: v for v, k in zip(ValueLabel, LabelsUnique)}
    EncodeLabel  = lambda S: [DictLabel[s] for s in S]
 
    Y_onehot     = np.array(EncodeLabel(Labels))
    Y_sparse     = np.argmax(Y_onehot, axis = 1)
    
    return Y_sparse, Y_onehot, Nclass, LabelsUnique, Labels
###############################################################################

###############################################################################
def LoadandEncodeData(my_file):
        
        [Names, NTSequence, _]                             = ReadMyFasta(my_file)
        
        Encoded                                            = SequenceEncoder(NTSequence)
        [Y_sparse, Y_onehot, Nclass, LabelsUnique, Labels] = LabelEncoder(Names)
        
        return Y_sparse, Y_onehot, Nclass, LabelsUnique, Encoded, Labels
###############################################################################


class Analyzer():
    
    def __init__(self, trainingFile = 'fasta.fas', \
                       testingFile  = 'fasta.fas', Ncut = 1000):
        
        [Y_sparse_train, Y_onehot_train, Nclass, LabelsUnique,\
                       Encoded_train, Labels] = LoadandEncodeData(trainingFile)
            
        [Y_sparse_test,  Y_onehot_test,      _ ,            _,
                       Encoded_test, _      ] = LoadandEncodeData(testingFile)
        
        #----------------------------------------------------------------------
        #turning data set into numpy array-------------------------------------
        #number of sequences x length of sequence x numner of features
        #and removing classes which are less frequent than Ncut
        labels, counts = np.unique(Y_sparse_train, return_counts = True)
        
        #using only those classes, that appear at least Ncut times
        IdxMost        = np.argwhere(counts > Ncut)
        Idx            = np.argwhere(Y_sparse_train == IdxMost)[:,1]
        Encoded_train  = [Encoded_train[i] for i in Idx]
        
        Llist   = len(Encoded_train)#number of sequences
        #length of sequences
        LenSeq  = np.array([S.shape[1] for S in Encoded_train])
        X_train = np.zeros((Llist, np.max(LenSeq), Encoded_train[0].shape[0]))
            
        for i in range(Llist-1):
            Seq                    = Encoded_train[i][:,:,0]
            (rows, cols)           = Seq.shape
            X_train[i,:cols,:rows] = Seq.transpose()
        #----------------------------------------------------------------------
        
        
        #filtered class vectors
        Nclass         = len(IdxMost)
        Y_sparse_train = Y_sparse_train[Idx]
        
        Y_onehot_train = Y_onehot_train[Idx,:]
        Y_onehot_train = Y_onehot_train[:, IdxMost]
        Y_onehot_train = Y_onehot_train[:,:,0]
        
        #filtering names
        LabelsIdx      = [Labels[i] for i in Idx]
        
        #----------------------------------------------------------------------
        plt.bar(labels, counts, align = 'center', width = 1, color = 'black')
        plt.plot([0, len(labels)], [Ncut, Ncut], '-', c = 'red', \
                 label = "$N_{cut}$ = " + str(Ncut))
        plt.xlabel('classes training data')
        plt.ylabel('log frequency')
        plt.title(str(Nclass) + " classes with more than " + str(Ncut)\
                  + ' samples')
        plt.legend()
        plt.yscale('log')
        plt.show()
        #----------------------------------------------------------------------


        #----------------------------------------------------------------------
        self.Y_sparse_train = Y_sparse_train
        self.Y_onehot_train = Y_onehot_train
        self.Nclass         = Nclass
        self.LabelsUnique   = LabelsUnique
        self.Encoded_train  = Encoded_train
        self.X_train        = X_train
        self.LabelsIdx      = LabelsIdx
        self.Filter         = Idx
        self.Labels         = Labels

        self.Y_sparse_test  = Y_sparse_test
        self.Y_onehot_test  = Y_onehot_test
        self.Encoded_test   = Encoded_test
        
        
    @lru_cache(maxsize = None)
    def RunLSTM(self, Nepochs = 200, n_neurons = 100, batch_size = 400):
        
        X                                 = self.X_train
        Y                                 = self.Y_onehot_train
        
        #####reducing X in order to make it computationally feasible###########
        [N_sample, _, _] = X.shape
        idx              = np.random.choice(N_sample, size = 1000,\
                                            replace = False)
        X = X[idx,:500,:]
        Y = Y[idx,:]
        #######################################################################
        
        [N_sample, LengthSeq, N_features] = X.shape
        
        model = Sequential()
        model.add(LSTM(n_neurons, activation = 'tanh',\
                       input_shape = (LengthSeq, N_features)))
        #model.add(LSTM(n_neurons, input_shape= (dt_past, N_features)))
        #model.add(Dropout(DA_rate))
        
        model.add(Dense(self.Nclass, activation = 'softmax'))
        
        opt = optimizers.Adam()
        model.compile(loss = 'categorical_crossentropy', optimizer = opt,\
                      metrics=['accuracy'])
        
        model.summary()
        
        print('running model...')
        out = model.fit(X, Y, epochs = Nepochs, batch_size = batch_size,\
                    validation_split = 0.2, #validation_data = (ValX, ValY),\
                    verbose = 2, shuffle = False)
        print('...fit completed')
        
        
        #plotting #############################################################
        plt.plot(out.history['accuracy'])
        plt.plot(out.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc = 'upper left')
        plt.savefig('training results.pdf')
        plt.show()

        plt.plot(out.history['loss'])
        plt.plot(out.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc = 'upper left')
        plt.savefig('training loss.pdf')
        plt.show()
        #######################################################################
        
        self.out     = out
        self.X_train = self.X_train[:,:500,:]


    @lru_cache(maxsize = None)
    def RunCNN(self, Nepochs = 20, batch_size = 400):
        
        X = self.X_train
        Y = self.Y_onehot_train
        
        X = X[:,:,:,np.newaxis]
        
        model = Sequential()
        model.add(Conv2D(24, kernel_size = (4, 4), strides = (2,2),\
                         activation = 'tanh', padding = 'valid',\
                         input_shape = X.shape[1:]))
        model.add(Flatten())
        model.add(Dense(84, activation = 'tanh'))
        model.add(Dense(self.Nclass, activation = 'softmax'))
        
        opt = optimizers.Adam()
        model.compile(loss ='categorical_crossentropy', optimizer = opt,\
                       metrics=['accuracy'])
        model.summary()
        
        print('running model...')
        out = model.fit(X, Y, epochs = Nepochs, batch_size = batch_size,\
                    validation_split = 0.2, verbose = 2)
        print('...fit completed')
            
        #plotting #############################################################
        plt.plot(out.history['accuracy'])
        plt.plot(out.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc = 'upper left')
        plt.savefig('training results.pdf')
        plt.show()

        plt.plot(out.history['loss'])
        plt.plot(out.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc = 'upper left')
        plt.savefig('training loss.pdf')
        plt.show()
        #######################################################################
        
        self.out = out
        
    @lru_cache(maxsize = None)
    def EvalModel(self):
        
        out       = self.out
        
        print('running prediction for evaluation...')
        probs     = out.model.predict(self.X_train)
        print('...prediction completed')
        
        Target    = list(self.LabelsIdx)
        ClassLabs = list(set(self.LabelsIdx))
        
        #----------------------------------------------------------------------
        #confusion matrix
        Y                = self.Y_onehot_train
        predicted_labels = np.argmax(probs, axis = 1)
        cm               = confusion_matrix(np.argmax(Y, axis = 1),\
                                            predicted_labels)

        plt.figure(figsize = (8, 6))
        sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',\
                    xticklabels = ClassLabs, yticklabels = ClassLabs)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        #----------------------------------------------------------------------
        
        
        #----------------------------------------------------------------------
        #entropy
        fig2, ax2 = plt.subplots(self.Nclass, 1, sharex = True)
        fig2.set_figheight(self.Nclass)
        fig2.subplots_adjust(hspace = 1.5)
        fig2.suptitle('entropy')
        
        for j, L in enumerate(ClassLabs):
            idx    = np.array([i for i, t in enumerate(Target) if t == L])
            
            #making sure we get the correct class assignment
            pclass   = probs[idx,:]
            Yclass   = Y[idx]
            idxclass = np.argmax(Yclass[0,:])
            (value, where) = np.histogram(pclass[:, idxclass],\
                                          bins = np.arange(0,1,0.01),\
                                          density = True)
            w = 0.5*(where[1:] + where[:-1])
            ax2[j].plot(w, value, 'k-')
            ax2[j].set_ylabel('frequency')
            ax2[j].set_title(L)
        ax2[self.Nclass-1].set_xlabel('probability')
        plt.show()
        #----------------------------------------------------------------------
        
        

        
        
        
        
        
        
        
        
        
        
        