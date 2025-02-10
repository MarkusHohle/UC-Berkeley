# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:27:56 2023

@author: hohle
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#pip install BioPython
from scipy import spatial
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio import Phylo
#phylogenetic tree

from ReadMyFasta import ReadMyFasta
[N,S,ToAlign] = ReadMyFasta('cytochromeC.txt')

#check
ToAlign

calc = DistanceCalculator('blosum62', skip_letters=('.'))
#check
calc.models
DistMa = calc.get_distance(ToAlign)

#check
DistMa.names
DistMa.matrix
# Construct the phlyogenetic tree using UPGMA algorithm
constructor = DistanceTreeConstructor()
UPGMATree   = constructor.upgma(DistMa)
NJTree      = constructor.nj(DistMa)

#visualizing tree
Phylo.draw(UPGMATree)
Phylo.draw(NJTree)

#turning the dist matrix into a squreform
# 2 times because  vector-form distance vector to a square-form 
#distance matrix, and vice-versa.
D = spatial.distance.squareform(DistMa)
D = spatial.distance.squareform(D)

D_df = pd.DataFrame(D,columns = DistMa.names)


sns.heatmap(D_df, square = True,  cmap = "Blues", \
            yticklabels = DistMa.names)
plt.show()


#sns has some basic cluster capability
sns.clustermap(D_df, cmap = "Blues", row_cluster = True, col_cluster = True,\
               metric = 'euclidean', method = 'average', yticklabels = True,\
               xticklabels = True)
plt.show()


#clustering_algorithms = (
#        ("Single Linkage", single),
#        ("Average Linkage", average), --> like UPGMA
#        ("Complete Linkage", complete),
#        ("Ward Linkage", ward),













