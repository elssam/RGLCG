#######################
#
# Script to generate the global connectivity metrics from a given connectome
#
#######################

import numpy as np

from bct.algorithms.clustering import  transitivity_wu,transitivity_bu, clustering_coef_bu
from bct.algorithms.distance import charpath, efficiency_wei, efficiency_bin
from bct.algorithms.modularity import community_louvain
from bct.algorithms.distance import   distance_wei 
from bct.algorithms.centrality import   betweenness_bin 
from bct.utils.other import weight_conversion
from bct.utils import invert

import csv

import pandas as pd

data = pd.read_csv('connectome.csv',header=None)
M = data.to_numpy()

#metrics
modularity = community_louvain(M)[1]
#Charpath needs some conversion
L = weight_conversion(M, 'lengths')
D = distance_wei( L )[1]
charpath,_,_,_,_ = charpath(D)
efficiency = efficiency_bin(M)
m_clustering_coef = transitivity_bu(M)  

outF = open("metrics_b.csv", "w")
outF.write("Subject, louvain, char_path_len, global_eff, transitivity \n")
outF.write( subject_name + ", " + str(modularity) + ", " + str(charpath) + ", " + str(efficiency) + ", " +str(m_clustering_coef) )
print('data written') 
 
