#######################
#
# Script to generate the local connectivity metrics from a given connectome
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
centrality = betweenness_bin(M)
efficiency = efficiency_bin(M,local=True)
m_clustering_coef = clustering_coef_bu(M)  
 
res = np.vstack((centrality,efficiency,m_clustering_coef))
res = res.T
print np.shape(res)

first_row = [subject_name + ":betweencentrality", "local_eff", "cluster_coef"]
with open("metrics_b.csv", "w") as output:
     writer = csv.writer(output, delimiter = ',', lineterminator='\n') #
     writer.writerow(first_row)
     for val in res:
        writer.writerow(val)  
print('data written') 
