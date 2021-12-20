# RGLCG
Relating Global and Local Connectome Changes to Dementia and Targeted Gene Expressions in Alzheimer’s Disease

This repo comprises the code for the manuscript: 
* Elsheikh SSM, Chimusa ER, Alzheimer's Disease Neuroimaging Initiative, Mulder NJ and Crimi A (2021) Relating Global and Local Connectome Changes to Dementia and Targeted Gene Expression in Alzheimer's Disease. Front. Hum. Neurosci. 15:761424. doi: [10.3389/fnhum.2021.761424](https://doi.org/10.3389/fnhum.2021.761424)

*For a brief summary of the paper, check out this [video](https://www.youtube.com/watch?v=bDPNtb5bFZc)*

This work uses data from the Alzheimer's Disease Neuroimaging Initiative publicly available

The image data are processed accoding to the pipeline reported in the script pre_process.sh Tractography, connectome creation and feature extraction is given in the scripts compute_conn_CSA.py, tracking_conn_ACT.py, metrics.py and metrics.py. Those scripts are inside the folder Scripts.

The main bioinformatics analyis is in the scipt analysis.py


![image](https://github.com/elssam/RGLCG/blob/master/radiogenomics2.png)

# Download the Data
The data extracted from MRI volumes  needed in this analysis are accessable in the [Data folder](https://github.com/elssam/RGLCG/tree/master/Data)
Those are the resulting connectomes, and the features.
While the neuroimaging scripts are inside  [this folder](https://github.com/elssam/RGLCG/tree/master/Scripts)
