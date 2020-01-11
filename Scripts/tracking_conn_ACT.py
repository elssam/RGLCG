############# This is the version of the code which performs also 
#### the ACT (Anatomically-constrained tractography)
#
# WARNING: it requires Dipy 1.0 and Python 3.6
#
#####################################################################

# Enables/disables interactive visualization
#interactive = False
import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti#, load_nifti_data
from dipy.tracking.utils import random_seeds_from_mask
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import auto_response
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.stopping_criterion import CmcStoppingCriterion
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.tracking import utils

fimg = "DTICAP_bet.nii.gz"
img = nib.load(fimg)
data = img.get_data()
fbval = "DTICAP.bval"
fbvec = "DTICAP.bvec"
affine = img.affine

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

mask, S0_mask = median_otsu(data[:, :, :, 0])
# create seeds
seeds = random_seeds_from_mask(mask, affine, seeds_count=1)
 

"""
 fit the data to a Constant Solid Angle ODF Model. This model will estimate the
Orientation Distribution Function (ODF) at each voxel. The ODF is the
distribution of water diffusion as a function of direction. The peaks of an ODF
are good estimates for the orientation of tract segments at a point in the
image. Here, we use ``peaks_from_model`` to fit the data and calculated the
fiber directions in all voxels of the white matter.
"""
 
response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             mask=mask)
 

# Chose traditional FA stopping criteria or ACT
stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .2)
'''
voxel_size = np.average(img_pve_wm.header['pixdim'][1:4])
step_size = 0.2

# Load data from the segmentation results (e.g. FSL)
wm_file = "WM.nii.gz"
mg_pve_wm = nib.load(wm_file)
gm_file = "GM.nii.gz"
img_pve_gm = nib.load(gm_file)
csf_file = "CSF.nii.gz"
img_pve_csf = nib.load(csf_file)

cmc_criterion = CmcStoppingCriterion.from_pve(img_pve_wm.get_data(),
                                              img_pve_gm.get_data(),
                                              img_pve_csf.get_data(),
                                              step_size=step_size,
                                              average_voxel_size=voxel_size)
''' 
""" 
Perform  EuDX algorithm [Garyfallidis12]_. ``EuDX`` [Garyfallidis12]_ is a fast
algorithm that we use here to generate streamlines. This algorithm is what is
used here and the default option when providing the output of peaks directly
in LocalTracking.
""" 
# Initialization of LocalTracking. The computation happens in the next step.
streamlines_generator = LocalTracking(csa_peaks, stopping_criterion,  seeds=seeds,
                                      affine=affine, step_size=.5)
# Generate streamlines object
streamlines = Streamlines(streamlines_generator)

# Remove streamlines shorter than 3 cm
lengths = list(length(streamlines))
long_streamlines = Streamlines()
for i, sl in enumerate(streamlines):
    if lengths[i] > 30:
        long_streamlines.append(sl)


sft = StatefulTractogram(long_streamlines, img, Space.RASMM)
save_trk(sft, "tractogram_EuDX.trk", streamlines)

# Generate connectome
atlas = nib.load('atlas_reg.nii.gz')
labels = atlas.get_data()
labelsint = labels.astype(int)
M, grouping = utils.connectivity_matrix(streamlines, affine, labelsint, return_mapping=True, mapping_as_streamlines=True )
#            utils.connectivity_matrix(streamlines, label_volume=labelsint.astype(int), affine=affine, return_mapping=False, mapping_as_streamlines=False)
#Remove background
M = M[1:,1:]
#Remove the last rows and columns since they are cerebellum and brainstem
M = M[:90,:90]
np.fill_diagonal(M,0)
np.savetxt("foo.csv", M, delimiter=",")
