 #!/bin/bash  

 
################## Pre-processing DWI  ###############
echo " ################################################"
echo "Pre-processing DTI"

baseline=1
suffix="/"

for d in */ ; do
    echo "-------------------------------------------------"
    echo "entering subfolder" "$d"
    cd $d
       cd Axial_DTI
       for subd in */ ; do
           cd $subd
 
              for subdd in */ ; do
                  cd $subdd

                  #Convert DCM 2 NII
                  dcm2nii -4 y -n y -v y *.dcm
                  mv *.nii.gz original.nii.gz
                  mv *.bval original.bval
                  mv *.bvec original.bvec


                  echo "Skull stripping"
                  bet  original.nii.gz    brain.nii.gz -m 
                  fslmaths original.nii.gz   -mas brain_mask.nii.gz betted.nii.gz
  
                  echo "Eddy current correction"	 
                  eddy_correct betted.nii.gz  eddycorrected.nii.gz   -interp trilinear

                  echo "register the atlas using the generated matrix"
                  if [ -f eddycorrected_c.nii.gz ]; then
                     echo "Cropped file here"
                     flirt -ref  eddycorrected_c.nii.gz -in  ../../../../aal.nii -out atlas_reg.nii.gz -interp nearestneighbour  -cost mutualinfo 
                  else
                     echo "Using normal file"
                     flirt -ref  eddycorrected.nii.gz -in  ../../../../aal.nii -out atlas_reg.nii.gz -interp nearestneighbour -cost mutualinfo 
                  fi
                 
                  foo=${d%$suffix}
                  echo "Generate Connectome matrix"
                  cp ../../../../*.py .
                  python  compute_conn.py $foo


                  if [ $baseline -eq  "1"  ]; then
                     mv metrics_b.csv   "../../../../${foo}_baseline.csv"   
                     baseline=0 
                  else
                     mv metrics_b.csv   "../../../../${foo}_followup.csv"  
                     baseline=1
                  fi

                  cd ..
              done
           cd ..
       done  
       cd ..                 
    cd ..
done





              
