import os
from pathlib import Path
import logging
import tempfile
import subprocess

import numpy as np

from miblab import pipe
import dbdicom as db
import vreg

from utils import data, dicom, dixon, nifti

# CONSTANTS
PIPELINE = data.get_pipeline(__file__) # get current pipeline


def autosegment_single(vol):

    print('Temporarily converting NumPy to Nifti')
    # Convert NumPy array into a Nifti image (TotalSpineSeg only takes nifti as input)
    nifti_object = nifti.convert_to_nifti(vol)

    # Perform autoseg on NIfTI image in RAM
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_input = os.path.join(tmpdir, "temp_input.nii.gz")
        temp_output = os.path.join(tmpdir, "temp_output") # output files will be saved with this as prefix
    
        vreg.write_nifti(vol, temp_input) # save NIfTI image to temp directory
        
        # TotalSpineSeg uses command-line interface ->        
        # define command as list of strings to input via subprocess
        command = [
            "totalspineseg",
            temp_input,
            temp_output,
            "--step1", # first model only (faster, includes spinal cord)
            "--keep-only", "step1_output", "step1_cord", # keeps spinal vertebrae and cord output only
           # "--max-workers", num_workers, # caps totalspineseg processes, unhash on hpc
           # "--max-workers-nnunet", num_workers, # caps underlying nnUNet processes, unhash on hpc
           # "--no-stalling" # sets multiprocessing method to "forkserver" to avoid deadlock issues - ONLY FOR HPC
        ]

        print('Running TotalSpineSeg')
        # Execute totalspineseg command using subprocess
        result = subprocess.run(command,
                                check=True, # catches explicit errors
                                text=True # capture logs in case of failure
                                )

        print('Obtaining autosegmented spinal cord and vertebrae files')
        # Obtain temp file paths of autosegmented spinal cord and vertebrae
        cord_file = os.path.join(temp_output, "step1_cord", "temp_input.nii.gz")
        vert_file = os.path.join(temp_output, "step1_output", "temp_input.nii.gz")

        # Reading the Nifti files
        cord = vreg.read_nifti(cord_file)
        vert = vreg.read_nifti(vert_file)

        print('Creating NumPy label map array')
        # Initialise blank integer label map array (filled with zeros)
        # matching dixon array size (e.g., 320 x 320 x 144)
        spine_labelmap = np.zeros(cord.shape, dtype=int)

        # Apply integer values sequentially (higher numbers overwrite lower if any overlap)
        # Label 1 = vertebrae (anywhere vertebrae array > 0)
        spine_labelmap[vert.values > 0] = 1
        # Label 2 = spinal cord (thresholding the soft probability mask at >80% confidence)
        # Gives every voxel identified as spinal cord a value of 1, 
        # & retreats from overlapping bone edges
        spine_labelmap[cord.values > 0.8] = 2
        
    # The directory and temp NIfTI files are erased here upon close of tempfile loop
    return spine_labelmap, cord, vert


def autosegment_all(dir_dixon, dir_output):
    
    print('Loading dixon series')
    # Load dixon DICOM series
    dixon_series = db.series(dir_dixon)

    # Read in dixon data csv containing correct references
    dixon_csv = os.path.join(os.getcwd(), 'src', 'data', 'dixon_data.csv')

    # If dixon_data.csv not already downloaded locally, download from GitHub source
    if not os.path.isfile(dixon_csv):
        dixon.download_record()
    
    record = dixon.dixon_record() # list selected dixon series

    # Loop through participant visits
    for series in dixon_series:

        print(f"Running on series: {series}")

        # Skip if series is not a dixon water series
        if 'water' not in series[-1][0]:
            print(f"{series} not a dixon water series - skipping!")
            continue
        
        # Extract participant ID and study & series descriptions
        participant, study, descr = dicom.get_attributes(series)

        # Check if autosegmentation already exists for this dataset
        if data.skip_existing(dir_output, participant, study) == True:
            logging.info(f"Series {participant} already exists --- skipping")
            continue

        # Skip if it is not the correct sequence in dixon sequence
        selected_sequence = dixon.dixon_series_desc(record, participant, study)
        if descr[:-6] != selected_sequence:
            continue

        dixon_vol = db.volume(series) # create dixon volume

        # create matching mask series
        mask_series = [dir_output, participant, (study, 0), (f"spine_masks", 0)]    

        print(f'Autosegmenting {participant} {study}')
        # Autosegment with TotalSpineSeg
        label_array, cord, vert = autosegment_single(dixon_vol)

        print('Saving label map array as DICOM')
        # Save TotalSpineSeg labelmap array as DICOM
        db.write_volume((label_array, dixon_vol.affine), mask_series)


def run(build, logfile):

    logging.info("Stage 1 --- Autosegment spinal cord ---")

    # Import external dixon (anatomical ref) pipeline
    dixon_ppln = data.get_external_buildpath(__file__, 'dixon', 'stage_5_clean_dixon_data')

    # Loop through sites
    for site in ['Leeds', 'Bari', 'Bordeaux', 'Exeter', 'Turku', 'Sheffield']:

        # Define site/patients totspineseg autosegment output folder
        patients_output = data.output_ibeat_dir(build, PIPELINE, __file__, 'Patients', site)

        # Define site/patients dixon input folder
        dir_dixon = os.path.join(dixon_ppln, 'Patients', site)

        print(f'Autosegmenting {site} patients')
        autosegment_all(dir_dixon, patients_output)
    
    # Define controls totspineseg autosegment output folder
    controls_output = data.output_ibeat_dir(build, PIPELINE, __file__, 'Controls')

    # Define controls dixon input folder
    dir_dixon = os.path.join(dixon_ppln, 'Controls')
    
    print(f'Autosegmenting controls')
    autosegment_all(dir_dixon, controls_output)


if __name__ == '__main__':

    # Define root build folder
    BUILD = data.get_ppln_buildpath(__file__)

    # Run totspineseg stage autosegment
    pipe.run_stage(run, BUILD, PIPELINE, __file__)
