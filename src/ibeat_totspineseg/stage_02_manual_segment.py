################################################

############## UNDER CONSTRUCTION ##############

# currently pydantic error with python=3.10

################################################
import logging
import os

import numpy as np

from miblab import pipe
import dbdicom as db

from utils import data, dicom, dixon, edit

# CONSTANTS
PIPELINE = data.get_pipeline(__file__) # get current pipeline


def get_binaries(mask_array, mask_label):

  binary_mask = (mask_array == mask_label).astype(np.float32)
  
  if np.sum(binary_mask) == 0:
    logging.error(f"{mask_label} - Error: binary mask is equal to zero")
    return
  else:
    return binary_mask


def manually_segment(anatomical_vol, full_mask_array, mask_label, binaries=False):
        
    if binaries == True:
        mask_array = get_binaries(full_mask_array, mask_label)
    else:
        mask_array = full_mask_array

    # Manually segment/edit mask
    mask_edited = edit.mask_with_napari(anatomical_vol.values, mask_array, mask_label)
    
    return mask_edited


def get_mask_output(dir_output, series, roi = None):

    if roi == None:
        mask_output = [dir_output, series[1], series[2], series[-1]]
    else:
        mask_output = [dir_output, series[1], series[2], (roi, 0)]

    return mask_output


def save_mask(mask_array, anatomical_vol, mask_output, anatomical_series):

    # Create vreg volume of manually segmented/edited mask array
    mask_vol = (mask_array.astype(np.int16), anatomical_vol.affine)

    # Save manually segmented/edited mask as DICOM
    db.write_volume(mask_vol, mask_output, ref=anatomical_series)


def edit_all(dir_input, dir_output, dir_dixon):

    print('Loading dixon series')
    # Load dixon DICOM series
    dixon_series = db.series(dir_dixon)

    record = dixon.dixon_record()  # list selected dixon series

    # Loop through participant visits
    for series in dixon_series:

        print(f"Running on series: {series}")

        # Skip if series is not a dixon water series
        if 'water' not in series[-1][0]:
            print(f"{series} not a dixon water series - skipping!")
            continue

        # Extract participant ID and study & series descriptions
        participant, study, descr = dicom.get_attributes(series)

        # Check if manual segmentation already exists for this dataset
        if data.skip_existing(dir_output, participant, study) == True:
            logging.info(f"Series {participant} already exists --- skipping")
            continue
        # Check if autosegmentation missing for this dataset
        if data.skip_missing(dir_dixon, participant, study) == True:
            logging.info(f"Series {participant} missing --- skipping")
            continue

        # Skip if it is not the correct sequence in dixon sequence
        selected_sequence = dixon.dixon_series_desc(record, participant, study)
        if descr[:-6] != selected_sequence:
            continue

        dixon_vol = db.volume(series) # create dixon volume

        # Find matching autosegmented mask series
        mask_input = data.get_related_study(dir_input, series)
        mask_series = db.series(mask_input)

        # Loop through masks
        for mask in mask_series:
            logging.info("Stage 2 --- Manually edit spinal cord segmentations ---")

            # Skip if mask and dixon series do not match
            if series[-2][0] not in mask[-2][0]:
                print(f"{series[-2][0]} (dixon) does not match {mask[-2][0]} (mask)")
                continue

            full_mask_array = db.volume(mask).values

            SPINE_MAP = {  1: "vertebrae",
                            2: "spinal_cord",
                        }

            # to increase cursor size: viewer.layers.selection.active.brush_size = 500
            
            logging.info(f"Opening napari for: {SPINE_MAP[2]} (Label ID: Spinal Cord)")
            full_mask_array = manually_segment(dixon_vol, full_mask_array, 2)
                
            mask_output = get_mask_output(dir_output, mask)
            save_mask(full_mask_array, dixon_vol, mask_output, series)


def run(build, logfile):

    logging.info("Stage 1 --- Read in anatomical reference series ---")

    # Import external dixon (anatomical ref) pipeline
    dixon_ppln = data.get_external_buildpath(__file__, 'dixon', 'stage_5_clean_dixon_data')

    # Loop through sites
    for site in ['Leeds', 'Bari', 'Bordeaux', 'Exeter', 'Turku', 'Sheffield']:

        # Define site/patients totspineseg autosegment input folder
        patients_input = data.input_ibeat_dir(build, PIPELINE, __file__, 'Patients', site)
        # Define site/patients totspineseg manual segment output folder
        patients_output = data.output_ibeat_dir(build, PIPELINE, __file__, 'Patients', site)

        # Define site/patients dixon input folder
        dir_dixon = os.path.join(dixon_ppln, 'Patients', site)

        print(f'Editing {site} patients')
        edit_all(patients_input, patients_output, dir_dixon)
    
    # Define controls totspineseg autosegment input folder
    controls_input = data.input_ibeat_dir(build, PIPELINE, __file__, 'Controls')
    # Define controls totspineseg manual segment output folder
    controls_output = data.output_ibeat_dir(build, PIPELINE, __file__, 'Controls')

    # Define controls dixon input folder
    dir_dixon = os.path.join(dixon_ppln, 'Controls')
    
    print(f'Editing controls')
    edit_all(controls_input, controls_output, dir_dixon)


if __name__ == '__main__':

    # Define root build folder
    BUILD = data.get_ppln_buildpath(__file__)

    # Run totspineseg stage manual segment/edit
    pipe.run_stage(run, BUILD, PIPELINE, __file__)
