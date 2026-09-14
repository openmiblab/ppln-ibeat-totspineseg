import os
import logging
from pathlib import Path

import numpy as np

import dbdicom as db
from miblab import pipe
from miblab_plot import mosaic_overlay

from utils import data, dicom, dixon

# CONSTANTS
PIPELINE = data.get_pipeline(__file__) # get current pipeline


def display_all(db_masks, db_mosaics, db_data):

    print('Loading dixon series')
    # Load dixon DICOM series
    dixon_series = db.series(db_data)

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

        # Check if display already exists for this dataset
        if data.skip_existing(db_mosaics, participant, study) == True:
            logging.info(f"Series {participant} already exists --- skipping")
            continue
        # Check if autosegmentation missing for this dataset
        if data.skip_missing(db_data, participant, study) == True:
            logging.info(f"Series {participant} missing --- skipping")
            continue

        # Skip if it is not the correct sequence in dixon sequence
        selected_sequence = dixon.dixon_series_desc(record, participant, study)
        if descr[:-6] != selected_sequence:
            continue

        dixon_vol = db.volume(series) # create dixon volume
        dixon_array = dixon_vol.values # create dixon vol array

        # Find matching autosegmented mask series
        mask_input = data.get_related_study(db_masks, series)
        mask_series = db.series(mask_input)

        # Loop through masks
        for mask in mask_series:

            # Skip if mask and dixon series do not match
            if series[-2][0] not in mask[-2][0]:
                print(f"{series[-2][0]} (dixon) does not match {mask[-2][0]} (mask)")
                continue
            
            png_file = os.path.join(db_mosaics, f"{participant}_{study}_{mask[-1][0]}.png")
            
            # Skip if file exists
            if os.path.exists(png_file):
                print('png already exists')
                continue
                    
            try:
                # Load arrays and build ROIs
                print('Loading mask array')
                mask_array = db.volume(mask).values

                SPINE_MAP = {  1: "vertebrae",
                                2: "spinal_cord",
                            }

                rois = {roi: (mask_array==idx).astype(np.int16) for idx, roi in SPINE_MAP.items()}

                # Build mosaic and log success
                print('building mosaics')
                mosaic_overlay(dixon_vol.values, rois, png_file, vmin=0, vmax=np.percentile(dixon_vol.values, 90), margin=[16,16,2], opacity=0.75)
                logging.info(f"Success building mosaic for {participant}, {study}, {mask[-1][0]}.")

            except:
                logging.exception(f"Error building mosaic for {participant}, {study}, {mask[-1][0]}.")


def run(build, logfile):

    logging.info("Stage 1 --- Create mosaic of mask overlays ---")

    # Import external dixon (anatomical ref) pipeline
    dixon_ppln = data.get_external_buildpath(__file__, 'dixon', 'stage_5_clean_dixon_data')
    
    # Loop through sites
    for site in ['Leeds', 'Bari', 'Bordeaux', 'Exeter', 'Turku', 'Sheffield']:
        
        # Define site/patients totspineseg autosegment input folder
        patients_input = data.input_ibeat_dir(build, PIPELINE, __file__, 'Patients', site, specific='01_auto_segment')
        # Define site/patients totspineseg display output folder
        patients_output = data.output_ibeat_dir(build, PIPELINE, __file__, 'Patients')

        # Define site/patients dixon input folder
        dir_dixon = os.path.join(dixon_ppln, 'Patients', site)

        print(f'Displaying {site} patients')
        display_all(patients_input, patients_output, dir_dixon)
    
    # Define controls totspineseg autosegment input folder
    controls_input = data.input_ibeat_dir(build, PIPELINE, __file__, 'Controls', specific='01_auto_segment')
    # Define controls totspineseg display output folder
    controls_output = data.output_ibeat_dir(build, PIPELINE, __file__, 'Controls')

    # Define controls dixon input folder
    dir_dixon = os.path.join(dixon_ppln, 'Controls')
    
    print(f'Displaying controls')
    display_all(controls_input, controls_output, dir_dixon)


if __name__=='__main__':

    # Define root build folder
    BUILD = data.get_ppln_buildpath(__file__)

    # Run totspineseg stage display
    pipe.run_stage(run, BUILD, PIPELINE, __file__)
