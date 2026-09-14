import os
import logging
from pathlib import Path

import numpy as np
import dbdicom as db
import pydmr
import numpyradiomics as npr

from miblab import pipe

from utils import data, dicom

# CONSTANTS
PIPELINE = data.get_pipeline(__file__) # get current pipeline

SPINE_MAP = {  #1: "vertebrae",
                2: "spinal_cord",
            }


def get_binaries(mask_array, mask_label):

  binary_mask = (mask_array == mask_label).astype(np.float32)
  
  if np.sum(binary_mask) == 0:
    logging.error(f"{mask_label} - Error: binary mask is equal to zero")
    return
  else:
    return binary_mask


def slicewise_csa(mask_series, binary_mask):

    csas = {}

    spacing_x = db.values(mask_series, 'PixelSpacing')[0][0]
    spacing_y = db.values(mask_series, 'PixelSpacing')[0][1]

    pixel_area = spacing_x * spacing_y

    # Calculate true area for every single slice
    valid_slice_areas = []
    for i in range(binary_mask.shape[1]):
        voxel_count = np.sum(binary_mask[:, i, :])
        if voxel_count > 0:  # only count slices where spinal cord is present
            # Direct pixel-counting approach used by SCT
            csa = voxel_count * pixel_area
            valid_slice_areas.append(csa)
            csas[f'spinal_cord-CSA_slice{i}'] = [csa.item(), f'CSA slice{i} (spinal_cord)', 'mm^2', 'float']

    # Calculate median CSA
    median_csa = np.median(valid_slice_areas)
    print(f"Median CSA: {median_csa:.2f} mm²")

    csas[f'spinal_cord-CSA_median'] = [median_csa, f'CSA Median (spinal_cord)', 'mm^2', 'float']

    return csas


def combine(dir_output, prefix, convert=False):
    """
    Concatenate all dmri files in a folder into a single dmr file. 
    Create long and wide format csvs for export.
    """

    # Combine all dmr files into one
    folder = Path(dir_output)
    dmr_files = list(folder.rglob("*.dmr.zip"))

    if dmr_files != []:
        dmr_files = [str(f) for f in dmr_files]

        dmr_file = os.path.join(dir_output, f'{prefix}_all_masks')

        dmr_concat = pydmr.concat(dmr_files, dmr_file, cleanup=True)

        if convert==True:
            pydmr.pars_to_long(dmr_concat, os.path.join(dir_output, f'{prefix}_all_masks_long.csv'))
            pydmr.pars_to_wide(dmr_concat, os.path.join(dir_output, f'{prefix}_all_masks_wide.csv'))


def calculate_stats(mask_series, binary_mask, roi, participant, study, dmr_file):

    # Compute median slice-wise csa
    slicewise_csas = slicewise_csa(mask_series, binary_mask)

    # Create basic csas dmr
    dmr = {'data':{}, 'pars':{}}
    dmr['data'] = dmr['data'] | {f"{roi}-shape-{p}": u[1:] for p, u in slicewise_csas.items()}
    dmr['pars'] = dmr['pars'] | {(participant, study, f"{roi}-shape-{p}"): v[0] for p, v in slicewise_csas.items()}

    # Get radiomics shape features in mm
    spacing = db.values(mask_series, 'PixelSpacing')[0] + [db.values(mask_series, 'SliceThickness')[0]]
    results = npr.shape(binary_mask, spacing = spacing)
    units = npr.shape_units(3, 'mm')

    # Write to dmr file
    dmr_radiomics = {
                'data': {f"{roi}-shape-{p}": [f"Shape measure {p} for {roi}", u, 'float'] for p, u in units.items()},
                'pars': {(participant, study, f"{roi}-shape-{p}"): v for p, v in results.items()}
            }
    
    # Merge both dmrs
    merged_dmr = {k: dmr[k] | dmr_radiomics[k] for k in dmr}

    pydmr.write(dmr_file, merged_dmr)
    logging.info(f"Successfully computed shapes: {roi}")


def align_masks(mask_series, idx):
    
    mask_vol = db.volume(mask_series)

    # Read binary mask
    binary_mask = get_binaries(mask_vol.values, idx)

    return binary_mask


def measure_mask(mask_series, dir_output):

    participant, study, descr = dicom.get_attributes(mask_series)
    
    for idx, roi in SPINE_MAP.items():

        # Define outputs
        fname = f"{participant}_{study}_{roi}.dmr.zip"
        dmr_file = os.path.join(dir_output, fname)

        # Skip if output exists
        if os.path.exists(dmr_file):
            continue

        try:
            binary_mask = align_masks(mask_series, idx)
            calculate_stats(mask_series, binary_mask, roi, participant, study, dmr_file)
        except:
            logging.exception(f"Error computing shapes: {fname}")


def measure_all(dir_masks, dir_output, site=None):
    
    print('Loading mask series')
    # Load totspineseg masks DICOM series
    mask_series = db.series(dir_masks)

    # Loop through mask series
    for mask in mask_series:
        
        # Extract participant ID and study & series descriptions
        participant, study, descr = dicom.get_attributes(mask)

        # Check if measure already exists for this dataset
        if data.skip_existing(dir_output, participant, study) == True:
            logging.info(f"Series {participant} already exists --- skipping")
            continue

        measure_mask(mask, dir_output)
    
    if site == None:
        combine(dir_output, f"Controls_totspineseg", convert=True)
    else:
        combine(dir_output, f"totspineseg_{site}")


def run(build, logfile):

    logging.info("Stage 1 --- Measuring shape metrics ---")

    # Define patients totspineseg measure output folder
    patients_output = data.output_ibeat_dir(build, PIPELINE, __file__, 'Patients')

    # Loop through sites
    for site in ['Leeds', 'Bari', 'Bordeaux', 'Exeter', 'Turku', 'Sheffield']:

        # Define site/patients totspineseg autosegment input folder
        patients_input = data.input_ibeat_dir(build, PIPELINE, __file__, 'Patients', site, specific='01_auto_segment')

        print(f'Measuring {site} patients')
        measure_all(patients_input, patients_output, site = site)
    
    # combine site measurements into one dmr
    combine(patients_output, f"Patients_totspineseg", convert=True)

    # Define controls totspineseg autosegment input folder  
    controls_input = data.input_ibeat_dir(build, PIPELINE, __file__, 'Controls', specific='01_auto_segment')
    # Define controls totspineseg measure output folder
    controls_output = data.output_ibeat_dir(build, PIPELINE, __file__, 'Controls')

    print(f'Measuring controls')
    measure_all(controls_input, controls_output)


if __name__=='__main__':

    # Define root build folder
    BUILD = data.get_ppln_buildpath(__file__)

    # Run totspineseg stage measure
    pipe.run_stage(run, BUILD, PIPELINE, __file__)
