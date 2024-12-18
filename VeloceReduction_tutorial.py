#!/usr/bin/env python
# coding: utf-8

# # VeloceReduction -- Tutorial
# 
# This tutorial provides an example on how to reduce data of a given night YYMMDD.

# In[ ]:


# Preamble
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
except:
    pass

# Basic packages
import numpy as np
from astropy.io import fits
from pathlib import Path
import sys
import argparse

# VeloceReduction modules and function
from VeloceReduction import config
from VeloceReduction.utils import identify_calibration_and_science_runs, polynomial_function
from VeloceReduction.extraction import extract_orders, extract_initial_order_ranges_and_coeffs
from VeloceReduction.calibration import calibrate_wavelength


# ## Adjust Date and Directory (possibly via argument parser)

# In[ ]:


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some inputs.")
    
    # Add arguments
    parser.add_argument('-d','--date', type=str, default="001122",
                        help='Date in the format DDMMYY (e.g., "001122")')
    parser.add_argument('-wd','--working_directory', type=str, default="./",
                        help='The directory where the script will operate.')
    
    # Parse the arguments
    args = parser.parse_args()
    return args

def get_script_input():
    if 'ipykernel' in sys.modules:
        # Assume default values if inside Jupyter
#         jupyter_date = "001122"
        jupyter_date = "231121"
        jupyter_working_directory = "./"
        print("Running in a Jupyter notebook. Using predefined values")
        args = argparse.Namespace(date=jupyter_date, working_directory=jupyter_working_directory)
    else:
        # Use argparse to handle command-line arguments
        print("Running as a standalone Python script")
        args = parse_arguments()

    return args

# Use the function to get input
args = get_script_input()
config.date = args.date
config.working_directory = args.working_directory
print(f"Date: {args.date}, Working Directory: {args.working_directory}")


# ## Identfiy Calibration and Science Runs

# In[ ]:


# Extract the Calibration and Science data from the night log
calibration_runs, science_runs = identify_calibration_and_science_runs(config.date, config.working_directory+'observations/')


# ## Extract orders and save in initial FITS files with an extension per order.

# In[ ]:


# Extract initial order ranges and coefficients
initial_order_ranges, initial_order_coeffs = extract_initial_order_ranges_and_coeffs()


# In[ ]:


# Extract Master Flat
print('Extracting Master Flat')
master_flat, noise = extract_orders(
    ccd1_runs = calibration_runs['Flat_60.0'][:1],
    ccd2_runs = calibration_runs['Flat_1.0'][:1],
    ccd3_runs = calibration_runs['Flat_0.1'][:1],
    Flat = True
)

# Extract Master ThXe
print('Extracting Master ThXe')
master_thxe, noise = extract_orders(
    ccd1_runs = calibration_runs['FibTh_180.0'][:1],
    ccd2_runs = calibration_runs['FibTh_60.0'][:1],
    ccd3_runs = calibration_runs['FibTh_15.0'][:1]
)

# Extract Master LC
print('Extracting Master LC')
master_lc, noise = extract_orders(
    ccd1_runs = calibration_runs['SimLC'][-1:],
    ccd2_runs = calibration_runs['SimLC'][-1:],
    ccd3_runs = calibration_runs['SimLC'][-1:],
    LC = True,
    # tramline_debug = True
)


# In[ ]:


# Extract Science Objects and save them into FITS files under reduced_data/
for science_object in list(science_runs.keys()):
    print('Extracting '+science_object)
    try:
        science, science_noise = extract_orders(
            ccd1_runs = science_runs[science_object],
            ccd2_runs = science_runs[science_object],
            ccd3_runs = science_runs[science_object],
            Science=True
        )

        # Create a primary HDU and HDU list
        primary_hdu = fits.PrimaryHDU()
        hdul = fits.HDUList([primary_hdu])

        # Loop over your extension names and corresponding data arrays
        for ext_index, ext_name in enumerate(initial_order_coeffs):
            # Create an ImageHDU object for each extension

            # Define the columns with appropriate formats
            col1_def = fits.Column(name='wave_vac',format='E', array=np.arange(len(science[ext_index,:]),dtype=float))
            col2_def = fits.Column(name='wave_air',format='E', array=np.arange(len(science[ext_index,:]),dtype=float))
            col3_def = fits.Column(name='science', format='E', array=science[ext_index,:])
            col4_def = fits.Column(name='science_noise',   format='E', array=science_noise[ext_index,:])
            col5_def = fits.Column(name='flat',    format='E', array=master_flat[ext_index,:])
            col6_def = fits.Column(name='thxe',    format='E', array=master_thxe[ext_index,:])
            col7_def = fits.Column(name='lc',      format='E', array=master_lc[ext_index,:])

            hdu = fits.BinTableHDU.from_columns([col1_def, col2_def, col3_def, col4_def, col5_def, col6_def, col7_def], name=ext_name.lower())

            # Append the HDU to the HDU list
            hdul.append(hdu)

        # Save to a new FITS file with an extension for each order
        Path(config.working_directory+'reduced_data/'+config.date+'/'+science_object).mkdir(parents=True, exist_ok=True)    
        hdul.writeto(config.working_directory+'reduced_data/'+config.date+'/'+science_object+'/veloce_spectra_'+science_object+'_'+config.date+'.fits', overwrite=True)

        print('Successfully extracted '+science_object)

    except:
        print('Failed to extract '+science_object)


# ## Wavelength calibration

# In[ ]:


for science_object in list(science_runs.keys()):
    try:
        calibrate_wavelength(science_object, create_overview_pdf=False)
        print('Succesfully calibrated wavelength without diagnostic plots for '+science_object)
    except:
        print('Failed to calibrate wavelength for '+science_object)


# In[ ]:


for science_object in list(science_runs.keys()):
    try:
        calibrate_wavelength(science_object, create_overview_pdf=True)
        print('Succesfully calibrated wavelength with diagnostic plots for '+science_object)
    except:
        print('Failed to calibrate wavelength for '+science_object)


# In[ ]:




