#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
from veloce_luminosa_reduction.utils import match_month_to_date, identify_calibration_and_science_runs


# In[ ]:


def main():
    
    main_directory = os.getcwd()
    
    # Get all necessary information through parser (otherwise assume we are debugging in jupyter)
    try:
        parser = argparse.ArgumentParser(description='Reduce CCD images from the Veloce echelle spectrograph.', add_help=True)
        parser.add_argument('date', default='240219', type=str, help='Observation night in YYMMDD format.')
        parser.add_argument('object_name', default='HIP71683', type=str, help='Name of the science object.')
        parser.add_argument('-in', '--raw_data_dir', default=main_directory+'../raw_data', type=str, help='Directory of raw data as created by Veloce YYMMDD/ccd_1 etc.')
        parser.add_argument('-out', '--reduced_data_dir', default=main_directory+'../reduced_data', type=str, help='Directory of raw data as created by Veloce YYMMDD/ccd_1 etc.')
        parser.add_argument('--debug', action='store_true')
        args = parser.parse_args()
        date = args.date
        object_name = args.object_name
        raw_data_dir = args.raw_data_dir
        reduced_data_dir = args.reduced_data_dir
        debug = args.debug
    except:
        date = '240219'
        object_name = 'HIP71683'
        raw_data_dir = '/Users/buder/git/veloce_luminosa_reduction/raw_data'
        reduced_data_dir = '/Users/buder/git/veloce_luminosa_reduction/reduced_data'
        debug = True

    print(date, object_name, debug)

    month = match_month_to_date(date)

    calibration_runs, science_runs = identify_calibration_and_science_runs(date, raw_data_dir)

#     # Step 1: Read in the night log and extract relevant file paths
#     # (You'll need to implement this function in your io module)
#     files_to_process = read_night_log(args.night, args.object)
    
#     # Step 2: Read in FITS files based on the extracted paths
#     # (This function also needs to be implemented in your io module)
#     fits_data = read_fits_files(files_to_process)
    
#     # Step 3: Create master calibration frames (flat, bias, dark, etc.)
#     # (Implement this in your calibration module)
#     master_frames = create_master_frames(fits_data['calibration'])
    
#     # Step 4: Reduce science observations using calibration frames
#     # (This function is part of your reduction module)
#     reduced_data = reduce_data(fits_data['science'], master_frames)
    
#     # Step 5: Calibrate the wavelength of the reduced observations
#     # (Part of your spectral_calibration module)
#     calibrated_data = calibrate_wavelengths(reduced_data)
    
#     # Step 6: Normalize the spectrum
#     # (Implement this in your normalization module)
#     normalized_orders = normalize_spectrum(calibrated_data)
    
#     # Step 7: Patch the orders to one spectrum
#     # (Implement this in your order_merging module)
#     normalized_merged_spectrum = merge_order(normalized_orders)
    
#     # Step 8: Output the final reduced and calibrated data
#     # This might involve saving to a new FITS file or another format
#     # You will need to define this function as well
#     save_final_spectrum(normalized_merged_spectrum, args.night, args.object)
    
#     print(f"Reduction and calibration complete for {args.object} on night {args.night}.")


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




