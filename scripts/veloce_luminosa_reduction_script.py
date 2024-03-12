#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
from veloce_luminosa_reduction.utils import identify_calibration_and_science_runs, match_month_to_date, read_veloce_fits_image_and_metadata
from veloce_luminosa_reduction.reduction import substract_overscan
from veloce_luminosa_reduction import config
# from veloce_luminosa_reduction.coadding import co_add_spectra

import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# # Create Master Calibration Frames
# master_bias = median_combine(bias_frames)
# bias_subtracted_darks = [dark - master_bias for dark in dark_frames]
# master_dark = median_combine(bias_subtracted_darks)
# bias_subtracted_flats = [flat - master_bias for flat in flatfield_frames]
# master_flat = median_combine(bias_subtracted_flats)
# master_flat /= np.median(master_flat)

# # Apply Master Calibration Frames
# bias_subtracted_science = science_frame - master_bias
# dark_scale_factor = science_exposure_time / dark_exposure_time
# scaled_master_dark = master_dark * dark_scale_factor
# dark_subtracted_science = bias_subtracted_science - scaled_master_dark
# flat_corrected_science = dark_subtracted_science / master_flat


# In[ ]:


def process_objects(date, object_name, calibration_runs, science_runs):
    
    # Create master calibration frames (flat, bias, dark, etc.)
#     master_frames = create_master_frames(fits_data['calibration'])
    
    if object_name == "all":
        object_names  = list(science_runs.keys())
    else:
        object_names = [object_name]
    
    for object_name in object_names:

        print('\nNow reducing '+object_name)
        
        # Loop over runs
        print(science_runs[object_name][1:])
        for run in science_runs[object_name][1:]:

            if config.debug:
                f, gs = plt.subplots(1,3,figsize=(12,5))
            
            # Loop over CCDs
            for ccd in [1,2,3]:
            
                full_image, metadata = read_veloce_fits_image_and_metadata(config.raw_data_dir+'/'+date+'/ccd_'+str(ccd)+'/'+date[-2:]+match_month_to_date(date)+str(ccd)+run+'.fits')

                trimmed_image, os_median, os_rms = substract_overscan(full_image, metadata)

                if config.debug:

                    ax = gs[ccd-1]
                    ax.set_title(object_name+' CCD'+str(ccd))
                    count_percentiles = np.percentile(trimmed_image, q=[20,95])
                    s = ax.imshow(trimmed_image,cmap='OrRd',vmin=np.max([0,count_percentiles[0]]),vmax=count_percentiles[-1])
                    cbar = plt.colorbar(s, ax=ax, extend='both', orientation='horizontal')
                    cbar.set_label('Counts')
                    
            if config.debug:
                plt.tight_layout()
                plt.show()
                plt.close()
    
#     # Read in FITS files based on the extracted paths
#     fits_data = read_fits_files(files_to_process)

#     # Reduce science observations using calibration frames
#     reduced_data = reduce_data(fits_data['science'], master_frames)

#     # Calibrate the wavelength of the reduced observations
#     calibrated_data = calibrate_wavelengths(reduced_data)
    
#     # Normalize the spectrum
#     normalized_orders = normalize_spectrum(calibrated_data)
    
#     # Patch the orders to one spectrum
#     normalized_merged_spectrum = merge_order(normalized_orders)

#     # Output the final reduced and calibrated data
#     save_final_spectrum(normalized_merged_spectrum, args.night, args.object)

#     # TO BE ADDED WHEN WORKING WITH MULTIPLE OBSERVATIONS!
#     # Coadding of spectra
#     normalised_coadded_spectrum = co_add_spectra(spectra, common_wavelength)

        print('\nReduction of '+object_name+' complete')


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
        config.date = args.date
        config.object_name = args.object_name
        config.raw_data_dir = args.raw_data_dir
        config.reduced_data_dir = args.reduced_data_dir
        config.debug = args.debug
    except:
        config.date = '240219'
        config.object_name = 'HIP71683'
        config.raw_data_dir = '/Users/buder/git/veloce_luminosa_reduction/raw_data'
        config.reduced_data_dir = '/Users/buder/git/veloce_luminosa_reduction/reduced_data'
        config.debug = True
    
    if config.debug:
        import matplotlib.pyplot as plt
    
    # Extract relevant information from night log
    calibration_runs, science_runs = identify_calibration_and_science_runs(config.date, config.raw_data_dir)
    
    # Process all frames of a given object_name
    process_objects(config.date, config.object_name, calibration_runs, science_runs)
    
    print(f"\nReduction and calibration succesfull and complete.")


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




