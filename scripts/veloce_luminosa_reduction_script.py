#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
from veloce_luminosa_reduction import config
from veloce_luminosa_reduction.utils import identify_calibration_and_science_runs, match_month_to_date, read_veloce_fits_image_and_metadata, polynomial_function, radial_velocity_shift
from veloce_luminosa_reduction.reduction import substract_overscan, extract_initial_order_ranges_and_coeffs
from veloce_luminosa_reduction.calibration import get_wavelength_coeffs_from_vdarc
# from veloce_luminosa_reduction.order_merging import interpolate_orders_and_merge
# from veloce_luminosa_reduction.post_processing import degrade_resolution_with_uncertainty
# from veloce_luminosa_reduction.coadding import co_add_spectra

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
SSO = EarthLocation.of_site('Siding Spring Observatory')


# In[ ]:


initial_order_ranges, initial_order_coeffs = extract_initial_order_ranges_and_coeffs()


# In[ ]:


wavelength_coeffs = get_wavelength_coeffs_from_vdarc()


# In[ ]:


def process_objects(date, object_name, master_flats, science_runs):
    
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

#             if config.debug:
#                 f, gs = plt.subplots(3,3,figsize=(12,12))
            
            # Loop over CCDs
            for ccd in [1,2,3]:
            
                full_image, metadata = read_veloce_fits_image_and_metadata(config.working_directory+'raw_data/'+date+'/ccd_'+str(ccd)+'/'+date[-2:]+match_month_to_date(date)+str(ccd)+run+'.fits')

                object_coordinates = SkyCoord(ra = metadata['MEANRA'], dec = metadata['MEANDEC'], frame="icrs", unit="deg")
                vbary_corr_kms = object_coordinates.radial_velocity_correction( 
                    kind='barycentric', 
                    obstime = Time(val=metadata['UTMJD'],format='mjd', scale='utc'),
                    location=SSO
                ).to(u.km/u.s).value

                # Overscan Subtraction
                trimmed_image, os_median, os_rms = substract_overscan(full_image, metadata)

                # Flat Field Correction
                flat_corrected_image = np.array(trimmed_image, dtype=float) / np.array(master_flats['ccd_'+str(ccd)], dtype=float)
                
#                 if config.debug:

#                     ax = gs[0,ccd-1]
#                     ax.set_title(object_name+' CCD'+str(ccd))
#                     count_percentiles = np.percentile(trimmed_image, q=[20,95])
#                     s = ax.imshow(trimmed_image,cmap='OrRd',vmin=np.max([0,count_percentiles[0]]),vmax=count_percentiles[-1])
#                     cbar = plt.colorbar(s, ax=ax, extend='both', orientation='horizontal')
#                     cbar.set_label('Counts')
                    
#                     ax = gs[1,ccd-1]
#                     ax.set_title(object_name+' CCD'+str(ccd))
#                     count_percentiles = np.percentile(master_flats['ccd_'+str(ccd)], q=[20,95])
#                     s = ax.imshow(master_flats['ccd_'+str(ccd)],cmap='OrRd',vmin=np.max([0,count_percentiles[0]]),vmax=count_percentiles[-1])
#                     cbar = plt.colorbar(s, ax=ax, extend='both', orientation='horizontal')
#                     cbar.set_label('Counts')

#                     ax = gs[2,ccd-1]
#                     ax.set_title(object_name+' CCD'+str(ccd))
#                     count_percentiles = np.percentile(flat_corrected_image, q=[20,95])
#                     s = ax.imshow(flat_corrected_image,cmap='OrRd',vmin=np.max([0,count_percentiles[0]]),vmax=count_percentiles[-1])
#                     cbar = plt.colorbar(s, ax=ax, extend='both', orientation='horizontal')
#                     cbar.set_label('Counts')

#             if config.debug:
#                 plt.tight_layout()
#                 plt.show()
#                 plt.close()

                # Extract orders
                counts_in_orders = dict()
                for order in initial_order_coeffs:
                    if order[4] == str(ccd):
                    
                        order_xrange_begin = np.array(polynomial_function(initial_order_ranges[order],*initial_order_coeffs[order])-45,dtype=int)
                        order_xrange_end   = np.array(polynomial_function(initial_order_ranges[order],*initial_order_coeffs[order]),dtype=int)

                        order_counts = []
                        for x_index, x in enumerate(initial_order_ranges[order]):
                            order_counts.append(np.sum(flat_corrected_image[x,order_xrange_begin[x_index]:order_xrange_end[x_index]],axis=0))
                        counts_in_orders[order] = np.array(order_counts)

                # Calibrate Wavelengths
                raw_wavelengths_per_order = dict()
                vbary_corr_wavelengths_per_order = dict()
                for order in wavelength_coeffs.keys():
                    if order[4] == str(ccd):
                        if order[4] == '1':
                            wavelength_reference_pixels = 2170
                        if order[4] == '2':
                            wavelength_reference_pixels = 2170
                        if order[4] == '3':
                            wavelength_reference_pixels = 2473

                        # important for telluric correction
                        raw_wavelengths_per_order[order] = 10. * polynomial_function(np.arange(len(counts_in_orders[order]))-wavelength_reference_pixels,*wavelength_coeffs[order])

                        # important to achieve rest-wavelengths
                        vbary_corr_wavelengths_per_order[order] = radial_velocity_shift(
                            vbary_corr_kms,
                            raw_wavelengths_per_order[order]
                        )
                        
                        plt.figure(figsize=(15,3))
                        plt.title(order)
                        plt.plot(
                            vbary_corr_wavelengths_per_order[order],
                            counts_in_orders[order]
                        )
                        plt.show()
                        plt.close

    
#     # Normalize the spectrum
#     normalized_orders = normalize_spectrum(calibrated_data)
    
#     # Patch the orders to one spectrum
#     normalized_merged_spectrum = interpolate_and_merge(wavelengths, fluxes, uncertainties, linear_wavelengths)

#     # Output the final reduced and calibrated data
#     save_final_spectrum(normalized_merged_spectrum, args.night, args.object)

#     # TO BE ADDED WHEN WORKING WITH MULTIPLE OBSERVATIONS!
#     # Coadding of spectra
#     normalised_coadded_spectrum = co_add_spectra(normalized_merged_spectrum, common_wavelength)

#     # Degrade spectrum down to 4MOST HR
#     spectrum_4most_hr = degrade_resolution_with_uncertainty(wavelength, flux, flux_uncertainty, 80000, 22000)


        print('\nReduction of '+object_name+' complete')


# In[ ]:


def main():
    
    main_directory = os.getcwd()
    
    # Get all necessary information through parser (otherwise assume we are debugging in jupyter)
    try:
        parser = argparse.ArgumentParser(description='Reduce CCD images from the Veloce echelle spectrograph.', add_help=True)
        parser.add_argument('date', default='240219', type=str, help='Observation night in YYMMDD format.')
        parser.add_argument('object_name', default='HIP71683', type=str, help='Name of the science object.')
        parser.add_argument('-wd', '--working_directory', default=os.getcwd(), type=str, help='Directory of raw/reduced data as created by Veloce YYMMDD/ccd_1 etc.')
        parser.add_argument('--debug', action='store_true')
        args = parser.parse_args()
        config.date = args.date
        config.object_name = args.object_name
        config.working_directory = args.working_directory
        config.reduced_data_dir = args.reduced_data_dir
        config.debug = args.debug
    except:
        pass
    
    if config.debug:
        import matplotlib.pyplot as plt
    
    # Extract relevant information from night log
    calibration_runs, science_runs = identify_calibration_and_science_runs(config.date, config.working_directory+'raw_data/')
    
    # Master Flat
    master_flats = dict()
    for ccd in [1,2,3]:
        master_flats['ccd_'+str(ccd)] = []
        if ccd == 1: flat_runs = calibration_runs['Flat_60.0'][:7]
        if ccd == 2: flat_runs = calibration_runs['Flat_1.0'][:7]
        if ccd == 3: flat_runs = calibration_runs['Flat_0.1'][:7]
        # Read in, overscan subtract and append images to array
        for run in flat_runs:
            full_image, metadata = read_veloce_fits_image_and_metadata(config.working_directory+'raw_data/'+config.date+'/ccd_'+str(ccd)+'/'+config.date[-2:]+match_month_to_date(config.date)+str(ccd)+run+'.fits')
            trimmed_image, os_median, os_rms = substract_overscan(full_image, metadata)
            master_flats['ccd_'+str(ccd)].append(trimmed_image)
        # Normalise across runs.
        master_flats['ccd_'+str(ccd)] = np.array(np.median(master_flats['ccd_'+str(ccd)],axis=0),dtype=float)
        # Normalise to Median
        overall_median = np.nanmedian(master_flats['ccd_'+str(ccd)])
        # Ensure that values 0.0 or below are getting the normalising factor 1
        master_flats['ccd_'+str(ccd)][master_flats['ccd_'+str(ccd)] <= 0.0] = overall_median
        master_flats['ccd_'+str(ccd)] /= overall_median
        
#     if config.debug:
#         f, gs = plt.subplots(1,3,figsize=(12,3))
#         for ccd in [1,2,3]:    
#             ax = gs[ccd-1]
#             ax.set_title('Master Flat CCD'+str(ccd))
#             count_percentiles = np.percentile(master_flats['ccd_'+str(ccd)], q=[0,100])
#             s = ax.imshow(master_flats['ccd_'+str(ccd)],cmap='OrRd',vmin=np.max([0,count_percentiles[0]]),vmax=count_percentiles[-1])
#             cbar = plt.colorbar(s, ax=ax, extend='both', orientation='horizontal')
#             cbar.set_label('Counts')
#         plt.show()
#         plt.close()
    
    # Process all frames of a given object_name
    process_objects(config.date, config.object_name, master_flats, science_runs)

    print(f"\nReduction and calibration succesfull and complete.")


# In[ ]:


if __name__ == "__main__":
    main()

