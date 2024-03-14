#!/usr/bin/env python
# coding: utf-8

# In[ ]:


try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
except:
    pass

import argparse
from pathlib import Path
import os

from veloce_luminosa_reduction import config

from veloce_luminosa_reduction.plotting import plot_ccd_imshow, create_rainbow_colormap, create_transparent_greyscale_colormap
from veloce_luminosa_reduction.utils import identify_calibration_and_science_runs, match_month_to_date, read_veloce_fits_image_and_metadata, polynomial_function, radial_velocity_shift, interpolate_spectrum
from veloce_luminosa_reduction.reduction import substract_overscan, extract_initial_order_ranges_and_coeffs
from veloce_luminosa_reduction.calibration import get_wavelength_coeffs_from_vdarc
from veloce_luminosa_reduction.korg_synthesis import calculate_synthetic_korg_spectrum
from veloce_luminosa_reduction.post_processing import degrade_resolution_with_uncertainty, interpolate_orders_and_merge, coadd_spectra

import time
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
SSO = EarthLocation.of_site('Siding Spring Observatory')


# In[ ]:


# Preparation for Veloce Luminosa
gbs_values = Table.read('../veloce_luminosa_reduction/literature_data/gbs_v3_and_v2.1_joined.fits')

# Telluric (as well as solar and arcturus) reference spectrum
# with dispersion of 0.0045 Å/px @3726.7Å to 0.0112 Å/px @9300.0Å
solar_arcturus_telluric_atlas = Table.read('../veloce_luminosa_reduction/literature_data/hinkle_2000_atlas_2000vnia.book..../solar_arcturus_telluric_atlas.fits')
solar_arcturus_telluric_atlas['TELLURIC'][solar_arcturus_telluric_atlas['TELLURIC'] <= 0.01] = 0.01
solar_arcturus_telluric_atlas['telluric_r80000'], dummy = degrade_resolution_with_uncertainty(
    solar_arcturus_telluric_atlas['WAVELENGTH'],
    solar_arcturus_telluric_atlas['TELLURIC'], 
    flux_uncertainty = 0.001 * np.ones(len(solar_arcturus_telluric_atlas['TELLURIC'])),
    original_resolution = 150000.,
    target_resolution = 80000.
)


# In[ ]:


def process_objects(date, object_name, master_flats, science_runs):
    """
    The function that runs through all observations for science_runs matches object_name.
    object_name can also be "all" in which case the function will run through all science_runs.keys()
    
    The function follows the following workflow:
    1) Create a synthetic Korg spectrum (if config.teff/logg/fe_h given or there is a match with the gbs_values)
    2) Read in FITS file for each CCD, subtract overscan, flat-field correct, estimate barycentric velocity
    3) Extract Orders from 2D image to 1D
    4) Calibrate Wavelength px->Å
    5) Normalise Orders with synthetic Korg spectrum
    ...
    N) Produce a rainbow plot

    INPUT:
    - date (str): The observation date in YYMMDD format.
    - object_name (str): The name of the target object matching keys in `science_runs`; use "all" to process data for all available objects.
    - master_flats (dict): A dictionary with Master Flat fields, keyed by 'ccd_X' for each CCD X, where X is in the range [1,2,3].
    - science_runs (dict): A dictionary where each key is associated with an array of run identifiers (e.g., [RRRR, RRRR, ...]).
    """
    if object_name == "all":
        object_names  = list(science_runs.keys())
    else:
        object_names = [object_name]
        
    # Initial wavelenght solutions by C. Tinney are optimised for specific reference pixels
    initial_order_ranges, initial_order_coeffs = extract_initial_order_ranges_and_coeffs()
    initial_wavelength_coeffs = get_wavelength_coeffs_from_vdarc()
    # Below we adjust the initial values to be even better before we would fit
    initial_order_references_pixels = dict()
    for order in initial_wavelength_coeffs.keys():
        if order[4] == '1':
            initial_order_references_pixels[order] = 2130
            if int(order[-3:]) >= 153:
                initial_order_references_pixels[order] = 2500
            if int(order[-3:]) >= 160:
                initial_order_references_pixels[order] = 2420
        if order[4] == '2':
            initial_order_references_pixels[order] = 2385
        if order[4] == '3':
            initial_order_references_pixels[order] = 2435

    for object_name in object_names:
        
        print('\nNow reducing '+object_name)
        print('    Available runs: ',science_runs[object_name])
        
        # Check if we have the necessary stellar parameters for a Korg synthesis
        # parsed as arguments or in the GBS reference list
        if config.teff is not None:
            teff = config.teff
            logg = config.logg
            fe_h = config.fe_h
            
            print('\n    Using parsed Teff/logg/[Fe/H] for '+object_name+': Teff='+str(teff)+', logg='+str(logg)+', [Fe/H]='+str(fe_h))

        elif object_name in gbs_values['hip']:
            object_name_match_in_gbs = np.where(object_name == gbs_values['hip'])[0][0]
            teff = gbs_values['teff'][object_name_match_in_gbs]
            logg = gbs_values['logg'][object_name_match_in_gbs]
            fe_h = gbs_values['fe_h'][object_name_match_in_gbs]
            
            print('\n    Found matching Teff/logg/[Fe/H] for '+object_name+' in GBS: Teff='+str(teff)+', logg='+str(logg)+', [Fe/H]='+str(fe_h))
            print('    Calculating Korg spectrum from 3900-9500Å')
            time_start = time.time()
            korg_synthesis, korg_wavelength_vac, korg_wavelength_air, korg_flux = calculate_synthetic_korg_spectrum(teff, logg, fe_h)
            print('    Calculated Korg spectrum in '+"{:.1f}".format(time.time() - time_start)+'s')
            
            # Interpolate the telluric spectrum onto the Korg wavelength grid.
            # Note: Telluric is air, so we are using the Korg air
            telluric_on_korg_wavelength = interpolate_spectrum(
                solar_arcturus_telluric_atlas['WAVELENGTH'],
                solar_arcturus_telluric_atlas['telluric_r80000'].clip(min=0.1),
                korg_wavelength_air
            )

            if object_name == 'HIP71683':
                radial_velocity = -24.7
                # shift Korg onto observed wavelength scale!
                korg_wavelength_vac = radial_velocity_shift(-radial_velocity, korg_wavelength_vac)
            if object_name == 'HIP76976':
                radial_velocity = -170.1
                # shift Korg onto observed wavelength scale!
                korg_wavelength_vac = radial_velocity_shift(-radial_velocity, korg_wavelength_vac)
            
        else:
            # safety mechanism in case Korg available, but no stellar parameters
            config.use_korg = False
            teff = None; logg = None; fe_h = None
            print('    No Teff/logg/[Fe/H] available. Not using Korg synthesis to normalise spectra')
            
        ########################################
        # Loop over runs
        for run in science_runs[object_name][1:]:
            
            ########################################
            # Flat Field Correction
            print('    Reading FITS for run '+str(run)+' and correcting with flat field')
            flat_corrected_image = dict()

            if config.debug:
                f, gs = plt.subplots(1,3,figsize=(12,3.5))
            
            for ccd in [1,2,3]:
            
                # Read in Science Image
                full_image, metadata = read_veloce_fits_image_and_metadata(config.working_directory+'raw_data/'+date+'/ccd_'+str(ccd)+'/'+date[-2:]+match_month_to_date(date)+str(ccd)+run+'.fits')
                
                # Overscan Subtraction
                trimmed_image, os_median, os_rms = substract_overscan(full_image, metadata)

                # Flat Field Correction
                flat_corrected_image['ccd_'+str(ccd)] = np.array(trimmed_image, dtype=float) / np.array(master_flats['ccd_'+str(ccd)], dtype=float)
                
                # Plot Flat Field Corrected Image
                if config.debug:
                    plot_ccd_imshow(ax=gs[ccd-1], image = flat_corrected_image['ccd_'+str(ccd)], panel_title = object_name+' ('+str(run)+') CCD'+str(ccd)+' FF-corrected')
            if config.debug:
                plt.tight_layout()
                Path(config.working_directory+'reduced_data/'+config.date+'/diagnostic_plots/'+str(run)).mkdir(parents=True, exist_ok=True)
                plt.savefig(config.working_directory+'reduced_data/'+config.date+'/diagnostic_plots/'+str(run)+'/'+config.date+'_'+str(run)+'_oscantrimmed_ffcorrected.pdf',dpi=200,bbox_inches='tight')
                plt.close()
                
            ########################################
            # Estimate Barycentric Velocity Correction
            object_coordinates = SkyCoord(ra = metadata['MEANRA'], dec = metadata['MEANDEC'], frame="icrs", unit="deg")
            vbary_corr_kms = object_coordinates.radial_velocity_correction( 
                kind='barycentric', 
                obstime = Time(val=metadata['UTMJD'],format='mjd', scale='utc'),
                location=SSO
            ).to(u.km/u.s).value

            ########################################
            print('    Extracting Orders')
            # Extract Orders (2D -> 1D spectra)
            counts_in_orders = dict()
            for order in initial_order_coeffs:
                ccd = order[4]
                # initial_order_ranges[order] are the initial orders reported by C.Tinney.
                # # Let's use the full range of the trimmed image
                order_xrange_begin = np.array(polynomial_function(np.arange(np.shape(flat_corrected_image['ccd_'+str(ccd)])[0]),*initial_order_coeffs[order])-45,dtype=int)
                order_xrange_end   = np.array(polynomial_function(np.arange(np.shape(flat_corrected_image['ccd_'+str(ccd)])[0]),*initial_order_coeffs[order]),dtype=int)
                order_counts = []
                for x_index, x in enumerate(initial_order_ranges[order]):
                    order_counts.append(np.sum(flat_corrected_image['ccd_'+str(ccd)][x,order_xrange_begin[x_index]:order_xrange_end[x_index]],axis=0))
                counts_in_orders[order] = np.array(order_counts)
                        
            ########################################
            # Calibrate Order Wavelengths
            print('    Calibration Order Wavelengths')
            raw_wavelengths_per_order = dict()
            vbary_corr_wavelengths_per_order = dict()
            
            for order in initial_wavelength_coeffs.keys():
                
                # Important for telluric correction
                # Note: coefficients above are reported for nm-scale, not Å-scale!
                raw_wavelengths_per_order[order] = 10. * polynomial_function(np.arange(len(counts_in_orders[order]))-initial_order_references_pixels[order],*initial_wavelength_coeffs[order])

                # Apply barycentric velocity correction
                vbary_corr_wavelengths_per_order[order] = radial_velocity_shift(
                    vbary_corr_kms,
                    raw_wavelengths_per_order[order]
                )

            ########################################
            # Normalise Orders
            print('    Normalising Orders')
            flux_in_orders = dict()
            korg_flux_in_orders = dict()
            korg_flux_with_tellurics_in_orders = dict()
            for order in initial_wavelength_coeffs.keys():
                # 1) If we use Korg
                if config.use_korg:
                    korg_flux_in_orders[order] = np.array(interpolate_spectrum(
                        korg_wavelength_vac,
                        korg_flux,
                        vbary_corr_wavelengths_per_order[order]
                    ))

                    # Shift telluric lines by -vbary_corr and the interpolate on vbary_corrected grid
                    telluric_flux_in_order = np.array(interpolate_spectrum(
                        radial_velocity_shift(
                            -vbary_corr_kms,
                            korg_wavelength_vac
                        ),
                        telluric_on_korg_wavelength,
                        vbary_corr_wavelengths_per_order[order]
                    ))
                    # Now multiply korg flux and the telluric spectrum that was shifted by -vbary_corr
                    korg_flux_with_tellurics_in_orders[order] = korg_flux_in_orders[order] * telluric_flux_in_order

                    flux_ratio = counts_in_orders[order] / korg_flux_in_orders[order]

                    # Identify absorption peaks above Xth percentile and also neglect first and last Y pixels for fitting
                    outlier_percentile = 10
                    edge_cut = 100
                    outlier_percentiles = np.percentile(flux_ratio,q=[outlier_percentile,100-outlier_percentile])
                    absorption_pixels = flux_ratio > outlier_percentiles[1]
                    absorption_pixels[flux_ratio < outlier_percentiles[0]] = True
                    absorption_pixels[:edge_cut] = True
                    absorption_pixels[-edge_cut:] = True

                    filter_kernel_size = 101
                    smooth_flux_ratio = medfilt(flux_ratio[~absorption_pixels], kernel_size=filter_kernel_size)

                    chebychev_degree = 5
                    chebychev_fit = Chebyshev.fit(vbary_corr_wavelengths_per_order[order][~absorption_pixels], smooth_flux_ratio, deg=chebychev_degree)

                    flux_in_orders[order] = counts_in_orders[order] / chebychev_fit(vbary_corr_wavelengths_per_order[order])

                    if config.debug:
                        f, gs = plt.subplots(2,1,figsize=(15,6),sharex=True)
                        
                        ax = gs[0]
                        ax.set_title(config.date+' '+str(run)+' '+str(object_name)+' CCD'+order[4]+' Order '+order.split('_')[-1]+' ('+str(initial_order_ranges[order][0])+','+str(initial_order_ranges[order][-1])+') Ref: '+str(initial_order_references_pixels[order]))
                        ax.plot(
                            vbary_corr_wavelengths_per_order[order],
                            flux_ratio,
                            label = 'Flux Ratio'
                        )
                        ax.plot(
                            vbary_corr_wavelengths_per_order[order][~absorption_pixels],
                            flux_ratio[~absorption_pixels],
                            label = 'Flux Ratio (w/o top/bottom '+str(outlier_percentile)+'th perc & '+str(edge_cut)+' px edge cut)'
                        )
                        ax.plot(
                            vbary_corr_wavelengths_per_order[order][~absorption_pixels],
                            smooth_flux_ratio,
                            label = 'Smoothed Flux Ratio (Kernel Size: '+str(filter_kernel_size)+')'
                        )
                        ax.plot(
                            vbary_corr_wavelengths_per_order[order],
                            chebychev_fit(vbary_corr_wavelengths_per_order[order]),
                            label = 'Chebychev Fit to Smooth Ratio ('+str(chebychev_degree)+' degrees)'
                        )
                        ax.legend(ncol=4)

                        ax = gs[1]
                        ax.set_title(config.date+' '+str(run)+' '+str(object_name)+' CCD'+order[4]+' Order '+order.split('_')[-1]+' ('+str(initial_order_ranges[order][0])+','+str(initial_order_ranges[order][-1])+') Ref: '+str(initial_order_references_pixels[order]))
                        ax.plot(
                            vbary_corr_wavelengths_per_order[order],
                            flux_in_orders[order],
                            label = 'Veloce', c = 'k', lw=1
                        )
                        ax.plot(
                            vbary_corr_wavelengths_per_order[order],
                            korg_flux_in_orders[order],
                            label = 'Korg', c='C0', lw=1
                        )
                        ax.plot(
                            vbary_corr_wavelengths_per_order[order],
                            korg_flux_with_tellurics_in_orders[order],
                            label = 'Korg w/ tellurics', c='C4', lw=1
                        )
                        ax.legend(ncol=3)
                        ax.set_xlabel(r'Wavelength (vacuum) / $\mathrm{\AA}$')
                        ax.set_ylabel(r'Flux / norm.')
                        ax_upper = ax.twiny()
                        wavelength_indices = np.linspace(100,len(vbary_corr_wavelengths_per_order[order])-100,5,dtype=int)
                        ax_upper.set_xticks(
                            wavelength_indices/(1.0*len(vbary_corr_wavelengths_per_order[order])),
                            wavelength_indices
                        )
                        ax.set_ylim(0.0,1.2)
                        plt.tight_layout()
                        Path(config.working_directory+'reduced_data/'+config.date+'/diagnostic_plots/'+str(run)).mkdir(parents=True, exist_ok=True)
                        plt.savefig(config.working_directory+'reduced_data/'+config.date+'/diagnostic_plots/'+str(run)+'/'+config.date+'_'+str(run)+'_'+order+'_normed_flux.pdf',dpi=200,bbox_inches='tight')
                        plt.close()

#     # Patch the orders to one spectrum
#     normalized_merged_spectrum = interpolate_and_merge(wavelengths, fluxes, uncertainties, linear_wavelengths)

#     # Output the final reduced and calibrated data
#     save_final_spectrum(normalized_merged_spectrum, args.night, args.object)

#     # TO BE ADDED WHEN WORKING WITH MULTIPLE OBSERVATIONS!
#     # Coadding of spectra
#     normalised_coadded_spectrum = co_add_spectra(normalized_merged_spectrum, common_wavelength)

#     # Degrade spectrum down to 4MOST HR
#     spectrum_4most_hr = degrade_resolution_with_uncertainty(wavelength, flux, flux_uncertainty, 80000, 22000)

        ########################################
        # Produce some beautiful outreach material
        print('    Producing Rainbow Plot')        
        matrix_color = []
        matrix_wavelength = []
        for ccd in [1,2,3]:
            # We have to go in reverse order, as increasing order means decreasing wavelength
            for order in list(flux_in_orders.keys())[::-1]:
                if order[4] == str(ccd):
                    if not (
                        ((ccd == 1) & (int(order.split('_')[-1]) in [138,139,140,154,155,156,157,158,159,160,161,162,163,164,165,166,167])) |
                        ((ccd == 2) & (int(order.split('_')[-1]) in [103])) |
                        ((ccd == 3) & (int(order.split('_')[-1]) in [65,104]))
                    ):

                        # better signal and less overlap
                        pixel_cutoff = 500

                        # color array of 
                        wavelength_array = np.linspace(vbary_corr_wavelengths_per_order[order][0],vbary_corr_wavelengths_per_order[order][-1],4094 - 2*pixel_cutoff)
                        colorrange_array = np.linspace(0,len(flux_in_orders[order][pixel_cutoff:-pixel_cutoff]),4094 - 2*pixel_cutoff)

                        flux_interpolate = interpolate_spectrum(
                            np.arange(len(flux_in_orders[order][pixel_cutoff:-pixel_cutoff])),
                            flux_in_orders[order][pixel_cutoff:-pixel_cutoff],
                            colorrange_array
                        )
                        for i in range(10):
                            matrix_color.append(flux_interpolate)
                            matrix_wavelength.append(wavelength_array)

        plt.figure(figsize=(10,10))
        shape = np.shape(matrix_color)
        aspect = shape[1]/shape[0]
        s = plt.imshow(matrix_wavelength, aspect = aspect, vmin = 3750, vmax = 7800, cmap = create_rainbow_colormap())
        s = plt.imshow(matrix_color, aspect = aspect, vmin = 0.5, vmax = 1, cmap = create_transparent_greyscale_colormap())
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        Path(config.working_directory+'reduced_data/'+config.date+'/diagnostic_plots/'+str(run)).mkdir(parents=True, exist_ok=True)
        plt.savefig(config.working_directory+'reduced_data/'+config.date+'/diagnostic_plots/'+str(run)+'/'+config.date+'_'+str(run)+'_'+object_name+'_rainbow.pdf',dpi=200,bbox_inches='tight')
        plt.savefig(config.working_directory+'reduced_data/'+config.date+'/diagnostic_plots/'+str(run)+'/'+config.date+'_'+str(run)+'_'+object_name+'_rainbow.png',dpi=200,bbox_inches='tight')
        plt.close()

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
        parser.add_argument('-teff', default=None, type=float, help='Effective temperature to calculate Korg spectra')
        parser.add_argument('-logg', default=None, type=float, help='Surface gravity to calculate Korg spectra')
        parser.add_argument('-fe_h', default=None, type=float, help='Iron abundance to calculate Korg spectra')
        parser.add_argument('--debug', action='store_true')
        args = parser.parse_args()
        if len(args.date) == 6:
            config.date = args.date
        else:
            raise ValueError('date must be a 6 digit string in the format YYMMDD')
        config.object_name = args.object_name
        config.working_directory = args.working_directory
        config.debug = args.debug
        
        if args.teff is not None:
            try:
                config.teff = float(args.teff)
            except:
                raise ValueError('teff argument must be float')
            try:
                config.logg = float(args.logg)
            except:
                raise ValueError('logg argument must be float')
            try:
                config.fe_h = float(args.fe_h)
            except:
                raise ValueError('fe_h argument must be float')
        try:
            from juliacall import Main as jl
            config.use_korg = True
        except:
            print('Could not import juliacall. Will not use Korg to normalise spectra.')
            config.use_korg = False
    except:
        config.teff = None
        config.logg = None
        config.fe_h = None
        pass
    
    if config.debug:
        import matplotlib.pyplot as plt
    
    # Extract relevant information from night log
    calibration_runs, science_runs = identify_calibration_and_science_runs(config.date, config.working_directory+'raw_data/')
    
    # Overwrite GBS stars that were also observed as part of the Halo program
    if 'Halo11' in science_runs.keys():
        science_runs['HIP76976'] = science_runs['Halo11']
        if config.object_name == 'Halo11':
            print('\nRewriting GBS object_name to HIP identifier (rather than HaloX)')
            config.object_name = 'HIP76976'
    
    # Create Master Flat
    master_flats = dict()
    for ccd in [1,2,3]:
        master_flats['ccd_'+str(ccd)] = []
        if ccd == 1: flat_runs = calibration_runs['Flat_60.0'][:1]
        if ccd == 2: flat_runs = calibration_runs['Flat_1.0'][:1]
        if ccd == 3: flat_runs = calibration_runs['Flat_0.1'][:1]
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

    # if debug: save master flat
    if config.debug:
        f, gs = plt.subplots(1,3,figsize=(12,4))
        for ccd in [1,2,3]:
            plot_ccd_imshow(ax = gs[ccd-1], image = master_flats['ccd_'+str(ccd)], panel_title = 'Master Flat CCD'+str(ccd))
        plt.tight_layout()
        Path(config.working_directory+'reduced_data/'+config.date+'/diagnostic_plots/').mkdir(parents=True, exist_ok=True)
        plt.savefig(config.working_directory+'reduced_data/'+config.date+'/diagnostic_plots/'+config.date+'_master_flat.pdf',dpi=200,bbox_inches='tight')
        plt.close()

    # Process all frames of a given object_name
    process_objects(config.date, config.object_name, master_flats, science_runs)

    print(f"\nReduction and calibration succesfull and complete.")


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




