import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import quad
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import config

from VeloceReduction.utils import polynomial_function, calculate_barycentric_velocity_correction, velocity_shift, fit_voigt_absorption_profile, radial_velocity_from_line_shift, voigt_absorption_profile, wavelength_vac_to_air, wavelength_air_to_vac, lc_peak_gauss, lasercomb_numbers_from_wavelength, lasercomb_wavelength_from_numbers, read_in_wavelength_solution_coefficients_tinney

def optimise_wavelength_solution_with_laser_comb(order_name, lc_pixel_values, overwrite=False, debug=False, use_ylim=False):
    """
    Optimises the wavelength solution for a given spectral order using the laser comb data.
    This function first identifies rough peaks with scipy.signal.find_peaks and then fits a Gaussian to these peaks.
    It then fits a polynomial (and in the case of 3-sigma outliers repeats the fit) of peak pixel to peak wavelength.

    Parameters:
        order_name (str):                   The name of the spectral order to optimise the wavelength solution for.
        lc_pixel_values (numpy.ndarray):    The pixel values of the laser comb data for the given spectral order.
        overwrite (bool, optional):         If True, the function will overwrite the existing wavelength solution coefficients.
        debug (bool, optional):             If True, the function will generate diagnostic plots showing the wavelength solution.
        use_ylim (bool, optional):          If True, the function will set the y-axis limits for the diagnostic plots.

    Returns:
        coeffs_lc (numpy.ndarray):          The optimised polynomial wavelength solution coefficients for the given spectral order.
    """

    # We are only fitting peaks and wavelength solutions within a certain pixel range.
    # Here, we define these pixel ranges (0-4128).
    lc_range = dict()
    lc_range['ccd_3_order_104'] = [1455,3900]
    lc_range['ccd_3_order_103'] = [780,3900]
    lc_range['ccd_3_order_102'] = [700,3900]
    lc_range['ccd_3_order_101'] = [650,3920]
    lc_range['ccd_3_order_100'] = [600,3820]
    lc_range['ccd_3_order_99'] = [490,3800]
    lc_range['ccd_3_order_98'] = [500,3750]
    lc_range['ccd_3_order_97'] = [325,3775]
    lc_range['ccd_3_order_96'] = [255,3720]
    lc_range['ccd_3_order_95'] = [250,3650]
    lc_range['ccd_3_order_94'] = [450,3650]
    lc_range['ccd_3_order_93'] = [270,3600]
    lc_range['ccd_3_order_92'] = [540,2800]
    lc_range['ccd_3_order_91'] = [630,3525]
    lc_range['ccd_3_order_90'] = [490,3525]
    lc_range['ccd_3_order_89'] = [355,3465]
    lc_range['ccd_3_order_88'] = [400,2900]
    lc_range['ccd_3_order_87'] = [505,3245]
    lc_range['ccd_3_order_86'] = [200,3200]
    lc_range['ccd_3_order_85'] = [455,3655]
    lc_range['ccd_3_order_84'] = [150,3578]
    lc_range['ccd_3_order_83'] = [145,3850]
    lc_range['ccd_3_order_82'] = [120,3870]
    lc_range['ccd_3_order_81'] = [125,4050]
    lc_range['ccd_3_order_80'] = [130,4000]
    lc_range['ccd_3_order_79'] = [120,4070]
    lc_range['ccd_3_order_78'] = [100,4100]
    lc_range['ccd_3_order_77'] = [100,4100]
    lc_range['ccd_3_order_76'] = [105,4095]
    lc_range['ccd_3_order_75'] = [99,4050]
    lc_range['ccd_3_order_74'] = [745,4050]
    lc_range['ccd_3_order_73'] = [99,4100]
    lc_range['ccd_3_order_72'] = [125,4090]
    lc_range['ccd_3_order_71'] = [95,4086]
    lc_range['ccd_3_order_70'] = [110,4077]
    lc_range['ccd_3_order_69'] = [90,4087]
    lc_range['ccd_3_order_68'] = [90,4090]
    lc_range['ccd_3_order_67'] = [100,4090]
    lc_range['ccd_3_order_66'] = [100,3730]
    lc_range['ccd_2_order_134'] = [1590,3850]
    lc_range['ccd_2_order_133'] = [1398,3850]
    lc_range['ccd_2_order_132'] = [1420,3850]
    lc_range['ccd_2_order_131'] = [1420,3850]
    lc_range['ccd_2_order_130'] = [1153,3900]
    lc_range['ccd_2_order_129'] = [1025,3960]
    lc_range['ccd_2_order_128'] = [825,3960]
    lc_range['ccd_2_order_127'] = [825,3965]
    lc_range['ccd_2_order_126'] = [905,3970]
    lc_range['ccd_2_order_125'] = [895,3970]
    lc_range['ccd_2_order_124'] = [795,3970]
    lc_range['ccd_2_order_123'] = [885,3870]
    lc_range['ccd_2_order_122'] = [785,3870]
    lc_range['ccd_2_order_121'] = [785,3870]
    lc_range['ccd_2_order_120'] = [785,3870]
    lc_range['ccd_2_order_119'] = [405,3870]
    lc_range['ccd_2_order_118'] = [355,3670]
    lc_range['ccd_2_order_117'] = [855,3850]
    lc_range['ccd_2_order_116'] = [480,3850]
    lc_range['ccd_2_order_115'] = [405,3790]
    lc_range['ccd_2_order_114'] = [475,3700]
    lc_range['ccd_2_order_113'] = [845,3700]
    lc_range['ccd_2_order_112'] = [445,3700]
    lc_range['ccd_2_order_111'] = [170,3830]
    lc_range['ccd_2_order_110'] = [150,3980]
    lc_range['ccd_2_order_109'] = [430,3730]
    lc_range['ccd_2_order_108'] = [320,3500]
    lc_range['ccd_2_order_107'] = [215,3580]
    lc_range['ccd_2_order_106'] = [215,3455]
    lc_range['ccd_2_order_105'] = [150,3680]
    lc_range['ccd_2_order_104'] = [200,3200]
    lc_range['ccd_2_order_103'] = [140,3200]

    # Check if the order is in the range of orders we can optimise
    if not ((order_name[4] != '1') & (order_name != 'ccd_3_order_65')):
        raise ValueError('This function is only implemented for CCD3 (exepct order 65) and CCD2 orders 103-134')
    else:

        # Use wavelength coefficients according to the following preference:
        # 1) Coefficients fitted with 18Sco and Korg synthesis
        # 2) Coefficients fitted with LC
        # 3) Coefficients fitted with ThXe
        try:
            previous_calibration_coefficients = np.loadtxt('./VeloceReduction/wavelength_coefficients/wavelength_coefficients_'+order_name+'_korg.txt')
        except:
            try:
                previous_calibration_coefficients = np.loadtxt('./VeloceReduction/wavelength_coefficients/wavelength_coefficients_'+order_name+'_lc.txt')
            except:
                previous_calibration_coefficients = np.loadtxt('./VeloceReduction/wavelength_coefficients/wavelength_coefficients_'+order_name+'_thxe.txt')

        wavelength = polynomial_function(np.arange(4128)-2064,*previous_calibration_coefficients)*10

        # Identify the range for which we will fit the peaks
        if order_name in lc_range.keys():
            lc_beginning, lc_ending = lc_range[order_name]
        else:
            lc_beginning = 500
            lc_ending = 3700
        in_panel = np.arange(lc_beginning,lc_ending+1)
        close_to_in_panel = np.arange(lc_beginning-100,np.min([lc_ending+100,4127]))

        # Adjust the peak distance, acknowledging that the distance differs based on wavelength
        # and order. We use the following separations:
        # 6-8 pxiels for the CCD3
        # 5-6 pixels for the red part of CCD2
        # 4-6 pixels for bluest part of CCD2
        if order_name[4] == '3':
            peak_distance1 = 6
            peak_distance2 = 8
        elif int(order_name[-3]) > 118:
            peak_distance1 = 4
            peak_distance2 = 6
        else:
            peak_distance1 = 5
            peak_distance2 = 6
        # Use the midpoint for a better application of find_peaks (since it can only take 1 integer as distance)
        lc_value_in_panel_midpoint = len(lc_pixel_values[in_panel]) // 2
        lc_values_half1 = lc_pixel_values[in_panel][:lc_value_in_panel_midpoint]  # First half
        lc_values_half2 = lc_pixel_values[in_panel][lc_value_in_panel_midpoint:]

        # Adjust the expected peak height and dominance - this is not yet robust, and only established for 1 LC exposure...
        if order_name in ['ccd_3_order_80','ccd_3_order_81','ccd_3_order_90','ccd_3_order_92','ccd_3_order_88','ccd_2_order_126']:
            peak_height = 10
            peak_prominence = 10
        elif order_name in [
            'ccd_3_order_86',
            'ccd_2_order_104','ccd_2_order_105',
            'ccd_2_order_112','ccd_2_order_113','ccd_2_order_114','ccd_2_order_115','ccd_2_order_116'
        ]:
            peak_height = 5
            peak_prominence = 5
        elif order_name in ['ccd_3_order_87','ccd_3_order_91']:
            peak_height = 1
            peak_prominence = 3

        elif order_name in ['ccd_2_order_106','ccd_2_order_107','ccd_2_order_108','ccd_2_order_109','ccd_2_order_117']:
            peak_height = 2
            peak_prominence = 2
        else:
            peak_height = 20
            peak_prominence = 20

        # Fit the peaks for the left and right half of the order and concatenate them to "peaks"
        peaks1, peak_metadata1 = find_peaks(
            lc_values_half1,
            height = peak_height,
            prominence = peak_prominence,
            distance = peak_distance1
        )
        peaks2, peak_metadat2 = find_peaks(
            lc_values_half2,
            height = peak_height,
            prominence = peak_prominence,
            distance = peak_distance2
        )
        peaks = np.concatenate((peaks1,peaks2+lc_value_in_panel_midpoint))

        # Identify gaps (>1.5 median_peak_distance) that are not within the first 50 and last 600 pixels and fill these gaps
        max_right_buffer = 600
        if order_name == 'ccd_2_order_111':
            max_right_buffer = 50
        median_peak_distance = np.median(np.diff(peaks))
        
        position_of_too_large_gap_between_peaks = np.where((np.diff(peaks) > 1.5*median_peak_distance) & (peaks[:-1] > 20) & (peaks[:-1] < lc_ending - max_right_buffer))[0]
        
        if len(position_of_too_large_gap_between_peaks) > 0:

            # Initialize the new peaks list from existing peaks
            new_peaks = list(peaks)
            new_peaks_added = []

            # Insert new peaks in the positions of the large gaps and overwrite peaks
            for index in reversed(position_of_too_large_gap_between_peaks):
                start_peak = peaks[index]
                end_peak = peaks[index + 1]
                try:
                    neighbour_distance = abs(peaks[index + 2] - end_peak)
                except:
                    neighbour_distance = abs(peaks[index - 1] - start_peak)

                # Enforce additional robustness of significant enough gap at specific location
                # Distance at gap is better than median distance, since the pixel distance increases across the order.
                if end_peak - start_peak > 1.5*neighbour_distance:
                    new_peak_position = (start_peak + end_peak) // 2
                    new_peaks.insert(index + 1, new_peak_position)
                    new_peaks_added.append(new_peak_position)
            peaks = new_peaks

            if debug:
                if len(new_peaks_added) > 0:
                    print('Found '+str(len(new_peaks_added))+' gaps: ', new_peaks_added)
                else:
                    print('Found 0 gaps')

        # Plot the laser rough comb peaks if we want to debug
        if debug:
            f, ax = plt.subplots(1,1,figsize=(15,5))
            ax.set_title(order_name)
            ax.plot(
                wavelength[close_to_in_panel],
                lc_pixel_values[close_to_in_panel],
                lw = 0.5
            )
            ax.set_ylim(0,1.1*np.percentile(lc_pixel_values[np.isfinite(lc_pixel_values)],q=99))

            for peak in peaks:
                ax.axvline(wavelength[in_panel][peak], c = 'C3', lw=0.5, ls='dashed')
            plt.tight_layout()
            plt.show()
            plt.close()

        # Now that we have the integer peak positions, let's fit more precise Gaussians.
        # Use the rough integer peaks, if the Gaussian fit fails (likely for weak peaks)
        fine_peaks = []
        for peak in peaks:

            # Find the pixels that are +- 0.5*peak_distance away from the peak
            pixels_around_peak = np.arange(
                np.max([0,peak - int(np.ceil(peak_distance1/2))]),
                np.min([len(lc_pixel_values[in_panel]),peak + int(np.ceil(peak_distance1/2))+1])
            )
            pixel_values_around_peak = lc_pixel_values[in_panel][pixels_around_peak]
            pixel_minmax = list(np.nanpercentile(pixel_values_around_peak,q=[1,99]))

            try:
                popt, pcov = curve_fit(
                    lc_peak_gauss,
                    pixels_around_peak,
                    pixel_values_around_peak,
                    p0 = [peak, 1, pixel_minmax[1]-pixel_minmax[0], pixel_minmax[0]]
                )

                # Make sure the Gaussian is not too far off!
                # Use initial peak integer otherwise
                if abs(peak - popt[0] > 1):
                    fine_peaks.append(peak)
                else:
                    fine_peaks.append(popt[0])
            except:
                if debug:
                    print('Failed fit for peak '+str(peak)+' at position '+str(peak+lc_beginning))
                fine_peaks.append(peak)
        fine_peaks = np.array(fine_peaks)

            # #Plot the Gaussian fits for each peak if we want to debug
            # if debug:
            #     f, ax = plt.subplots(1,1)
            #     ax.scatter(
            #         pixels_around_peak,
            #         pixel_values_around_peak,
            #         s = 20
            #     )
            #     ax.plot(
            #         np.linspace(pixels_around_peak[0],pixels_around_peak[-1],50),
            #         lc_peak_gauss(np.linspace(pixels_around_peak[0],pixels_around_peak[-1],50), *popt),
            #         c = 'C1'
            #     )
            #     plt.tight_layout()
            #     plt.show()
            #     plt.close()

        # Now that we have the fine peaks, let's fit a polynomial to the pixel and wavelength data
        # For this, we first have to determine the laser comb numbers and wavelengths
        lc_number_upper = np.floor(lasercomb_numbers_from_wavelength(wavelength[in_panel][0]))
        lc_number_lower = np.ceil(lasercomb_numbers_from_wavelength(wavelength[in_panel][-1]))
        lc_wavelengths = lasercomb_wavelength_from_numbers(np.arange(lc_number_lower, lc_number_upper+1))[::-1]

        # In some cases, the number of peaks and modes differ. In this case, we only use the first n peaks and modes.
        if debug:
            print('peaks: ',len(peaks))
            print('modes: ',len(lc_wavelengths))
        if len(peaks) != len(lc_wavelengths):
            use_peaks_and_modes = np.min([len(peaks),len(lc_wavelengths)])
            if debug:
                print('Only using first '+str(use_peaks_and_modes)+' entries')
        else:
            use_peaks_and_modes = len(peaks)

        # Fit a polynomial function to pixel and wavelength data
        lc_pixels_to_fit = lc_beginning + fine_peaks[:use_peaks_and_modes] - 2064
        lc_wavelengths_to_fit = lc_wavelengths[:use_peaks_and_modes]
        coeffs_lc, _ = curve_fit(
            polynomial_function,
            lc_pixels_to_fit,
            lc_wavelengths_to_fit/10.,
            p0=[np.median(lc_wavelengths_to_fit), 0.05, 0.0, 0.0, 0.0, 0.0]
        )

        # Calculate the RMS wavelength and velocity
        wavelength_residuals = (lc_wavelengths_to_fit - (polynomial_function(lc_pixels_to_fit,*coeffs_lc)*10)) # Aangstroem
        rms_wavelength = np.std(wavelength_residuals)
        rms_velocity = 299792.46 * np.std(wavelength_residuals/(lc_wavelengths_to_fit))

        if debug:
            plt.figure(figsize=(15,5))
            plt.title(order_name,fontsize=15)

        # Calculate X-sigma RMS velocity outliers, clip them, and refit the wavelength solution
        rms_sigma = 3
        rms_velocity_x_sigma_outlier = np.where(299792.46 * np.abs(wavelength_residuals/(lc_wavelengths_to_fit)) / rms_velocity > rms_sigma)[0]        
        if len(rms_velocity_x_sigma_outlier) > 0:

            if debug:
                print('Refitting wavelength solution after clipping '+str(len(rms_velocity_x_sigma_outlier))+' '+str(rms_sigma)+'-sigma RMS velocity outliers: ',np.round(lc_pixels_to_fit[rms_velocity_x_sigma_outlier]+2064))
                plt.scatter(
                    lc_pixels_to_fit[rms_velocity_x_sigma_outlier]+2064,
                    lc_wavelengths_to_fit[rms_velocity_x_sigma_outlier] - (polynomial_function(lc_pixels_to_fit[rms_velocity_x_sigma_outlier],*coeffs_lc)*10),
                    s = 5, c = 'C3',
                    label = str(len(rms_velocity_x_sigma_outlier))+' '+str(rms_sigma)+'-sigma RMS velocity outlier(s)'
                )
            lc_pixels_to_fit = np.delete(lc_pixels_to_fit, rms_velocity_x_sigma_outlier)
            lc_wavelengths_to_fit = np.delete(lc_wavelengths_to_fit, rms_velocity_x_sigma_outlier)
            coeffs_lc, _ = curve_fit(
                polynomial_function,
                lc_pixels_to_fit,
                lc_wavelengths_to_fit/10.,
                p0=[np.median(lc_wavelengths_to_fit), 0.05, 0.0, 0.0, 0.0, 0.0]
            )
            wavelength_residuals = lc_wavelengths_to_fit - (polynomial_function(lc_pixels_to_fit,*coeffs_lc)*10) # Aangstroem
            rms_wavelength = np.std(wavelength_residuals)
            rms_velocity = 299792.46 * np.std(wavelength_residuals/lc_wavelengths_to_fit)

        if overwrite:
            np.savetxt('./VeloceReduction/wavelength_coefficients/wavelength_coefficients_'+order_name+'_lc.txt',coeffs_lc)

        # Plot the difference between the LC wavelength solution and
        #   a) the LC peaks (as scatter points) as dots,
        #   b) the Tinney wavelength solution, and
        #   c) the ThXe wavelength solution.
        # Also inform the user about the wavelength and RV RMS.
        if debug:

            # LC Wavelength Solution
            plt.scatter(
                lc_pixels_to_fit+2064,
                (lc_wavelengths_to_fit - (polynomial_function(lc_pixels_to_fit,*coeffs_lc))*10),
                s = 1,
                label = 'LC Peaks, RMS = '+str(np.round(rms_wavelength,4))+' Å or '+str(np.round(rms_velocity,3))+' km/s'
            )
            plt.plot(
                np.arange(4128),
                np.zoers(4128),
                label = 'LC Wavelength Solution'
            )

            # Tinney Wavelength Solution
            coeffs_tinney = read_in_wavelength_solution_coefficients_tinney()
            wavelength_tinney = polynomial_function(np.arange(4128)-2450-3,*coeffs_tinney[order_name][:-1])*10
            if order_name[4] == '2':
                wavelength_tinney = wavelength_air_to_vac(wavelength_tinney)
            plt.plot(
                np.arange(4128),
                wavelength_tinney -
                polynomial_function(np.arange(4128)-2064,*coeffs_lc)*10,
                label = 'Tinney Wavelength Solution'
            )

            # ThXe Wavelegnth Solution
            coeffs_thxe = np.loadtxt('./VeloceReduction/wavelength_coefficients/wavelength_coefficients_'+order_name+'_thxe.txt')
            plt.plot(
                np.arange(4128),
                polynomial_function(np.arange(4128)-2064,*coeffs_thxe)*10 - 
                polynomial_function(np.arange(4128)-2064,*coeffs_lc)*10,
                label = 'ThXe Wavelength Solution'
            )

            if use_ylim:
                plt.ylim(-0.5,0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()

        return(coeffs_lc)

def calibrate_single_order(file, order, barycentric_velocity=None, optimise_lc_solution=True):
    """
    Calibrates a single spectral order by fitting a polynomial to the pixel-to-wavelength relationship 
    and optionally applying a barycentric velocity correction. The calibration updates the wavelength 
    data for both vacuum and air wavelengths directly within the provided FITS file object.

    Parameters:
        file (HDUList): An open FITS file object from astropy.io.fits, which should contain the spectral data.
                        The spectral data of each order should be accessible via indexing.
        order (int): The index of the order within the FITS file to calibrate. This specifies which HDU in the
                     FITS file is being calibrated.
        barycentric_velocity (float, optional): The barycentric radial velocity (in km/s) to correct for
                                                the Doppler shift due to Earth's motion. If provided, this velocity
                                                is used to adjust the calculated wavelengths for the motion of the Earth.
                                                If None, no correction is applied.
        optimise_lc_solution (bool, optional):  If True, the function will use the laser comb data of the FITS extension
                                                and optimise the wavelength solution based on refitted peaks of the laser comb.

    Returns:
        None: The function modifies the 'file' object in-place, updating the wavelength data for the specified order.

    This function processes the spectral data by:
    - Determining the center pixel based on the number of pixels in the order, which varies by readout mode.
    - Loading a predefined pixel-to-wavelength calibration from a text file.
    - Fitting a polynomial to these data points to create a wavelength solution in vacuum.
    - Converting this vacuum wavelength solution to air wavelength using a standard conversion formula.
    - Optionally correcting for barycentric velocity if a value is provided.

    Note:
    - The polynomial fit and vacuum-to-air wavelength conversion rely on specific coefficients and formulae
      that are expected to be defined or included via external resources or modules.

    Example:
        >>> from astropy.io import fits
        >>> fits_file = fits.open('path_to_fits_file.fits')
        >>> calibrate_single_order(fits_file, 0, barycentric_velocity=30.5)
        >>> fits_file.close()  # Always close the FITS file after modification to ensure data integrity
    """

    order_name = file[order].header['EXTNAME'].lower()

    # Because of the different number of pixels for the 2Amp and 4Amp readout, we have to adjust the centre pixel
    # This is usually 2048 for 4Amp readout and 2064 for 2Amp readout.
    order_centre_pixel = int(len(file[order].data['WAVE_AIR'])/2)
    
    # Use the initial pixel <-> wavelength information per order to fit a polynomial function to it.

    # Use wavelength coefficients according to the following preference:
    # 1) Coefficients fitted with 18Sco (RV = 11.7640 +- 0.0004 km/s) and Korg synthesis
    # 2) Coefficients fitted with LC
    # 3) Coefficients fitted with ThXe
    try:
        wavelength_solution_vacuum_coefficients = np.loadtxt('./VeloceReduction/wavelength_coefficients/wavelength_coefficients_'+order_name+'_korg.txt')
    except:
        try:
            wavelength_solution_vacuum_coefficients = np.loadtxt('./VeloceReduction/wavelength_coefficients/wavelength_coefficients_'+order_name+'_lc.txt')
        except:
            wavelength_solution_vacuum_coefficients = np.loadtxt('./VeloceReduction/wavelength_coefficients/wavelength_coefficients_'+order_name+'_thxe.txt')
    
    # Optimise the LC solution based on the refitted peaks of the laser comb, if enabled
    if optimise_lc_solution:
        if (
            ((order_name[4] == '3') & (order_name != 'ccd_3_order_65')) |
            # Let's use the last 2 digits to avoid warnings, because not all of CCD3 are > 100 (but all of CCD2).
            ((order_name[4] == '2') & (int(order_name[-2:]) >= 3) & (int(order_name[-2:]) <= 34))
        ):
            wavelength_solution_vacuum_coefficients = optimise_wavelength_solution_with_laser_comb(order_name, lc_pixel_values = file[order].data['LC'])

    # Calculate vacuum wavelengths and convert them to air wavelengths
    wavelength_solution_vacuum = polynomial_function(
        np.arange(len(file[order].data['WAVE_VAC'])) - order_centre_pixel,
        *wavelength_solution_vacuum_coefficients
    ) * 10  # Convert from nm to Å
    file[order].data['WAVE_VAC'] = wavelength_solution_vacuum

    if barycentric_velocity is not None:
        file[order].data['WAVE_VAC'] = velocity_shift(barycentric_velocity, file[order].data['WAVE_VAC'])

    # Using conversion from Birch, K. P., & Downs, M. J. 1994, Metro, 31, 315
    # Consistent to the 2024 version of Korg (https://github.com/ajwheeler/Korg.jl)
    file[order].data['WAVE_AIR'] = wavelength_vac_to_air(file[order].data['WAVE_VAC'])

def plot_wavelength_calibrated_order_data(order, science_object, file, overview_pdf):
    """
    Generates a five-panel plot for a single calibrated spectral order and saves it to the provided PDF. The panels include:
    1. Science spectrum.
    2. Science signal-to-noise ratio.
    3. Flat-field data.
    4. ThXe emission lines used for wavelength calibration.
    5. Laser Comb (LC) data used for wavelength calibration.

    Parameters:
        order (int): The index of the spectral order to plot.
        science_object (str): The name of the science object associated with the data.
        file (HDUList): The FITS file object containing the spectral data and associated headers.
        overview_pdf (PdfPages): A PdfPages object where the generated plot will be saved as a new page.

    Returns:
        None: The function does not return anything but saves the generated plot as a new page in the provided PDF.

    This function performs the following:
    - Reads the air wavelength data and optionally applies a radial velocity correction based on the FITS header.
    - Creates a five-panel plot with shared x-axes showing:
        - Science spectrum with pixel and wavelength labels.
        - Signal-to-noise ratio of the science spectrum.
        - Flat-field spectrum.
        - ThXe calibration lamp spectrum (logarithmic scale).
        - Laser Comb calibration spectrum (logarithmic scale).
    - Saves the plot to the specified PDF and closes the plot to free resources.

    Example:
        >>> from matplotlib.backends.backend_pdf import PdfPages
        >>> from astropy.io import fits
        >>> fits_file = fits.open('path_to_fits_file.fits')
        >>> with PdfPages('overview.pdf') as pdf:
        ...     plot_wavelength_calibrated_order_data(0, 'HD12345', fits_file, pdf)
        >>> fits_file.close()
    """

    # Extract wavelength data to plot
    wavelength_to_plot = file[order].data['WAVE_AIR']

    # Create the figure and set up shared x-axis
    f, gs = plt.subplots(5,1,figsize=(15,10),sharex=True)
    
    # Plot title with radial and barycentric velocity information
    if isinstance(file[0].header['VRAD'], float):
        wavelength_to_plot = velocity_shift(velocity_in_kms=file[0].header['VRAD'], wavelength_array=wavelength_to_plot)
        f.suptitle(config.date+' '+science_object+' '+file[order].header['EXTNAME']+ r' $v_\mathrm{rad} = '+str(file[0].header['VRAD'])+r' \pm '+str(file[0].header['E_VRAD'])+'\,\mathrm{km\,s^{-1}}$, $v_\mathrm{bary} = '+"{:.2f}".format(np.round(file[0].header['BARYVEL'],2))+'\,\mathrm{km\,s^{-1}}$')
    else:
        f.suptitle(config.date+' '+science_object+' '+file[order].header['EXTNAME']+ r' $v_\mathrm{rad} = N/A, $v_\mathrm{bary} = '+"{:.2f}".format(np.round(file[0].header['BARYVEL'],2))+'\,\mathrm{km\,s^{-1}}$')

    # Panel 1: Science spectrum
    ax = gs[0]
    ax.plot(file[order].data['SCIENCE'], lw=1)
    ax.set_ylabel('Science')
    ax.set_ylim(-0.1, 1.2)
    ticks = np.arange(0, len(wavelength_to_plot), 100)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=90)
    ax.set_xlim(ticks[0], ticks[-1])
    ax2 = ax.twiny()
    ax2.set_xticks(ticks - ticks[0])
    ax2.set_xticklabels(np.round(wavelength_to_plot[ticks], 1), rotation=90)
    ax2.set_xlabel(r'Air Wavelength $\lambda_\mathrm{air}~/~\mathrm{\AA}$')

    # Panel 2: Signal-to-noise ratio
    ax = gs[1]
    ax.plot(file[order].data['SCIENCE'] / file[order].data['SCIENCE_NOISE'], lw=1)
    ax.set_ylabel('Science S/N')

    # Panel 3: Flat-field data
    ax = gs[2]
    ax.plot(file[order].data['FLAT'], lw=1)
    ax.set_ylabel('Flat')

    # Panel 4: ThXe emission lines
    ax = gs[3]
    ax.plot(file[order].data['THXE'], lw=1)
    ax.set_yscale('log')
    ax.set_ylabel('ThXe')

    # Panel 5: Laser Comb data
    ax = gs[4]
    ax.plot(file[order].data['LC'], lw=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('LC')
    ax.set_xlabel('Pixel')
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=90)

    # Save the plot to the provided PDF
    overview_pdf.savefig()
    plt.close()

def calibrate_wavelength(science_object, optimise_lc_solution=True, correct_barycentric_velocity=True, fit_voigt_for_rv=True, create_overview_pdf=False):
    """
    Calibrates the wavelength data for a given science object by applying a series of corrections and enhancements.
    This includes barycentric velocity correction, radial velocity estimation via Voigt profile fitting, and
    optionally, generating an overview PDF with plots of the calibrated data.

    Parameters:
        science_object (str): The identifier for the science object. This is used to locate the corresponding
                              FITS file and to label outputs appropriately.
        optimise_lc_solution (bool): If True, optimise the wavelength solution based on the refitted peaks of the laser comb.
        correct_barycentric_velocity (bool): If True, apply a correction for the barycentric velocity to the
                                             wavelength data based on header information.
        fit_voigt_for_rv (bool): If True, fits a Voigt profile to selected spectral lines (Halpha and CaII triplet)
                                 to estimate the radial velocity of the object.
        create_overview_pdf (bool): If True, generates a PDF file containing plots of the calibrated spectral data
                                    for each order in the FITS file.

    Returns:
        None: This function modifies the FITS files in-place and may generate output files (PDFs), but does not
              return any values.

    Workflow:
        - The function first identifies the appropriate directory and FITS file based on the provided science_object name.
        - If barycentric velocity correction is enabled, it calculates and applies this correction.
        - If Voigt profile fitting is enabled, it estimates the radial velocity for the object using selected spectral lines
          and updates the FITS header with these values.
        - Optionally, it can generate a comprehensive PDF overview of the calibrated spectral data.
        - The function handles updates to the FITS file in-place and ensures all changes are saved.

    Note:
        The function assumes that the directory structure and naming conventions are consistent with a predefined
        configuration, which should be verified in the actual implementation environment.

    Example:
        >>> calibrate_wavelength('HD12345', correct_barycentric_velocity=True, fit_voigt_for_rv=True, create_overview_pdf=True)
    """

    print('Calibrating wavelength for ' + science_object)

    # Directory where the reduced data for this science_object can be found
    input_output_directory = config.working_directory+'reduced_data/'+config.date+'/'+science_object

    # Read in FITS file and prepare it to be updated
    with fits.open(input_output_directory+'/veloce_spectra_'+science_object+'_'+config.date+'.fits', mode='update') as file:
    
        # Let user know if we are correcting for barycentric velocity and creating overview PDF
        if correct_barycentric_velocity:
            # ToDo: Check why the barycentric velocity correction seems to be exactly off in the wrong direction?! For now, we will just flip the sign.
            barycentric_velocity = -calculate_barycentric_velocity_correction(science_header=file[0].header)
            print('  -> Correcting for barycentric velocity: '+"{:.2f}".format(np.round(barycentric_velocity,2))+' km/s')
            file[0].header['BARYVEL'] = barycentric_velocity
        else:
            print('  -> Not correcting for barycentric velocity.')
            barycentric_velocity = None
            file[0].header['BARYVEL'] = 'None'

        if optimise_lc_solution:
            print('  -> Optimising wavelength solution based on Laser Comb data where available (and using previous ThXe otherwise)')
        else:
            print('  -> Using previous LC and ThXe calibrations as wavelength solution')

        # Now loop through the FITS file extensions aka Veloce orders to apply the wavelength calibration
        for order in range(1,len(file)):
            calibrate_single_order(file, order, barycentric_velocity=barycentric_velocity, optimise_lc_solution=optimise_lc_solution)

        # If requested, fit a Voigt profile to the Halpha and CaII triplet lines to estimate the radial velocity
        if fit_voigt_for_rv:
            print('  -> Estimating rough Radial Velocity from Halpha and CaII triplet')

            # Define the lines and orders to fit a Voigt profile to.
            # Only fit a few at the moment, as they deliver the most reliable results of ~3 km/s of literature values for the tests done for 240219.
            lines_air_and_orders_for_rv = [
                 # Note: CaII 8662.1410 has to be the first one, as we will use it as initial RV estimate for the other lines
                [r'CaII triplet 8662.1410',8662.1410,103,'CCD_3_ORDER_70'],
                [r'CaII triplet 8662.1410',8662.1410,102,'CCD_3_ORDER_71'],
                [r'CaII triplet 8542.0910',8542.0910,102,'CCD_3_ORDER_71'],
                [r'CaII triplet 8542.0910',8542.0910,101,'CCD_3_ORDER_72'],
                [r'CaII triplet 8498.0230',8498.0230,101,'CCD_3_ORDER_72'],
                [r'MgI 8806.7570',8806.7570,103,'CCD_3_ORDER_70'],
                [r'NiI 7788.9299',7788.9299,94,'CCD_3_ORDER_79'],
                [r'FeI 7511.0187',7511.0187,91,'CCD_3_ORDER_82'],
                [r'$\mathrm{H_\alpha}$ 6562.7970',6562.7970,80,'CCD_3_ORDER_93'],
                [r'$\mathrm{H_\alpha}$ 6562.7970',6562.7970,79,'CCD_3_ORDER_94'],
                [r'FeI 5324.1787',5324.1787,58,'CCD_3_ORDER_115'],
                [r'FeI 5324.1787',5324.1787,57,'CCD_3_ORDER_116'],
            ]

            f, gs = plt.subplots(2,int(np.ceil(len(lines_air_and_orders_for_rv)/2)),figsize=(15,7),sharey=True)
            gs = gs.flatten()
            gs[0].set_ylabel('Flux')

            rv_estimates = []
            rv_estimates_upper = []
            rv_estimates_lower = []
            rv_from_8662 = None

            for index, (line_name, line_centre, order, order_name) in enumerate(lines_air_and_orders_for_rv):
                ax = gs[index]
                ax.set_title(line_name+r'$\,\mathrm{\AA}$'+'\n'+order_name.replace('_',' '))
                ax.set_xlabel(r'Wavelength $\lambda_\mathrm{air}~/~\mathrm{\AA}$')

                if file[order].header['EXTNAME'] != order_name:
                    print('  -> Warning: '+file[order].header['EXTNAME']+' != '+order_name)

                # Restrice fitting region to +- 600 km/s around line centre
                close_to_line_centre = (
                    (file[order].data['WAVE_AIR'] > line_centre * (1 - 600/299792.458)) &
                    (file[order].data['WAVE_AIR'] < line_centre * (1 + 600/299792.458))
                )
                # Estimate the wavelength of the pixel with the lowest flux value within +- 300 km/s around the line centre
                line_within_300_kms = (
                    (file[order].data['WAVE_AIR'] > line_centre * (1 - 300/299792.458)) &
                    (file[order].data['WAVE_AIR'] < line_centre * (1 + 300/299792.458))
                )
                flux_index_within_300_kms = np.argmin(file[order].data['SCIENCE'][line_within_300_kms])
                wavelength_with_lowest_flux_within_300_kms = file[order].data['WAVE_AIR'][line_within_300_kms][flux_index_within_300_kms]

                # Let's make use of the 8662.1410 line as initial RV estimate for the other lines
                if line_centre == 8662.1410:
                    initial_centre_wavelength = wavelength_with_lowest_flux_within_300_kms
                else:
                    if rv_from_8662 is not None:
                        initial_centre_wavelength = (rv_from_8662 / 299792.4658 + 1.0 )* line_centre
                    else:
                        initial_centre_wavelength = wavelength_with_lowest_flux_within_300_kms

                wavelength_to_fit = file[order].data['WAVE_AIR'][close_to_line_centre]
                flux_to_fit = file[order].data['SCIENCE'][close_to_line_centre]

                # Avoid outlier pixels and renormalise locally
                # Estimate 90th percentile and clip all values above 2*90th percentile
                local_90th_percentile = np.nanpercentile(flux_to_fit,q=90)
                flux_to_fit[flux_to_fit > 2*local_90th_percentile] = local_90th_percentile
                # Then renormalise to 95th percentile
                flux_to_fit /= np.nanpercentile(flux_to_fit,q=95)

                ax.plot(
                    wavelength_to_fit,
                    flux_to_fit,
                    c = 'C0',
                    zorder = 1,
                    label = 'Veloce'
                )

                # Make sure that broad lines can be fitted with larger sigmas and gammas
                if line_centre in [6562.7970, 8498.0230, 8542.0910, 8662.1410]:
                    sigma_gamma_max = 5.0
                # But for narrow lines, allow only small sigmas and gammas
                else:
                    sigma_gamma_max = 2.0

                voigt_profile_parameters, voigt_profile_covariances = fit_voigt_absorption_profile(
                    wavelength_to_fit,
                    flux_to_fit,
                    # initial_guess: [line_centre, offset, amplitude, sigma, gamma]
                    initial_guess = [initial_centre_wavelength, np.median(flux_to_fit), 0.5, 0.5, 0.5],
                    # Let's assume an absolute RV below 500 km/s and otherwise (hopefully) reasonable estimates for the line profile.
                    bounds = (
                        [line_centre * (1-500./299792.), 0.1, 0.05, 0.0, 0.0],
                        [line_centre * (1+500./299792.), 1.2, 1.0, sigma_gamma_max, sigma_gamma_max]
                    )
                )

                if voigt_profile_parameters[-2] == sigma_gamma_max:
                    print('  -> Warning: Voigt profile fit hit upper boundary ('+str(sigma_gamma_max)+') for sigma')
                if voigt_profile_parameters[-1] == sigma_gamma_max:
                    print('  -> Warning: Voigt profile fit hit upper boundary ('+str(sigma_gamma_max)+') for gamma')

                rv_voigt = radial_velocity_from_line_shift(voigt_profile_parameters[0], line_centre)
                e_line_centre = np.sqrt(np.diag(voigt_profile_covariances))[0]
                rv_upper_voigt = radial_velocity_from_line_shift(voigt_profile_parameters[0]+e_line_centre, line_centre)
                rv_lower_voigt = radial_velocity_from_line_shift(voigt_profile_parameters[0]-e_line_centre, line_centre)

                # Let's remember the radial velocity estimate for the 8662.1410 line
                if line_centre == 8662.1410:
                    rv_from_8662 = rv_voigt

                if line_centre != 6562.7970:
                    # Estimate equivalent width of the line
                    x_min = voigt_profile_parameters[0] * (1 - 600./299792.458)
                    x_max = voigt_profile_parameters[0] * (1 + 600./299792.458)
                    line_equivalent_width, _ = quad(lambda x: voigt_profile_parameters[1] - voigt_absorption_profile(x, *voigt_profile_parameters), x_min, x_max)

                    if line_equivalent_width > 0.1:
                        rv_estimates.append(rv_voigt)
                        rv_estimates_upper.append(rv_upper_voigt)
                        rv_estimates_lower.append(rv_lower_voigt)
                    else:
                        print('  -> Neglecting RV estimate for '+line_name+' due to weak line (EW '+str(int(np.round(100*line_equivalent_width)))+' < 100 mÅ).')
                elif order == 79:
                    print('  -> Fitting Halpha, but neglecting for RV estimate.')

                ax.plot(
                    wavelength_to_fit,
                    voigt_absorption_profile(wavelength_to_fit, *voigt_profile_parameters),
                    c = 'C1',
                    label = 'Voigt\n'+str(np.round(rv_voigt,1))+' km/s'
                )

                ax.axvline(line_centre, lw = 1, ls = 'dashed', c = 'C3', zorder = 3)
                ax.legend(fontsize=8,handlelength=1)
                ax.set_ylim(-0.1,1.1)

            rv_estimates = np.array(rv_estimates)
            rv_mean = np.round(np.mean(rv_estimates),2)

            def mad_based_outlier(points, thresh=3.5):
                if len(points.shape) == 1:
                    points = points[:,None]
                median = np.median(points, axis=0)
                diff = np.sum((points - median)**2, axis=-1)
                diff = np.sqrt(diff)
                med_abs_deviation = np.median(diff)

                modified_z_score = 0.6745 * diff / med_abs_deviation

                return modified_z_score > thresh

            # Identify and neglect outliers in the RV estimates (either based on MAD or on a threshold of 50 km/s with respect to RV 8662)
            # But ensure at least 3 estimates remain.
            outliers = mad_based_outlier(rv_estimates)
            outliers[abs(rv_estimates - rv_from_8662) > 50] = True
            if (len(np.where(outliers)[0]) > 0) & (len(rv_estimates) - len(np.where(outliers)[0]) >= 3):
                print('  -> Neglecting '+str(np.sum(outliers))+' RV outlier(s): ',np.round(rv_estimates[outliers],2))
                filtered_rv = rv_estimates[~outliers]
            else:
                print('  -> No RV outlier(s) identified.')
                filtered_rv = rv_estimates

            rv_mean = np.round(np.mean(filtered_rv),2)
            rv_std  = np.round(np.std(filtered_rv),2)
            rv_unc  = np.round(np.median(np.array(rv_estimates_upper)-np.array(rv_estimates_lower)),2)
            print(r'  -> $v_\mathrm{rad}  = '+str(rv_mean)+' \pm '+str(rv_std)+' \pm '+str(rv_unc)+r'\,\mathrm{km\,s^{-1}}$ (mean, scatter, unc.) based on '+str(len(filtered_rv))+' lines.')
            
            file[0].header['VRAD'] = rv_mean
            file[0].header['E_VRAD'] = rv_std

            plt.tight_layout()
            plt.savefig(input_output_directory+'/veloce_spectra_'+science_object+'_'+config.date+'_rough_rv_estimate.pdf',bbox_inches='tight')
            plt.show()
            plt.close()

    # Let's create an overview PDF if requested
    if create_overview_pdf:
        with PdfPages(input_output_directory+'/veloce_spectra_'+science_object+'_'+config.date+'_overview.pdf') as overview_pdf:
            with fits.open(input_output_directory+'/veloce_spectra_'+science_object+'_'+config.date+'.fits') as file:
                print('  -> Creating overview PDF. This may take some time for the '+str(len(file))+' orders.\n')
                for order in range(1,len(file)):
                    plot_wavelength_calibrated_order_data(order, science_object, file, overview_pdf)
    else:
        print('  -> Not creating overview PDF.\n')

def fit_thxe_polynomial_coefficients():
    """
    Fits a polynomial function to the pixel-to-wavelength relationship for the ThXe calibration lamp.

    Parameters:
        None

    Returns:
        None: The function saves the fitted polynomial coefficients to a text file for each order.
    """

    # Create array of orders to loop through
    orders = []
    for ccd in ['1','2','3']:
        if ccd == '1': orders.append(['ccd_1_order_'+str(x) for x in np.arange(167, 138-1, -1)])
        if ccd == '2': orders.append(['ccd_2_order_'+str(x) for x in np.arange(140, 103-1, -1)])
        if ccd == '3': orders.append(['ccd_3_order_'+str(x) for x in np.arange(104,  65-1, -1)])
    orders = np.concatenate((orders))

    for order in orders:
        # Read in ThXe pixel and wavelength data
        thxe_pixels_and_wavelengths = np.array(np.loadtxt('./VeloceReduction/VeloceReduction/veloce_reference_data/thxe_pixels_and_positions/' + order + '_px_wl.txt'))

        # Fit a polynomial function to pixel and wavelength data
        thxe_coefficients, _ = curve_fit(
            polynomial_function,
            thxe_pixels_and_wavelengths[:,0] - 2064,
            thxe_pixels_and_wavelengths[:,1],
            p0=[np.median(thxe_pixels_and_wavelengths[:,1]), 0.05, 0.0, 0.0, 0.0, 0.0]
        )

        # Save the fitted polynomial coefficients to a text file
        np.savetxt('./VeloceReduction/VeloceReduction/wavelength_coefficients/wavelength_coefficients_'+order+'_thxe.txt', thxe_coefficients)