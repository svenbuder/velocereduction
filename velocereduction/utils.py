from . import config

import glob
from pathlib import Path
import subprocess
import platform
import os
import sys

# Numpy package
import numpy as np

# Scipy package
from scipy.special import wofz
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

# scikit-image package
from skimage.registration import phase_cross_correlation

# Matplotlib package
import matplotlib.pyplot as plt

# Astropy package
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
SSO = EarthLocation.of_site('Siding Spring Observatory')

# # Astroquery package
# from astroquery.simbad import Simbad
# # Astroquery just for identifier
# simbad_simple = Simbad()
# # Astroquery for different identifiers, radial velocity, and parallax
# simbad_ids_vrad_plx_query = Simbad()
# simbad_ids_vrad_plx_query.add_votable_fields('ids')
# simbad_ids_vrad_plx_query.add_votable_fields('velocity')
# simbad_ids_vrad_plx_query.add_votable_fields('parallax')
# # Astroquery for Fe/H and magnitudes
# simbad_fe_h_query = Simbad()
# simbad_fe_h_query.add_votable_fields('mesfe_h')
# # Astroquery for B, V, G, R magnitudes
# simbad_magnitudes_query = Simbad()
# simbad_magnitudes_query.add_votable_fields('B')
# simbad_magnitudes_query.add_votable_fields('V')
# simbad_magnitudes_query.add_votable_fields('G')
# simbad_magnitudes_query.add_votable_fields('R')

def phase_correlation_shift(reference_image, moving_image, upsample_factor = 100):
    """
    Calculate the occured shift between two images.
    Uses scikit-image package's skimage.registration.phase_cross_correlation.

    Parameters:
        reference_image:    reference array
        moving_image:       second array with same shape as reference array
        upsample_factor:    100 == 0.01 pixel scaling; factors of 10–200 typical; higher = slower, more precise

    Returns:
        dx:    Occured shift in x-direction
        dy:    Occured shift in y-direction
        error: Error of skimage.registration.phase_cross_correlation
    """
    # reference_image, moving_image: 2D numpy arrays of reference and new image
    shift, error, phasediff = phase_cross_correlation(
        reference_image,
        moving_image,
        upsample_factor=upsample_factor,
        normalization="phase" # robust to intensity scaling
    )
    # shift == shift in y and x required to register moving image with reference_image
    # Use negative to see how much images shifted from reference to moving one.
    dy, dx = -shift

    return(dx, dy, error)

def apply_velocity_shift_to_wavelength_array(velocity_in_kms, wavelength_array):
    """
    Applies a Doppler shift to a wavelength array based on the provided radial velocity.

    Parameters:
        velocity_in_kms (float): The radial velocity in kilometers per second.
        wavelength_array (array): The array of wavelengths to be shifted.
        
    Returns:
        array: A new array containing the shifted wavelengths.
    """
    return(wavelength_array / (1.+velocity_in_kms/299792.458))

def radial_velocity_from_line_shift(line_centre_observed, line_centre_rest):
    """
    Calculate the radial velocity from the observed and rest wavelengths of a spectral line.

    Parameters:
        line_centre_observed (float): The observed central wavelength of the spectral line.
        line_centre_rest (float): The rest central wavelength of the spectral line.

    Returns:
        float: The radial velocity in km/s.
    """
    return((line_centre_observed - line_centre_rest) / line_centre_rest * 299792.458)

def voigt_absorption_profile(wavelength, line_centre, line_offset, line_depth, sigma, gamma):
    """
    Returns the Voigt line shape at wavelengths `wavelength` for an absorption line with a continuum.

    Parameters:
        wavelength : array-like
            Wavelength array over which to compute the Voigt profile.
        line_centre : float
            Central wavelength of the absorption line.
        line_offset : float
            Offset of the absorption line from the central wavelength.
        line_depth : float
            The depth of the absorption relative to the continuum.
        sigma : float
            Gaussian standard deviation.
        gamma : float
            Lorentzian half-width at half-maximum (HWHM).
    """
    z = ((wavelength - line_centre) + 1j*gamma) / (sigma*np.sqrt(2))
    profile = wofz(z).real
    return line_offset - line_depth * profile

def fit_voigt_absorption_profile(wavelength, flux, initial_guess, bounds=None):
    """
    Fits a Voigt absorption profile to a given spectrum. The Voigt profile is a convolution of a Gaussian and a Lorentzian,
    commonly used to model the broadening of spectral lines due to both Doppler and natural (pressure or collisional) effects.

    Parameters:
        wavelength (array):                 The array of wavelength data points across which the spectrum is measured.
        flux (array):                       The array of flux measurements corresponding to each wavelength.
        initial_guess (list):               A list of initial guess parameters for the Voigt profile fit:
                                            [line_centre, line_offset, line_depth, sigma, gamma]
        bounds (tuple or list, optional):   Bounds to be used when calling curve_fit optimization.

    Returns:
        tuple: A tuple containing the fitted parameters of the Voigt profile. These parameters include the position (center wavelength),
               amplitude, Gaussian sigma, and Lorentzian gamma.
    """

    if len(initial_guess) != 5:
        raise ValueError("Initial guess must contain 5 values: [line_centre, line_offset, line_depth, sigma, gamma].")
    if bounds is not None and len(bounds) != 2:
        raise ValueError("Bounds must be a tuple or list of length 2, containing the lower and upper bounds for each of the 5 parameters [line_centre, line_offset, line_depth, sigma, gamma].")

    # Fit a Voigt Profile to the spectrum
    if bounds is not None:
        popt, pcov = curve_fit(voigt_absorption_profile, wavelength, flux, p0=initial_guess, bounds=bounds)
    else:
        popt, pcov = curve_fit(voigt_absorption_profile, wavelength, flux, p0=initial_guess)
    return (popt, pcov)

def lc_peak_gauss(pixels, center, sigma, amplitude, offset):
    """
    Gaussian profile for the laser comb peak.

    Parameters:
        pixels : array-like
            Pixel values at which to compute the Gaussian profile.
        center : float
            Center of the Gaussian profile.
        sigma : float
            Standard deviation of the Gaussian profile.
        amplitude : float
            Amplitude of the Gaussian profile.
        offset : float
            Offset of the Gaussian profile.

    Returns:
        array-like: The Gaussian profile evaluated at the given
        pixel values.
    """
    return offset + amplitude * np.exp(-0.5 * ((pixels - center) / sigma) ** 2)
            
def gaussian_absorption_profile(wavelength, line_centre, line_depth, line_sigma):
    """
    Calculates the Gaussian absorption profile for a given set of spectroscopic data. This function models the
    absorption feature using a Gaussian function, which is characterized by a center, depth, and width (sigma).

    Parameters:
        wavelength (array): Array of wavelength data points at which the absorption profile is evaluated.
        line_centre (float): The central wavelength of the Gaussian absorption feature.
        line_depth (float): The depth of the Gaussian line, representing the maximum amount of light absorbed at the line center.
        line_sigma (float): The standard deviation of the Gaussian distribution, determining the width of the absorption line.

    Returns:
        array: An array representing the Gaussian absorption profile across the input wavelength array. The profile
               is calculated as 1 minus the Gaussian function, representing the absorption from a continuum level of 1.

    This function is particularly useful for simulating or fitting spectral data where absorption features are expected
    to follow a Gaussian distribution due to Doppler broadening or instrumental effects.
    """
    return 1 - line_depth * np.exp(-0.5 * ((wavelength - line_centre) / line_sigma) ** 2)

def fit_gaussian_absorption_profile(wavelength, flux, initial_guess, bounds=None):
    """
    Fits a Gaussian absorption profile to given spectroscopic data. This function employs a curve fitting
    method to optimize the parameters of a Gaussian model to best match the observed data, useful in spectral analysis
    where features exhibit Gaussian-shaped absorption due to Doppler broadening or other effects.

    Parameters:
        wavelength (array):                 The array of wavelength data points across which the spectrum is measured.
        flux (array):                       The array of flux measurements corresponding to each wavelength.
        initial_guess (list):               A list of initial guess parameters for the Gaussian fit:
                                            [line_centre, line_depth, line_sigma]
        bounds (tuple or list, optional):   Bounds to be used when calling curve_fit optimization.

    Returns:
        tuple: 
            - popt (array): The optimized parameters of the Gaussian model [line_centre, line_depth, line_sigma].
            - pcov (2D array): The covariance matrix of the parameters estimated. Diagonal elements provide the variance
                               of the parameter estimate. Off-diagonal elements are covariances between parameters.

    This function is particularly useful for analyzing astronomical spectra to characterize the properties
    of absorption lines.
    """

    if len(initial_guess) != 3:
        raise ValueError("Initial guess must contain 3 values: [line_centre, line_depth, line_sigma].")
    if bounds is not None and len(bounds) != 2:
        raise ValueError("Bounds must be a tuple or list of length 2, containing the lower and upper bounds for each of the 3 parameters [line_centre, line_depth, line_sigma].")

    # Fit a Gaussian to the spectrum
    if bounds is not None:
        popt, pcov = curve_fit(gaussian_absorption_profile, wavelength, flux, p0=initial_guess, bounds=bounds)
    else:
        popt, pcov = curve_fit(gaussian_absorption_profile, wavelength, flux, p0=initial_guess)
    return (popt, pcov)

def calculate_barycentric_velocity_correction(fits_header):
    """
    Calculates the barycentric velocity correction for a given astronomical observation by taking into account
    the Earth's motion relative to the solar system's barycenter. This correction is computed based on the
    right ascension (RA) and declination (Dec) of the observed object, as well as the Universal Time (UT)
    expressed in Modified Julian Date (MJD), as specified in the header of a FITS file.

    This function uses astropy.coordinates.SkyCoord for celestial coordinate handling and astropy.time.Time
    for time format conversions, ensuring precise astronomical calculations. The Siding Spring Observatory (SSO)
    location should be predefined as an astropy EarthLocation object within the function.

    Parameters:
        fits_header (dict): Header from a Veloce FITS file that must include:
            - 'MEANRA': Mean right ascension of the observation in degrees.
            - 'MEANDEC': Mean declination of the observation in degrees.
            - 'UTMJD': Universal Time of the observation in Modified Julian Date format.

    Returns:
        float: The barycentric velocity correction in kilometers per second (km/s). This value represents the velocity
               necessary to adjust for the Earth's motion when analyzing spectral data, improving the accuracy of radial
               velocity measurements.
    """

    object_coordinates = SkyCoord(ra = fits_header['MEANRA'], dec = fits_header['MEANDEC'], frame="icrs", unit="deg")
    vbary_corr_kms = object_coordinates.radial_velocity_correction( 
        kind='barycentric', 
        obstime = Time(val=fits_header['UTMJD'],format='mjd', scale='utc'),
        location=SSO
    ).to(u.km/u.s).value
    return vbary_corr_kms

def match_month_to_date(date):
    """
    Extracts the month from a date string formatted as 'YYYYMMDD' or 'YYMMDD' and returns it as a three-letter abbreviation.

    Parameters:
        date (str): A string representing a date, formatted as 'YYYYMMDD' or 'YYMMDD'. The year ('YYYY' or 'YY'), month ('MM'),
                    and day ('DD') must each be correctly positioned in the string for accurate processing.

    Returns:
        str: The three-letter abbreviation of the month (e.g., 'jan' for January, 'feb' for February, etc.),
             corresponding to the month number extracted from the date string.
    """
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    
    return(months[int(date[-4:-2])-1])

def polynomial_function(x, *coeffs):
    """
    Evaluates a polynomial at specified points. This function computes the value of a polynomial with given coefficients
    at each point in the input array 'x'. The polynomial is defined as y = c0 + c1*x + c2*x^2 + ... + cn*x^n, where 'cn'
    represents the nth coefficient in the polynomial.

    Parameters:
        x (array-like): An array of points at which the polynomial is to be evaluated. This can be any sequence of numbers,
                        wavelength data, or pixel numbers.
        coeffs (float): Variable length argument list representing the coefficients of the polynomial (c0, c1, c2, ..., cn).
                        These are passed in the order from the constant term (c0) to the highest power's coefficient.

    Returns:
        array: An array of the same shape as 'x', containing the calculated values of the polynomial at each point.
    """
    y = np.zeros_like(x,dtype=float)
    for i, coeff in enumerate(coeffs):
        y += coeff * x**i
    return y

def read_veloce_fits_image_and_metadata(file_path):
    """
    Reads an image and its associated metadata from a Veloce FITS file. This function extracts the full image data
    and selected metadata attributes relevant to astronomical observations.

    Parameters:
        file_path (str): Path to the FITS file from which data is to be read. The file should conform to the FITS standard
                         and contain both image data and a header with metadata.

    Returns:
        tuple:
            - full_image (numpy.ndarray): The 2D array of the image data extracted from the FITS file's primary HDU.
            - metadata (dict): A dictionary containing specific metadata entries extracted from the FITS header, including:
                * 'OBJECT': The name of the observed object
                * 'UTMJD': Universal Time at mid-exposure, expressed in Modified Julian Date
                * 'MEANRA': Mean right ascension of the observed object
                * 'MEANDEC': Mean declination of the observed object
                * 'EXPTIME': Exposure time in seconds
                * 'READOUT': Readout mode of the detector (either '4Amp' or '2Amp')
    """

    # Initialize a dictionary to store metadata
    metadata = dict()
    
    # Open the FITS file and extract the image and metadata
    with fits.open(file_path) as fits_file:
        full_image = np.array(fits_file[0].data, dtype=float) # We are reading in the images in as float to allow more image corrections.
        for key in ['OBJECT','UTMJD','MEANRA','MEANDEC','EXPTIME']:
            metadata[key] = fits_file[0].header[key]

        # Set the readout mode to 4Amp or 2Amp based on the presence of the 'DETA3X' keyword in the header (more than 2 Detector Amplifiers)
        if 'DETA3X' in fits_file[0].header:
            metadata['READOUT'] = '4Amp'
        else:
            metadata['READOUT'] = '2Amp'

    return(full_image, metadata)

def identify_calibration_and_science_runs(date, raw_data_dir, each_science_run_separately = False, print_information = True):
    """
    Parses a log file in a specified directory to categorize and list calibration and science runs based on 
    the observation data. This function is tailored to handle the file structure and content specific to 
    Veloce spectrograph observations.

    Parameters:
        date (str): Date in the format 'YYMMDD'. This is used to locate the log file in the directory
                    structure that organizes files by date.
        raw_data_dir (str): The path to the directory containing raw data and log files. This should be the 
                            root directory under which data is organized by date.
        each_science_run_separately (bool): If True, each science run will be treated as a separate object. 
                                            If False, all science runs will be coadded. Default is False.

    Returns:
        tuple: 
            - calibration_runs (dict): A dictionary categorizing different types of calibration runs, each
                                       key corresponds to a run type (e.g., 'Flat', 'Bias') and its value is 
                                       a list of run identifiers.
            - science_runs (dict): A dictionary where each key is a science observation object name and its 
                                   value is a list of run identifiers associated with that object.

    The function searches for a log file matching the provided date, reads the file, and processes each line
    to extract relevant information such as run type, object observed, and other metadata. Calibration runs 
    and science runs are distinguished based on predefined criteria extracted from the log file content.

    """
    
    if print_information: print('\n  =================================================')
    if print_information: print('  ||\n  ||  --> Identifying calibration and science runs now\n  ||')

    raw_file_path = raw_data_dir+'/'+date

    log_file_path = glob.glob(raw_file_path+'*.log')
    
    raw_file_path = f"{raw_data_dir}/{date}/"

    log_file_path = glob.glob(f"{raw_file_path}*.log")
    if not log_file_path:
        raise ValueError('No Log file present in '+f"{raw_file_path}*.log")
    elif len(log_file_path) > 1:
        print('  ||'+f'  --> More than 1 Log file present, continuing with {log_file_path[0]}')
    else:
        print('  ||'+f'  --> Found Log file {log_file_path[0]}')
    log_file_path = log_file_path[0]

    with open(log_file_path, "r") as log_file:
        log_file_text = log_file.read().split('\n')

    # Initialization of dictionaries to collect calibration and science runs
    calibration_runs = {key: [] for key in [
        'FibTh_15.0', 'FibTh_60.0', 'FibTh_180.0', 'SimTh_15.0', 'SimTh_60.0', 'SimTh_180.0', 'SimLC',
        'Flat_0.1', 'Flat_1.0', 'Flat_10.0', 'Flat_60.0', 'Bias'
    ]}
    calibration_runs['Bstar'] = dict()
    calibration_runs['Darks'] = dict()
    
    science_runs = {}


    # Process each line in the log file to extract and categorize runs
    for line in log_file_text:

        # Identify runs via their numeric value
        run = line[:4]
        if not run.isnumeric():
            pass
        elif (('CRAP' in line) | ('crap' in line) | ('Crap' in line) | ('Unknown' in line)):
            pass
        else:

            ccd = line[6]
            # To handle long object names (e.g. Gaia DR3 source_ids,
            # we reverse engineer via the UTC ":" that follows the object)
            utc_colon = line.find(":")
            run_object = line[8:utc_colon-2].strip()
            utc = line[utc_colon-2:utc_colon-25+32].strip()
            exposure_time = line[utc_colon-25+34:utc_colon-25+42].strip()
            # snr_noise = line[utc_colon-25+42:utc_colon-25+48].strip()
            # snr_photons = line[utc_colon-25+48:utc_colon-25+53].strip()
            # seeing = line[utc_colon-25+55:utc_colon-25+59].strip()
            # lc_status = line[utc_colon-25+60:utc_colon-25+62].strip()
            # thxe_status = line[utc_colon-25+63:utc_colon-25+67].strip()
            # read_noise = line[utc_colon-25+70:utc_colon-25+85].strip()
            # airmass = line[utc_colon-25+87:utc_colon-25+91].strip()
            overscan = line[utc_colon-25+95:].split()[0]
            comments = line[utc_colon-25+96+len(overscan):]
            if len(comments) > 1:
                if (run_object != 'FlatField-Quartz') & print_information:
                    print('  ||  --> Warning for '+run_object+' (run '+run+'): '+comments)

            # Read in type of observation from CCD3 info (since Rosso should always be available)
            if ccd == '3':
                if run_object == 'SimLC':
                    calibration_runs['SimLC'].append(run)
                elif run_object == 'BiasFrame':
                    calibration_runs['Bias'].append(run)
                elif run_object == 'FlatField-Quartz':
                    calibration_runs['Flat_'+exposure_time].append(run)
                elif run_object == 'ARC-ThAr':
                    calibration_runs['FibTh_'+exposure_time].append(run)
                elif run_object == 'SimThLong':
                    calibration_runs['SimTh_'+exposure_time].append(run)
                elif run_object == 'SimTh':
                    calibration_runs['SimTh_'+exposure_time].append(run)
                elif run_object == 'Acquire':
                    pass
                elif run_object == 'DarkFrame':
                    if exposure_time in calibration_runs['Darks'].keys():
                        calibration_runs['Darks'][exposure_time].append(run)
                    else:
                        calibration_runs['Darks'][exposure_time] = [run]
                else:
                    # Run Bstars both as calibration and science frames (the latter is important for flux calibration post-processing)
                    if run_object in [
                    "10144","14228","37795","47670","50013","56139","89080","91465","93030","98718","105435","105937","106490","108248","108249","108483",
                    "109026","109668","110879","118716","120324","121263","121743","121790","122451","125238","127972","129116","132058","134481","136298",
                    "136504","138690","139365","142669","143018","143118","143275","144470","157246","158094","158427","158926","160578","165024","169022",
                    "175191","209952"
                    ]:
                        calibration_runs['Bstar'][utc] = [run_object, run, utc]
                        run_object = 'HD'+run_object

                    if each_science_run_separately:
                        science_runs[run_object+'_'+str(run)] = [run]
                    else:
                        if run_object in science_runs.keys():
                            science_runs[run_object].append(run)
                        else:
                            science_runs[run_object] = [run]

    # Print all DarkFrame exposures, if any
    dark_frames = [key for key in calibration_runs['Darks'].keys()]
    if (len(dark_frames) > 0):
        if print_information: print('  ||\n  || DarkFrame observations: '+', '.join(dark_frames))
    else:
        if print_information: print('  ||\n  || No DarkFrame observations identified.')
                        
    if len(calibration_runs['Bstar']) > 0:
        if print_information: print('  ||\n  || Bstar observations happened at: '+', '.join(calibration_runs['Bstar'].keys()))
    else:
        if print_information: print('  ||\n  || No Bstar observations identified.')

    if print_information: print('  ||\n  || The following science observations were identified: '+', '.join(list(science_runs.keys())))

    if len(science_runs) > 0:
        directory_path = Path(config.working_directory+'reduced_data/'+config.date)
        directory_path.mkdir(parents=True, exist_ok=True)
        if print_information: print('  ||\n  || Will save reduced data into directory '+str(directory_path))
        if print_information: print('  ||\n  =================================================\n')

    return(calibration_runs, science_runs)


def interpolate_spectrum(wavelength, flux, target_wavelength):
    """
    Interpolates a spectrum's flux values to match a target wavelength array. This function uses cubic 
    interpolation to estimate flux values at new wavelength points specified by the target array.

    Parameters:
        wavelength (array-like): The array of wavelength data points corresponding to the input spectrum.
                                 These should be in increasing order.
        flux (array-like): The array of flux values associated with each wavelength in the input spectrum.
        target_wavelength (array-like): The array of wavelength data points where the flux values need to
                                        be interpolated. This array should also be in increasing order.

    Returns:
        array: An array of flux values interpolated at the target wavelengths. The interpolation is performed 
               using a cubic spline, which provides a smooth and continuous estimate of the flux between 
               known data points. If the target wavelength extends beyond the range of the input wavelength,
               the function fills the values with 1.0 based on the fill_value parameter.

    """

    interpolation_function = interp1d(wavelength, flux, bounds_error=False, fill_value=(1.0,1.0), kind='cubic')
    return interpolation_function(target_wavelength)

def lasercomb_wavelength_from_numbers(n, repeat_frequency_ghz = 25.00000000, offset_frequency_ghz = 9.56000000000):
    """
    Calculates the wavelength of a laser comb (LC) line based on its mode number, repeat frequency,
    and offset frequency using the formula:
    
    lambda_n = speed_of_light / (n * f_repeat + f_offset)
    
    Here, we are using the LC mode frequency f_n = n * f_repeat + f_offset.
    For ease, we use speed_of_light already in units of ÅGHz, i.e. c = 2.9979246 * 10**9 ÅGHz.

    Parameters:
        n (int or array-like): The mode number(s) of the LC line.
        repeat_frequency_ghz (float): The repeat frequency of the laser comb in GHz. Default is 25.0 GHz.
        offset_frequency_ghz (float): The offset frequency of the laser comb in GHz. Default is 9.56 GHz.

    Returns:
        float or ndarray: The calculated wavelength(s) in Angstroms (Å).
    """
    
    return(2.9979246 * 10**9 / (n * repeat_frequency_ghz + offset_frequency_ghz))

def lasercomb_numbers_from_wavelength(wavelength_aangstroem, repeat_frequency_ghz = 25.00000000, offset_frequency_ghz = 9.56000000000):
    """
    Calculates the mode number of a laser comb (LC) line based on a wavelength, repeat frequency,
    and offset frequency using the formula:
    
    n  = (speed_of_light / lambda_n - f_offset) / f_repeat

    Here, we are using the LC mode frequency f_n = n * f_repeat + f_offset.
    For ease, we use speed_of_light already in units of ÅGHz, i.e. c = 2.9979246 * 10**9 ÅGHz.

    Parameters:
        wavelength_aangstroem (float or array-like): The wavelength(s) of the LC line in Ångström.
        repeat_frequency_ghz (float): The repeat frequency of the laser comb in GHz. Default is 25.0 GHz.
        offset_frequency_ghz (float): The offset frequency of the laser comb in GHz. Default is 9.56 GHz.

    Returns:
        float or ndarray: The calculated mode number(s). Note: These are floats, not integers!
    """

    return(((2.9979246*10**9 / wavelength_aangstroem) - offset_frequency_ghz) / repeat_frequency_ghz)

def read_in_wavelength_solution_coefficients_tinney():
    """
    Reads wavelength solution coefficients by C. Tinney from predefined vdarc* files.

    In original files, Azzurro and Verde are in air and Rosso in vacuum.
    Reference pixels (DYO) are defined for Verde and Rosso, and assumed to be 2450 for Azzurro.
    
    For consistency, the function is converted to vacuum and reevaluated for reference pixel 2048

    Returns:
        dict: A dictionary containing wavelength solution coefficients for each CCD and spectral order.
              The keys are formatted as 'ccd_{ccd_number}_order_{order_number}' and the values are arrays
              of coefficients, with the last array entry being DY0 if present, otherwise 2450 as default.
    """
    
    wavelength_solution_coefficients_tinney = dict()

    pixel_array = np.arange(4096)-2048

    for ccd in [1,2,3]:
        if ccd == 1: vdarc_file = Path(__file__).resolve().parent / 'veloce_reference_data' / 'vdarc_azzurro_230915.txt' # in air
        if ccd == 2: vdarc_file = Path(__file__).resolve().parent / 'veloce_reference_data' / 'vdarc_verde_230920.txt' # in air
        if ccd == 3: vdarc_file = Path(__file__).resolve().parent / 'veloce_reference_data' / 'vdarc_rosso_230919.txt' # in vacuum

        with open(vdarc_file, "r") as vdarc:
            vdarc_text = vdarc.read().split('\n')

        for line in vdarc_text:
            if 'START' in line:
                order = line[6:]
                DYO = None
            elif 'COEFFS' in line:
                coeffs = np.array(line[7:].split(' '),dtype=float)
            elif 'DY0' in line:
                DYO = int(line[4:])
            elif 'STOP' in line:
                if DYO is None:
                    DYO = 2450
                
                # Reformat coefficients to go from Tinney reference pixel (DYO) to 2048
                x = np.arange(0, 4096)
                dx_old = x - DYO
                lam = np.zeros_like(x,dtype=float)
                for i, coeff in enumerate(coeffs):
                    lam += coeff * dx_old**i
                dx_new = x - 2048

                coeffs_wrt_2048 = np.polyfit(dx_new, lam, len(coeffs)-1)[::-1]  # reverse to get d0..d5

                wavelength_solution_coefficients_tinney['ccd_'+str(ccd)+'_order_'+order] = np.concatenate((coeffs_wrt_2048, [DYO]))
                
                # convert native air of CCDs 1 and 2 to vacuum (to be consistent with CCD3 and LC etc.)    
                if ccd in [1,2]:
                    wavelengths_native_tinney = polynomial_function(pixel_array, *coeffs_wrt_2048)*10
                    wavelengths_vacuum = wavelength_air_to_vac(wavelengths_native_tinney)

                    thxe_coefficients_vacuum, _ = curve_fit(
                        polynomial_function,
                        pixel_array,
                        wavelengths_vacuum/10.,
                        p0=coeffs_wrt_2048
                    )
                    wavelength_solution_coefficients_tinney['ccd_'+str(ccd)+'_order_'+order] = np.concatenate((thxe_coefficients_vacuum, [DYO]))
                else:
                    thxe_coefficients_vacuum = coeffs_wrt_2048

                # In case the ThXe reference files have to be overwritten, uncomment the line below.
                # np.savetxt(Path(__file__).resolve().parent / 'wavelength_coefficients' / 'wavelength_coefficients_'+'ccd_'+str(ccd)+'_order_'+order+'_thxe.txt',thxe_coefficients_vacuum)

    return(wavelength_solution_coefficients_tinney)

def wavelength_vac_to_air(wavelength_vac):
    """
    Convert vacuum wavelengths to air wavelengths using the formula by Birch & Downs (1994, Metro, 31, 315).
    Consistent to the 2024 version of Korg (https://github.com/ajwheeler/Korg.jl).

    Parameters:
        wavelength_vac (float, array): The vacuum wavelength to be converted to air wavelength.

    Returns:
        float, array: The corresponding air wavelength(s) calculated from the input vacuum wavelength(s).
    """
    return(wavelength_vac / (1 + 0.0000834254 + 0.02406147 / (130 - (1e4/wavelength_vac)**2) + 0.00015998 / (38.9 - (1e4/wavelength_vac)**2)))
           
def wavelength_air_to_vac(wavelength_air):
    """
    Convert air wavelengths to vacuum wavelengths using the formula by Birch & Downs (1994, Metro, 31, 315).
    Consistent to the 2024 version of Korg (https://github.com/ajwheeler/Korg.jl).

    Parameters:
        wavelength_air (float, array): The air wavelength to be converted to vacuum wavelength.

    Returns:
        float, array: The corresponding vacuum wavelength(s) calculated from the input air wavelength(s).
    """
    return(wavelength_air * (1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - (1e4 / wavelength_air)**2) + 0.0001599740894897 / (38.92568793293 - (1e4 / wavelength_air)**2)))

def check_repeated_observations(science_runs):
    """
    This function checks if any science run targets have been observed more than once and
    returns the names of the specific runs that were observed repeatedly.

    Parameters:
    science_runs (dict): A dictionary where keys are target names with run identifiers
                         (formatted as target+'_'+run) and values are some details about the run.

    Returns:
    dict: A dictionary with each target and a list of runs if it has been observed repeatedly.
    """
    # Create a dictionary to store the runs for each base target (without the run number)
    target_runs = {}

    # Loop through each key in the input dictionary
    for key in science_runs.keys():
        # Check if the key contains an underscore and split appropriately
        if '_' in key:
            target, run = key.rsplit('_', 1)
            # Collect runs associated with each target
            if target in target_runs:
                target_runs[target].append(run)
            else:
                target_runs[target] = [run]

    # Create a dictionary to store targets with multiple observations
    repeated_observations = {target: runs for target, runs in target_runs.items() if len(runs) > 1}

    return repeated_observations

def monitor_vrad_for_repeat_observations(date, repeated_observations):
    """
    This function reads in the RV measurements for repeated observations and plots them for comparison.
    
    Parameters:
    date (str): The date of the observations in the format 'YYMMDD'.
    repeated_observations (dict): A dictionary containing the repeated observations, where each key is the target
                                  and the value is a list of run identifiers for that target.
    """

    if len(repeated_observations) == 0: print('  --> No repeated observations found.')
    else:
        print('  --> Repeat observations found for: '+','.join(list(repeated_observations.keys())))

        # Loop through each repeated observation
        for repeated_observation in repeated_observations.keys():
            print('\n  --> Monitoring RV for '+repeated_observation)
            
            # We will read out UTMJD, VRAD, and E_VRAD
            utmjd = []
            vrad = []
            e_vrad = []
            
            for run in repeated_observations[repeated_observation]:
                expected_path = config.working_directory+'/reduced_data/'+date+'/'+repeated_observation+'_'+run+'/veloce_spectra_'+repeated_observation+'_'+run+'_'+date+'.fits'
                try:
                    with fits.open(expected_path) as file:
                        utmjd.append(file[0].header['UTMJD'])
                        vrad.append(file[0].header['VRAD'])
                        e_vrad.append(file[0].header['E_VRAD'])
                except:
                    print('\n  --> Could not read '+repeated_observation+'_'+run)
                    print('  --> Expected path was: '+expected_path)

            utmjd = np.array(utmjd)
            vrad = np.array(vrad)
            e_vrad = np.array(e_vrad)

            finite_vrad = np.where([vrad_value != 'None' for vrad_value in vrad])[0]

            if len(finite_vrad) > 1:

                # Limit to finite float values (aka not 'None')
                utmjd = np.array(utmjd[finite_vrad], dtype=float)
                vrad = np.array(vrad[finite_vrad], dtype=float)
                e_vrad = np.array(e_vrad[finite_vrad], dtype=float)

                # Plot the RV measurements
                f, ax = plt.subplots()
                ax.errorbar(
                    utmjd - int(np.floor(utmjd[0])), # MJD of the first observation
                    vrad,
                    yerr = e_vrad,
                    fmt = 'o'
                )
                ax.set_xlabel('Modified Julian Date MJD - '+str(int(np.floor(utmjd[0]))),fontsize=15)
                ax.set_ylabel(r'Radial Velocity $v_\mathrm{rad}~/~\mathrm{km\,s^{-1}}$',fontsize=15)
                ax.axhline(np.mean(vrad),c = 'C3',lw=2,ls='dashed',label = r'$\langle v_\mathrm{rad} \rangle = '+"{:.2f}".format(np.round(np.mean(vrad),2))+r' \pm '+"{:.2f}".format(np.round(np.std(vrad),2))+r'\,\mathrm{km\,s^{-1}}$')
                ax.axhline(np.mean(vrad)-np.std(vrad),c = 'C1',lw=1,ls='dashed')
                ax.axhline(np.mean(vrad)+np.std(vrad),c = 'C1',lw=1,ls='dashed')
                ax.legend()
                plt.savefig(config.working_directory+'/reduced_data/'+date+'/'+repeated_observation+'_vrad_monitoring.pdf')
                if 'ipykernel' in sys.modules: plt.show()
                plt.close()
            else:
                print('Less than two observations could be read in for '+repeated_observation)
                print('Skipping plotting for '+repeated_observation)

def get_memory_usage():
    """
    Get the memory usage of the system and print it in a human-readable format.
    This function uses the `free` command on Linux systems and `vm_stat` on macOS to retrieve memory usage information.
    """
    os_type = platform.system()
    
    if os_type == 'Linux':
        command = ["free", "-m"]
    elif os_type == 'Darwin':
        command = ["vm_stat"]
    else:
        raise Exception("Unsupported operating system")
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if os_type == 'Darwin':
        # Process macOS memory output
        lines = result.stdout.split('\n')
        for line in lines:
            if "Pages free" in line:
                # Extracting the number of free pages
                free_pages = line.split(':')[1].strip().replace('.', '')
                free_pages = int(free_pages)
                # macOS uses a page size of 4096 bytes, convert pages to megabytes
                page_size = os.sysconf('SC_PAGE_SIZE') / 1024 / 1024  # Convert bytes to MB
                free_memory = free_pages * page_size
                return('Run on Apple/Darwin: '+"{:.1f}".format(free_memory)+'MB')
    else:
        return(result.stdout)

def degrade_spectral_resolution(wavelength, flux, original_resolution, target_resolution):

    """
    Degrade the spectral resolution of a given flux from an original to a target resolution.

    Parameters:
        wavelength (numpy.ndarray): Wavelength array of the spectrum.
        flux (numpy.ndarray): Flux array of the spectrum.
        original_resolution (float): Original spectral resolution.
        target_resolution (float): Target spectral resolution to degrade to.

    Returns:
        numpy.ndarray: The degraded flux array.
    """

    # Calculate the FWHM at the original and target resolutions
    fwhm_original = wavelength / original_resolution
    fwhm_target = wavelength / target_resolution
    
    # Calculate the required broadening in FWHM
    fwhm_diff = np.sqrt(np.maximum(fwhm_target**2 - fwhm_original**2, 0))
    
    # Convert FWHM difference to sigma for the Gaussian kernel
    sigma = fwhm_diff / (2.0 * np.sqrt(2 * np.log(2)))

    # Assume sigma varies across wavelength; use mean sigma for simplicity in this example
    mean_sigma = np.mean(sigma)

    # Convolve flux with Gaussian kernel to degrade resolution
    degraded_flux = gaussian_filter1d(flux, mean_sigma)

    return(degraded_flux)

def update_fits_header_via_crossmatch_with_simbad(fits_header):
    """
    Update the FITS header by crossmatching the object ID with Simbad and adding relevant information.

    Parameters:
        fits_header (astropy.io.fits.header.Header): The FITS header to be updated. It should contain the following keys:
            - 'OBJECT': The name of the observed object.
            - 'MEANRA': The mean right ascension of the observation.
            - 'MEANDEC': The mean declination of the observation.

    Returns:
        astropy.io.fits.header.Header: The updated FITS header with additional information from the Simbad crossmatch.
    """

    # Let's get the most important info from the existing fits_header
    object_id = fits_header['OBJECT']
    ra        = fits_header['MEANRA']
    dec       = fits_header['MEANDEC']

    print('\n  --> Updating FITS header for '+object_id+' with RA and Dec: '+str(ra)+' and '+str(dec))

    if object_id[-5] == '_':
        print('\n  --> this seems to be the specific run '+object_id[-4:]+' of an object; adjusting OBJECT to reflect this')
        fits_header['OBJECT'] = object_id[:-5]
        object_id = object_id[:-5]

    # First, let's identify the likely way we want to query Simbad
    if object_id[:3] == 'HIP':
        print('\n  --> Identified '+object_id+' as a HIP catalogue entry. Crossmatching accordingly.')
        query_simbad_object = object_id
    elif object_id.isdigit() and len(object_id) > 10:
        print('\n  --> Identified '+object_id+' as a Gaia DR3 catalogue entry. Crossmatching accordingly.')
        query_simbad_object = 'Gaia DR3 '+object_id
    elif 'TIC' in object_id:
        print('\n  --> Identified '+object_id+' as a TIC catalogue entry. Crossmatching accordingly.')
        query_simbad_object = object_id
    else:
        print('\n  --> '+object_id+' does not look like Gaia DR3 or HIP entry. Crossmatching with Object Name and otherwise via Ra/Dec.')
        query_simbad_object = object_id

    # Now we query Simbad for object
    try:
        simbad_match = simbad_simple.query_object(query_simbad_object)
    except:
        print('  --> ConnectionError for Simbad. Try again later!')
        return(fits_header)

    if len(simbad_match) == 0:
        print('  --> Did not find a match in Simbad via object_id. Trying Ra/Dec now.')
        object_coordinate = SkyCoord(ra = float(ra), dec = float(dec), frame='icrs', unit='deg')
        try:
            simbad_match = simbad_simple.query_region(object_coordinate, radius=10*u.arcsec)
        except ConnectionError as e:
            raise ConnectionError('  --> ConnectionError for Simbad. Try again later! Error Message: '+str(e))

    # Let's check how many matches we got and return if there are none:
    if len(simbad_match) == 0:
        print('  --> Did not find a match in Simbad.')
        return(fits_header)
    elif len(simbad_match) == 1:
        simbad_match = simbad_match[0]
    else:
        print('  --> Found more than one entry in Simbad. Using the first match.')
        simbad_match = simbad_match[0]

    # Now let's try to identify other IDS, radial velocity, and parallax
    try:
        simbad_ids_vrad_plx = simbad_ids_vrad_plx_query.query_object(simbad_match['main_id'])
    except ConnectionError as e:
        raise ConnectionError('  --> ConnectionError for Simbad. Try again later! Error Message: '+str(e))
    # Let's check how many matches we got and return if there are none:
    if len(simbad_ids_vrad_plx) >= 1:
        simbad_ids_vrad_plx = simbad_ids_vrad_plx[0]
    elif len(simbad_ids_vrad_plx) == 0:
        print('  --> Did not find a match in Simbad for other IDS, radial velocity, and parallax.')

    # Now let's try to identify Teff, logg, and [Fe/H] measurements.
    try:
        simbad_match_fe_h = simbad_fe_h_query.query_object(simbad_match['main_id'])
    except ConnectionError as e:
        raise ConnectionError('  --> ConnectionError for Simbad. Try again later! Error Message: '+str(e))
    
    # Check how many measurements we have for Teff, logg, and [Fe/H]
    if len(simbad_match_fe_h) == 0: print('  --> Did not find a match in Simbad for Fe/H.')
    elif len(simbad_match_fe_h) == 1: simbad_match_fe_h = simbad_match_fe_h[0]
    else:
        print('  --> Found more than one entry in Simbad for Fe/H. Using Soubiran et al. (2024) if available, otherwise latest publication with finite values.')

        # Check if the star is in Soubiran et al. (2024):
        soubiran_2024 = np.where(simbad_match_fe_h['mesfe_h.bibcode'] == '2024A&A...682A.145S')[0]
        if len(soubiran_2024) > 0:
            simbad_match_fe_h = simbad_match_fe_h[soubiran_2024[0]]
            print('      --> Found '+object_id+' in Soubiran et al. (2024). Using this entry for TEFF/LOGG/FE_H:',simbad_match_fe_h['mesfe_h.teff'], simbad_match_fe_h['mesfe_h.log_g'], simbad_match_fe_h['mesfe_h.fe_h'])
        else:

            finite_entries = simbad_match_fe_h[~(simbad_match_fe_h['mesfe_h.teff'].mask | simbad_match_fe_h['mesfe_h.log_g'].mask | simbad_match_fe_h['mesfe_h.fe_h'].mask)]
            if len(finite_entries) > 0:
                finite_entries.sort('mesfe_h.bibcode', reverse=True)
                simbad_match_fe_h = finite_entries[0]
                print('      --> Average values for this star are: ',int(np.mean(finite_entries['mesfe_h.teff'])),np.round(np.mean(finite_entries['mesfe_h.log_g']),2),np.round(np.mean(finite_entries['mesfe_h.fe_h']),2))
                print('      --> Using measurements from '+simbad_match_fe_h['mesfe_h.bibcode']+' for TEFF/LOGG/FE_H:',simbad_match_fe_h['mesfe_h.teff'], simbad_match_fe_h['mesfe_h.log_g'], simbad_match_fe_h['mesfe_h.fe_h'])
            else:
                print('      --> Did not find any finite entries for TEFF/LOGG/FE_H. Using the first entry.')
                simbad_match_fe_h = simbad_match_fe_h[0]
                if not abs(simbad_match_fe_h['mesfe_h.teff']) >= 0.0:
                    simbad_match_fe_h['mesfe_h.teff'] = None
                if not abs(simbad_match_fe_h['mesfe_h.log_g']) >= 0.0:
                    simbad_match_fe_h['mesfe_h.log_g'] = None
                if not abs(simbad_match_fe_h['mesfe_h.fe_h']) >= 0.0:
                    simbad_match_fe_h['mesfe_h.fe_h'] = 0.0
                    print('      --> Setting [Fe/H] to 0.0')

    # Now let's try to identify magnitude measurements.
    try:
        simbad_match_magnitudes = simbad_magnitudes_query.query_object(simbad_match['main_id'])
    except ConnectionError as e:
        raise ConnectionError('  --> ConnectionError for Simbad. Try again later! Error Message: '+str(e))

    # Check how many measurements we have for magnitudes
    if len(simbad_match_magnitudes) == 0:
        print('  --> Did not find a match in Simbad for magnitudes.')
    else:
        if len(simbad_match_magnitudes) > 1:
            print('  --> Found more than one entry in Simbad for magnitudes. Using the first match.')
        simbad_match_magnitudes = simbad_match_magnitudes[0]

        for filter in ['B','V','G','R']:
            if isinstance(simbad_match_magnitudes[filter], np.ma.MaskedArray):
                simbad_match_magnitudes[filter] = simbad_match_magnitudes[filter].filled(np.nan)

        # Veloce is meant to observe only down to 12th magnitude.
        # Let's test if the object is bright enough for Veloce (G < 12 mag) and print a warning if not.
        if np.isfinite(simbad_match_magnitudes['G']):
            if simbad_match_magnitudes['G'] > 12: print('  --> Warning: Match fainter than G > 12 mag. Right match for a Veloce observations?')

    # Let's add some more information from the crossmatches with HIP/2MASS/Gaia DR3 and other literature where available
    if len(simbad_ids_vrad_plx['ids']) > 0:
        ids = simbad_ids_vrad_plx['ids']
        unique_ids = np.array(ids.split("|"))
        match_with_hip_tmass_gaia = []
        
        # Let's check if the star is in HIP, 2MASS, and Gaia DR3 according to Simbad
        if 'HIP ' in ids:
            hip = unique_ids[[x[:4] == 'HIP ' for x in unique_ids]][0]
            match_with_hip_tmass_gaia.append(hip)
            fits_header['HIP_ID'] = (int(hip[4:]), 'Hipparcos Catalogue Identifier')
        if '2MASS ' in ids:
            tmass = unique_ids[[x[:6] == '2MASS ' for x in unique_ids]][0]
            match_with_hip_tmass_gaia.append(tmass)
            fits_header['TMASS_ID'] = (tmass[7:], '2MASS catalogue identifier (2MASS J removed)')
        if 'Gaia DR3 ' in ids:
            source_id = unique_ids[[x[:9] == 'Gaia DR3 ' for x in unique_ids]][0]
            match_with_hip_tmass_gaia.append(source_id)
            fits_header['GAIA_ID'] = (int(source_id[9:]), 'Gaia DR3 source_id (Gaia DR3 removed)')
        print('  --> Matches in HIP/2MASS/Gaia DR3: '+', '.join(match_with_hip_tmass_gaia))

        # Now let's add some literature information on radial velocity (if they are finite)
        if not np.ma.is_masked(simbad_ids_vrad_plx['rvz_radvel']):
            fits_header['VRAD_LIT'] = (simbad_ids_vrad_plx['rvz_radvel'], 'Radial velocity from literature')
        if not np.ma.is_masked(simbad_ids_vrad_plx['rvz_err']): fits_header['HIERARCH E_VRAD_LIT'] = (simbad_ids_vrad_plx['rvz_err'], 'Radial velocity error from literature')
        if 'VRAD_LIT' in fits_header.keys():
            fits_header['VRAD_BIB'] = (simbad_ids_vrad_plx['rvz_bibcode'], 'Bibcode of VRAD_LIT')
        
        # Now let's add some literature information on parallax (if they are finite)
        if not np.ma.is_masked(simbad_ids_vrad_plx['plx_value']): fits_header['PLX'] = (simbad_ids_vrad_plx['plx_value'], 'Parallax in mas ('+simbad_ids_vrad_plx['plx_bibcode']+')')
        if not np.ma.is_masked(simbad_ids_vrad_plx['plx_err']): fits_header['E_PLX'] = (simbad_ids_vrad_plx['plx_err'], 'Parallax error in mas ('+simbad_ids_vrad_plx['plx_bibcode']+')')

    if np.all(['VRAD_LIT' in fits_header.keys(),'E_VRAD_LIT' in fits_header.keys()]):
        print('  --> Found literature VRAD/E_VRAD in Simbad: '+str(fits_header['VRAD_LIT'])+' +/- '+str(fits_header['E_VRAD_LIT'])+' km/s by '+str(fits_header['VRAD_BIB']))
    elif 'VRAD_LIT' in fits_header.keys():
        print('  --> Found literature VRAD in Simbad: '+str(fits_header['VRAD_LIT'])+' km/s by '+str(fits_header['VRAD_BIB']))

    # Add literature information on stellar parameters Teff/logg/[Fe/H]
    if len(simbad_match_fe_h) > 0:
        if np.isfinite(simbad_match_fe_h['mesfe_h.teff']): fits_header['TEFF_LIT'] = (simbad_match_fe_h['mesfe_h.teff'], 'Effective temperature from Simbad')
        if np.isfinite(simbad_match_fe_h['mesfe_h.log_g']): fits_header['LOGG_LIT'] = (simbad_match_fe_h['mesfe_h.log_g'], 'Surface gravity from Simbad')
        if np.isfinite(simbad_match_fe_h['mesfe_h.fe_h']): fits_header['FE_H_LIT'] = (simbad_match_fe_h['mesfe_h.fe_h'], 'Iron abundance from Simbad')
        if simbad_match_fe_h['mesfe_h.bibcode'] is not None: fits_header['TLF_BIB'] = (simbad_match_fe_h['mesfe_h.bibcode'], 'Bibcode of Simbad TEFF/LOGG/FE_H')
    else:
        fits_header['TEFF_LIT'] = ('None', 'Effective temperature from Simbad')
        fits_header['LOGG_LIT'] = ('None', 'Surface gravity from Simbad')
        fits_header['FE_H_LIT'] = ('None', 'Iron abundance from Simbad')
        fits_header['TLF_BIB'] = ('None', 'Bibcode of Simbad TEFF/LOGG/FE_H')

    if np.all(['TEFF_LIT' in fits_header.keys(),'TEFF_LIT' in fits_header.keys(),'LOGG_LIT' in fits_header.keys(),'FE_H_LIT' in fits_header.keys()]):
        print('  --> Found literature TEFF/LOGG/FE_H in Simbad: '+str(fits_header['TEFF_LIT'])+'/'+str(fits_header['LOGG_LIT'])+'/'+str(fits_header['FE_H_LIT'])+' by '+str(fits_header['TLF_BIB']))
    elif 'FE_H_LIT' in fits_header.keys():
        print('  --> Found literature FE_H in Simbad: '+str(fits_header['FE_H_LIT'])+' by '+str(fits_header['TLF_BIB']))

    # Add information on B/V/G/R filters (where available) and parallax
    if len(simbad_match_magnitudes) > 0:
        for bvgr_filter in ['B','V','G','R']:
            if abs(simbad_match_magnitudes[bvgr_filter]) >= 0.0:
                try:
                    fits_header[bvgr_filter+'MAG'] = (simbad_match_magnitudes[bvgr_filter], 'Mag in '+bvgr_filter+' ('+simbad_match_magnitudes['FLUX_BIBCODE_'+bvgr_filter]+')')
                except:
                    fits_header[bvgr_filter+'MAG'] = (simbad_match_magnitudes[bvgr_filter], 'Mag in '+bvgr_filter)
            else:
                pass

    return(fits_header)

def find_best_radial_velocity_from_fits_header(fits_header):
    """
    Find the best radial velocity measurement from literature and Veloce measurements.

    Parameters:
        fits_header (astropy.io.fits.header.Header): The FITS header of the 0th extension of Veloce spectra containing the radial velocity information.

    Returns:
        float: The best radial velocity measurement for calibration.
    """

    vrad_value = fits_header.get('VRAD', None)
    if vrad_value == 'None': vrad_value = None
    e_vrad_value = fits_header.get('E_VRAD', None)
    if e_vrad_value == 'None': e_vrad_value = None
    vrad_lit_value = fits_header.get('VRAD_LIT', None)
    if vrad_lit_value == 'None': vrad_lit_value = None
    e_vrad_lit_value = fits_header.get('E_VRAD_LIT', None)
    if e_vrad_lit_value == 'None': e_vrad_lit_value = None
    vrad_bib = fits_header.get('VRAD_BIB', 'N/A')

    # Check which VRAD measurement has a smaller uncertainty and use that one
    if e_vrad_value is not None and e_vrad_lit_value is not None:
        print('  --> Found literature VRAD in Simbad: '+str(vrad_lit_value)+' +/- '+str(e_vrad_lit_value)+' km/s by '+str(vrad_bib))
        print('  --> Found VRAD from Veloce measurements: '+str(vrad_value)+' +/- '+str(e_vrad_value)+' km/s')
        if e_vrad_value < e_vrad_lit_value:
            vrad_for_calibration = vrad_value
            print('  --> Using VRAD from Veloce measurements because it has a smaller uncertainty.')
        else:
            vrad_for_calibration = vrad_lit_value
            print('  --> Using literature VRAD from Simbad because it has a smaller uncertainty.')
    elif e_vrad_lit_value is not None:
        if vrad_lit_value is not None: print('  --> Found literature VRAD in Simbad: '+str(vrad_lit_value)+' by '+str(vrad_bib)+' but without uncertainty.')           
        print('  --> Found VRAD from Veloce measurements: '+str(vrad_value)+' +/- '+str(e_vrad_value)+' km/s')
        vrad_for_calibration = vrad_value
        print('  --> Using VRAD from Veloce measurements because no literature VRAD with uncertainty is available.')
    elif vrad_lit_value is not None:
        if e_vrad_lit_value is not None: print('  --> Found literature VRAD in Simbad: '+str(vrad_lit_value)+' +/- '+str(e_vrad_lit_value)+' km/s by '+str(vrad_bib))
        else:
            print('  --> Found literature VRAD in Simbad: '+str(vrad_lit_value)+' by '+str(vrad_bib)+' but without uncertainty.')
        vrad_for_calibration = vrad_lit_value
        print('  --> Using literature VRAD from Simbad because no Veloce VRAD is available.')
    elif e_vrad_value is not None:
        print('  --> Found VRAD from Veloce measurements: '+str(vrad_value)+' +/- '+str(e_vrad_value)+' km/s')
        vrad_for_calibration = vrad_value
        print('  --> Using VRAD from Veloce measurements because no literature VRAD is available.')
    else: raise ValueError('No valid option for VRAD avaialble. Aborting calibration via synthesis.')

    return(vrad_for_calibration)