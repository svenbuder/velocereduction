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

# Matplotlib package
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Astropy package
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
SSO = EarthLocation.of_site('Siding Spring Observatory')

# Astroquery package
import astroquery
from astroquery.simbad import Simbad
Simbad.add_votable_fields('ids')
Simbad.add_votable_fields('velocity')
Simbad.add_votable_fields('parallax')
if astroquery.__version__ < '0.4.8':
    Simbad.add_votable_fields('fe_h')
    simbad_teff = 'Fe_H_Teff'
    simbad_logg = 'Fe_H_log_g'
    simbad_fe_h = 'Fe_H_Fe_H'
    simbad_fe_h_ref = 'Fe_H_bibcode'
    Simbad.add_votable_fields('fluxdata(B)'); simbad_b = 'FLUX_B'
    Simbad.add_votable_fields('fluxdata(V)'); simbad_v = 'FLUX_V'
    Simbad.add_votable_fields('fluxdata(G)'); simbad_g = 'FLUX_G'
    Simbad.add_votable_fields('fluxdata(R)'); simbad_r = 'FLUX_R'
    plx_error = 'PLX_ERROR'
    rvz_error = 'RVZ_ERROR'
else:
    Simbad.add_votable_fields('mesFe_h')
    simbad_teff = 'mesfe_h.teff'
    simbad_logg = 'mesfe_h.log_g'
    simbad_fe_h = 'mesfe_h.fe_h'
    simbad_fe_h_ref = 'mesfe_h.bibcode'
    Simbad.add_votable_fields('B'); simbad_b = 'B'
    Simbad.add_votable_fields('V'); simbad_v = 'V'
    Simbad.add_votable_fields('G'); simbad_g = 'G'
    Simbad.add_votable_fields('R'); simbad_r = 'R'
    plx_error = 'plx_err'
    rvz_error = 'rvz_err'

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

def velocity_shift(velocity_in_kms, wavelength_array):
    """
    Applies a Doppler shift to a wavelength array based on the provided radial velocity. The Doppler effect
    shifts the wavelengths of emitted light depending on the relative motion of the source towards or away from the observer.

    Parameters:
        velocity_in_kms (float): The radial velocity in kilometers per second. A positive value indicates
                                 that the object is moving away from the observer, resulting in a redshift;
                                 a negative value indicates motion towards the observer, resulting in a blueshift.
        wavelength_array (array): The array of wavelengths to be shifted. Each wavelength in the array is
                                  adjusted according to the Doppler formula to reflect the effects of the
                                  radial velocity.

    Returns:
        array: A new array containing the shifted wavelengths. The shift is calculated using the formula:
               shifted_wavelength = original_wavelength / (1 + (velocity_in_kms / c))
               where 'c' is the speed of light in km/s (approximately 299,792.458 km/s).
    """
    return(wavelength_array / (1.+velocity_in_kms/299792.458))

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
        if 'DETA3X' in fits_file[0].header:
            readout_mode = '4Amp'
            metadata['READOUT'] = '4Amp'
        else:
            readout_mode = '2Amp'
            metadata['READOUT'] = '2Amp'

    return(full_image, metadata)

def identify_calibration_and_science_runs(date, raw_data_dir, each_science_run_separately = False):
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
    
    print('\n=============================================')
    print('\nIdentifying calibration and science runs now\n')

    raw_file_path = raw_data_dir+'/'+date+'/'

    log_file_path = glob.glob(raw_file_path+'*.log')
    
    raw_file_path = f"{raw_data_dir}/{date}/"

    log_file_path = glob.glob(f"{raw_file_path}*.log")
    if not log_file_path:
        raise ValueError('No Log file present in '+f"{raw_file_path}*.log")
    elif len(log_file_path) > 1:
        print(f'More than 1 Log file present, continuing with {log_file_path[0]}\n')
    else:
        print(f'Found Log file {log_file_path[0]}\n')
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
        elif (('CRAP' in line) | ('crap' in line) | ('Crap' in line)):
            pass
        else:

            ccd = line[6]
            run_object = line[8:25].strip()
            utc = line[25:33].strip()
            exposure_time = line[35:42].strip()
            # snr_noise = line[42:48].strip()
            # snr_photons = line[48:53].strip()
            # seeing = line[55:59].strip()
            # lc_status = line[60:62].strip()
            # thxe_status = line[63:67].strip()
            # read_noise = line[70:85].strip()
            # airmass = line[87:91].strip()
            overscan = line[97:].split()[0]
            comments = line[98+len(overscan):]
            if len(comments) != 0:
                if run_object != 'FlatField-Quartz':
                    print('Warning for '+run_object+' (run '+run+'): '+comments)

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
                elif run_object in [
                    "10144","14228","37795","47670","50013","56139","89080","91465","93030","98718","105435","105937","106490","108248","108249","108483",
                    "109026","109668","110879","118716","120324","121263","121743","121790","122451","125238","127972","129116","132058","134481","136298",
                    "136504","138690","139365","142669","143018","143118","143275","144470","157246","158094","158427","158926","160578","165024","169022",
                    "175191","209952"
                    ]:
                    calibration_runs['Bstar'][utc] = [run_object, run, utc]
                else:
                    if each_science_run_separately:
                        science_runs[run_object+'_'+str(run)] = [run]
                    else:
                        if run_object in science_runs.keys():
                            science_runs[run_object].append(run)
                        else:
                            science_runs[run_object] = [run]

    # Print all DarkFrame exposures, if any
    dark_frames = [key for key in calibration_runs['Darks'].keys()]
    if len(dark_frames) > 0:
        print('\nDarkFrame observations: '+', '.join(dark_frames))
    else:
        print('\nNo DarkFrame observations identified.')
                        
    if len(calibration_runs['Bstar']) > 0:
        print('\nBstar observations happened at: '+', '.join(calibration_runs['Bstar'].keys()))
    else:
        print('\nNo Bstar observations identified.')

    print('\nThe following science observations were identified: '+', '.join(list(science_runs.keys())))

    if len(science_runs) > 0:
        directory_path = Path(config.working_directory+'reduced_data/'+config.date)
        directory_path.mkdir(parents=True, exist_ok=True)
        print('\nWill save reduced data into directory '+str(directory_path))

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

    Reference pixels (DYO) are defined for Verde and Rosso, and assumed to be 2450 for Azzurro.

    Returns:
        dict: A dictionary containing wavelength solution coefficients for each CCD and spectral order.
              The keys are formatted as 'ccd_{ccd_number}_order_{order_number}' and the values are arrays
              of coefficients, with the last array entry being DY0 if present, otherwise 2450 as default.
    """
    
    wavelength_solution_coefficients_tinney = dict()

    for ccd in [1,2,3]:
        if ccd == 1: vdarc_file = Path(__file__).resolve().parent / 'veloce_reference_data' / 'vdarc_azzurro_230915.txt'
        if ccd == 2: vdarc_file = Path(__file__).resolve().parent / 'veloce_reference_data' / 'vdarc_verde_230920.txt'
        if ccd == 3: vdarc_file = Path(__file__).resolve().parent / 'veloce_reference_data' / 'vdarc_rosso_230919.txt'

        with open(vdarc_file, "r") as vdarc:
            vdarc_text = vdarc.read().split('\n')

        for line in vdarc_text:
            if 'START' in line:
                order = line[6:]
                has_DYO = False
            elif 'COEFFS' in line:
                coeffs = np.array(line[7:].split(' '),dtype=float)
            elif 'DY0' in line:
                coeffs = np.concatenate((coeffs, [int(line[4:])]))
                has_DYO = True
            elif 'STOP' in line:
                if not has_DYO:
                    wavelength_solution_coefficients_tinney['ccd_'+str(ccd)+'_order_'+order] = np.concatenate((coeffs, [2450]))
                else:
                    wavelength_solution_coefficients_tinney['ccd_'+str(ccd)+'_order_'+order] = coeffs

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
        # Assume the format is target_run and split to get the target and the run identifier
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

    if len(repeated_observations) == 0:
        print('No repeated observations found.')
    else:
        print('Repeat observations found for:\n'+','.join(list(repeated_observations.keys())))

        # Loop through each repeated observation
        for repeated_observation in repeated_observations.keys():
            print('\nMonitoring RV for '+repeated_observation)
            
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
                    print('\nCould not read '+repeated_observation+'_'+run)
                    print('Expected path was: '+expected_path)

            utmjd = np.array(utmjd)
            vrad = np.array(vrad)
            e_vrad = np.array(e_vrad)
            
            if len(vrad) > 1:
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
                ax.axhline(np.mean(vrad),c = 'C3',lw=2,ls='dashed',label = r'$\leftangle v_\mathrm{rad} \rightangle = '+"{:.2f}".format(np.round(np.mean(vrad),2))+r' \pm '+"{:.2f}".format(np.round(np.std(vrad),2))+r'\,\mathrm{km\,s^{-1}}$')
                ax.axhline(np.mean(vrad)-np.std(vrad),c = 'C1',lw=1,ls='dashed')
                ax.axhline(np.mean(vrad)+np.std(vrad),c = 'C1',lw=1,ls='dashed')
                ax.legend()
                plt.savefig(config.working_directory+'/reduced_data/'+date+'/'+repeated_observation+'_vrad_monitoring.pdf')
                if 'ipykernel' in sys.modules:
                    plt.show()
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

    # Let's identify what catalogue the object_id is from likely
    # We test for HIP and Gaia DR3, otherwise we crossmatch via RA/Dec.
    if object_id[:3] == 'HIP':
        print('\n  --> Identified '+object_id+' as a HIP catalogue entry. Crossmatching accordingly.')
        simbad_match = Simbad.query_object(object_id)        
    elif object_id.isdigit() and len(object_id) > 10:
        print('\n  --> Identified '+object_id+' as a likely Gaia DR3 source_id. Crossmatching accordingly.')
        simbad_match = Simbad.query_object('Gaia DR3 '+object_id)
    elif '_' in object_id:
        print('\n  --> '+object_id+' looks like a non-catalogue entry. Crossmatching via Ra/Dec within 2 arcsec.')
        object_coordinate = SkyCoord(ra = float(ra), dec = float(dec), frame='icrs', unit='deg')
        simbad_match = Simbad.query_region(object_coordinate, radius=2*u.arcsec)
    else:
        print('\n  --> '+object_id+' does not look like either Gaia DR3 or HIP entry. Crossmatching with Simbad first and otherwise via Ra/Dec.')
        try:
            simbad_match = Simbad.query_object(object_id)
        except:
            print('  --> '+object_id+' not a valid Simbad entry. Crossmatching via Ra/Dec within 2 arcsec.')
            object_coordinate = SkyCoord(ra = float(ra), dec = float(dec), frame='icrs', unit='deg')
            simbad_match = Simbad.query_region(object_coordinate, radius=2*u.arcsec)

    # Let's check how many matches we got and return if there are none:
    if len(simbad_match) == 0:
        print('  --> Did not find a match in Simbad.')
        return(fits_header)
    elif len(simbad_match) == 1:
        simbad_match = simbad_match[0]
    else:
        print('  --> Found more than one entry in Simbad. Using the first match.')
        simbad_match = simbad_match[0]

    # Veloce is meant to observe only down to 12th magnitude.
    # Let's test if the object is bright enough for Veloce (G < 12 mag or V < 12 mag) and print a warning if not.
    if simbad_g in simbad_match.keys():
        if abs(simbad_match[simbad_g])>=0.0:
            if simbad_match[simbad_g] > 12:
                print('  --> Warnging: Match fainter than G > 12 mag. Right match for a Veloce observations?')
        elif simbad_v in simbad_match.keys():
            if simbad_match[simbad_v] > 12:
                print('  --> Warnging: Match fainter than V > 12 mag. Right match for a Veloce observations?')
    elif simbad_v in simbad_match.keys():
        if simbad_match[simbad_v] > 12:
            print('  --> Warnging: Match fainter than V > 12 mag. Right match for a Veloce observations?')

    # Let's add some more information from the crossmatches with HIP/2MASS/Gaia DR3 and other literature where available
    ids = simbad_match['ids']
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
        fits_header['SOURCE_ID'] = (int(source_id[9:]), 'Gaia DR3 source_id (Gaia DR3 removed)')
    print('  --> Matches in HIP/2MASS/Gaia DR3: '+', '.join(match_with_hip_tmass_gaia))

    # Now let's add some literature information
    # Add literature information on radial velocity
    if 'rvz_radvel' in simbad_match.keys():
        fits_header['VRAD_LIT'] = (simbad_match['rvz_radvel'], 'Radial velocity from Simbad')
    if rvz_error in simbad_match.keys():
        fits_header['HIERARCH E_VRAD_LIT'] = (simbad_match[rvz_error], 'Radial velocity error from Simbad')
    if 'rvz_bibcode' in simbad_match.keys():
        fits_header['VRAD_BIB'] = (simbad_match['rvz_bibcode'], 'Bibcode of Simbad VRAD')

    if np.all(['VRAD_LIT' in fits_header.keys(),'E_VRAD_LIT' in fits_header.keys()]):
        print('  --> Found literature VRAD/E_VRAD in Simbad: '+str(fits_header['VRAD_LIT'])+' +/- '+str(fits_header['E_VRAD_LIT'])+' km/s by '+str(fits_header['VRAD_BIB']))
    elif 'VRAD_LIT' in fits_header.keys():
        print('  --> Found literature VRAD in Simbad: '+str(fits_header['VRAD_LIT'])+' km/s by '+str(fits_header['VRAD_BIB']))

    # Add literature information on stellar parameters Teff/logg/[Fe/H]
    if simbad_teff in simbad_match.keys():
        fits_header['TEFF_LIT'] = (simbad_match[simbad_teff], 'Effective temperature from Simbad')
    if simbad_logg in simbad_match.keys():
        fits_header['LOGG_LIT'] = (simbad_match[simbad_logg], 'Surface gravity from Simbad')
    if simbad_fe_h in simbad_match.keys():
        fits_header['FE_H_LIT'] = (simbad_match[simbad_fe_h], 'Iron abundance from Simbad')
    if simbad_fe_h_ref in simbad_match.keys():
        fits_header['TLF_BIB'] = (simbad_match[simbad_fe_h_ref], 'Bibcode of Simbad TEFF/LOGG/FE_H')

    if np.all(['TEFF_LIT' in fits_header.keys(),'TEFF_LIT' in fits_header.keys(),'LOGG_LIT' in fits_header.keys(),'FE_H_LIT' in fits_header.keys()]):
        print('  --> Found literature TEFF/LOGG/FE_H in Simbad: '+str(fits_header['TEFF_LIT'])+'/'+str(fits_header['LOGG_LIT'])+'/'+str(fits_header['FE_H_LIT'])+' by '+str(fits_header['TLF_BIB']))
    elif 'FE_H_LIT' in fits_header.keys():
        print('  --> Found literature FE_H in Simbad: '+str(fits_header['FE_H_LIT'])+' by '+str(fits_header['TLF_BIB']))

    # Add information on B/V/G/R filters (where available) and parallax
    for bvgr_filter in [simbad_b,simbad_v,simbad_g,simbad_r]:
        if abs(simbad_match[bvgr_filter]) >= 0.0:
            try:
                fits_header[bvgr_filter+'MAG'] = (simbad_match[bvgr_filter], 'Mag in '+bvgr_filter+' ('+simbad_match['FLUX_BIBCODE_'+bvgr_filter]+')')
            except:
                fits_header[bvgr_filter+'MAG'] = (simbad_match[bvgr_filter], 'Mag in '+bvgr_filter)

    fits_header['PLX'] = (simbad_match['plx_value'], 'Parallax in mas ('+simbad_match['plx_bibcode']+')')
    fits_header['E_PLX'] = (simbad_match[plx_error], 'Parallax error in mas ('+simbad_match['plx_bibcode']+')')

    return(fits_header)

def find_best_radial_velocity_from_fits_header(fits_header):
    """
    Find the best radial velocity measurement from literature and Veloce measurements.

    Parameters:
        fits_header (astropy.io.fits.header.Header): The FITS header of the 0th extension of Veloce spectra containing the radial velocity information.

    Returns:
        float: The best radial velocity measurement for calibration.
    """

    # Check which VRAD measurement has a smaller uncertainty and use that one
    if 'E_VRAD' in fits_header and 'E_VRAD_LIT' in fits_header:
        print('  --> Found literature VRAD in Simbad: '+str(fits_header['VRAD_LIT'])+' +/- '+str(fits_header['E_VRAD_LIT'])+' km/s by '+str(fits_header['VRAD_BIB']))
        print('  --> Found VRAD from Veloce measurements: '+str(fits_header['VRAD'])+' +/- '+str(fits_header['E_VRAD'])+' km/s')
        if fits_header['E_VRAD'] < fits_header['E_VRAD_LIT']:
            vrad_for_calibration = fits_header['VRAD']
            print('  --> Using VRAD from Veloce measurements because it has a smaller uncertainty.')
        else:
            vrad_for_calibration = fits_header['VRAD_LIT']
            print('  --> Using literature VRAD from Simbad because it has a smaller uncertainty.')
    elif 'E_VRAD' in fits_header:
        if 'VRAD_LIT' in fits_header:
            print('  --> Found literature VRAD in Simbad: '+str(fits_header['VRAD_LIT'])+' by '+str(fits_header['VRAD_BIB'])+' but without uncertainty.')
        print('  --> Found VRAD from Veloce measurements: '+str(fits_header['VRAD'])+' +/- '+str(fits_header['E_VRAD'])+' km/s')
        vrad_for_calibration = fits_header['VRAD']
        print('  --> Using VRAD from Veloce measurements because no literature VRAD with uncertainty is available.')
    elif 'VRAD_LIT' in fits_header:
        if 'E_VRAD_LIT' in fits_header:
            print('  --> Found literature VRAD in Simbad: '+str(fits_header['VRAD_LIT'])+' +/- '+str(fits_header['E_VRAD_LIT'])+' km/s by '+str(fits_header['VRAD_BIB']))
        else:
            print('  --> Found literature VRAD in Simbad: '+str(fits_header['VRAD_LIT'])+' by '+str(fits_header['VRAD_BIB'])+' but without uncertainty.')
        vrad_for_calibration = fits_header['VRAD_LIT']
        print('  --> Using literature VRAD from Simbad because no Veloce VRAD is available.')
    else:
        raise ValueError('No valid option for VRAD avaialble. Aborting calibration via synthesis.')

    return(vrad_for_calibration)

def find_closest_korg_spectrum(available_korg_spectra,fits_header):
    """
    Find the closest Korg spectrum based on the object's properties and literature values.

    Parameters:
        available_korg_spectra (dict): A dictionary containing the available Korg spectra with their names as keys.
        fits_header (astropy.io.fits.header.Header): The FITS header of the 0th extension of Veloce spectra containing the object information.

    Returns:
        str: The name of the closest Korg spectrum to be used for calibration.
    """

    print('  -> Available Korg Spectra: '+', '.join(list(available_korg_spectra.keys())[2:]))

    # If the star is radial velocity standard 18 Sco, let's use the 18 Sco spectrum.
    if fits_header['OBJECT'] == 'HIP79672':
        closest_korg_spectrum = '18sco'
        print('  --> Object is 18Sco. Using 18Sco spectrum.')
    # Let's check if we have a literature TEFF/LOGG/FE_H available
    elif 'TEFF_LIT' in fits_header.keys() and 'LOGG_LIT' in fits_header.keys() and 'FE_H_LIT' in fits_header.keys():
        print('  --> Literature values for TEFF/LOGG/FE_H are: '+str(fits_header['TEFF_LIT'])+'/'+str(fits_header['LOGG_LIT'])+'/'+str(fits_header['FE_H_LIT']))
        # If the star is a giant with TEFF < 5000 and LOGG < 3.5, use 'arcturus'
        if fits_header['TEFF_LIT'] < 5000 and fits_header['LOGG_LIT'] < 3.5:
            closest_korg_spectrum = 'arcturus'
            print('  --> Object is a giant. Using Arcturus spectrum.')
        elif fits_header['TEFF_LIT'] < 5000 and fits_header['LOGG_LIT'] >= 3.5:
            closest_korg_spectrum = '61cyga'
            print('  --> Object is a cool dwarf. Using 61 Cyg A spectrum.')
        elif fits_header['FE_H_LIT'] < -0.25:
            closest_korg_spectrum = 'hd22879'
            print('  --> Object is metal-poor dwarf. Using HD22879 spectrum.')
        else:
            print('  --> Object is a dwarf. Using Sun spectrum.')

    # Let's use the Sun, if there is no information on TEFF/LOGG/FE_H available.
    elif 'FE_H_LIT' in fits_header.keys():
        if fits_header['FE_H_LIT'] < -0.25:
            closest_korg_spectrum = 'hd22879'
            print('  --> Object is metal-poor. Using HD22879 spectrum.')
        else:
            closest_korg_spectrum = 'sun'
            print('  --> Object has solar [Fe/H]. Using Sun spectrum.')

    # If we have no stellar parameters, let's use the absolute magnitude.
    elif 'PLX' in fits_header.keys():
        if fits_header['PLX'] > 0:
            if 'G' in fits_header.keys():
                absolute_mag = fits_header['G'] + 5 * np.log10(fits_header['PLX']/10)
                print('  --> Object has no stellar parameters measured, but absolute M_G is '+"{:.1f}".format(absolute_mag))
            elif 'V' in fits_header.keys():
                absolute_mag = fits_header['V'] + 5 * np.log10(fits_header['PLX']/10)
                print('  --> Object has no stellar parameters measured, but absolute M_V is '+"{:.1f}".format(absolute_mag))
            else:
                print('  --> Object has no stellar parameters measured, nor absolute magnitude (although parallax available). Using Sun spectrum by default.')
                closest_korg_spectrum = 'sun'
                return(closest_korg_spectrum)

            # Let's try to estimate a color. Assume color = 0.5 if none available
            if 'B' in fits_header.keys() and 'R' in fits_header.keys():
                color = fits_header['B'] - fits_header['R']
            elif 'V' in fits_header.keys() and 'R' in fits_header.keys():
                color = fits_header['V'] - fits_header['R']
            else:
                color = 0.5
            
            if absolute_mag < 8 and color > 1:
                closest_korg_spectrum = 'arcturus'
                print('  --> Object is likely giant based on magnitude (< 8 mag) and color  (> 1 mag). Using Arcturus spectrum.')
            elif absolute_mag >= 8 and color > 1:
                closest_korg_spectrum = '61cyga'
                print('  --> Object is likely cool dwarf based on magnitude (> 8 mag) and color (> 1 mag). Using 61 Cyg A spectrum.')
            else:
                closest_korg_spectrum = 'sun'
                print('  --> Object is likely dwarf based on color. Using Sun spectrum.')
        else:
            closest_korg_spectrum = 'sun'
            print('  --> Object has no stellar parameters measured, nor absolute magnitude (parallax measurement is negative). Using Sun spectrum by default.')
    else:
        closest_korg_spectrum = 'sun'
        print('  --> No TEFF/LOGG/FE_H nor absolute magnitude values available. Using Sun by default.')

    return(closest_korg_spectrum)