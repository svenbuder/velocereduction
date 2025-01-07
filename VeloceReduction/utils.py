from . import config

import numpy as np
import glob
from scipy.special import wofz
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
SSO = EarthLocation.of_site('Siding Spring Observatory')

def radial_velocity_from_line_shift(line_centre_observed, line_centre_rest):
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

def calculate_barycentric_velocity_correction(science_header):
    """
    Calculates the barycentric velocity correction for a given astronomical observation by taking into account
    the Earth's motion relative to the solar system's barycenter. This correction is computed based on the
    right ascension (RA) and declination (Dec) of the observed object, as well as the Universal Time (UT)
    expressed in Modified Julian Date (MJD), as specified in the header of a FITS file.

    This function uses astropy.coordinates.SkyCoord for celestial coordinate handling and astropy.time.Time
    for time format conversions, ensuring precise astronomical calculations. The Siding Spring Observatory (SSO)
    location should be predefined as an astropy EarthLocation object within the function.

    Parameters:
        science_header (dict): Header from a science FITS file that must include:
            - 'MEANRA': Mean right ascension of the observation in degrees.
            - 'MEANDEC': Mean declination of the observation in degrees.
            - 'UTMJD': Universal Time of the observation in Modified Julian Date format.

    Returns:
        float: The barycentric velocity correction in kilometers per second (km/s). This value represents the velocity
               necessary to adjust for the Earth's motion when analyzing spectral data, improving the accuracy of radial
               velocity measurements.
    """

    object_coordinates = SkyCoord(ra = science_header['MEANRA'], dec = science_header['MEANDEC'], frame="icrs", unit="deg")
    vbary_corr_kms = object_coordinates.radial_velocity_correction( 
        kind='barycentric', 
        obstime = Time(val=science_header['UTMJD'],format='mjd', scale='utc'),
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
        full_image = fits_file[0].data
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
        raise ValueError('No Log file present')
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
        'Flat_0.1', 'Flat_1.0', 'Flat_10.0', 'Flat_60.0', 'Bstar', 'Bias'
    ]}
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
            # utc = line[25:33].strip()
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
                    if 'Dark_'+exposure_time in calibration_runs.keys():
                        calibration_runs['Dark_'+exposure_time].append(run)
                    else:
                        calibration_runs['Dark_'+exposure_time] = [run]
                elif run_object in [
                    "10144","14228","37795","47670","50013","56139","89080","91465","93030","98718","105435","105937","106490","108248","108249","108483",
                    "109026","109668","110879","118716","120324","121263","121743","121790","122451","125238","127972","129116","132058","134481","136298",
                    "136504","138690","139365","142669","143018","143118","143275","144470","157246","158094","158427","158926","160578","165024","169022",
                    "175191","209952"
                    ]:
                    calibration_runs['Bstar'].append([run, run_object])
                else:
                    if each_science_run_separately:
                        science_runs[run_object+'_'+str(run)] = [run]
                    else:
                        if run_object in science_runs.keys():
                            science_runs[run_object].append(run)
                        else:
                            science_runs[run_object] = [run]
                        
    if len(calibration_runs['Bstar']) > 0:
        print('\nThe following Bstar observations were identified: '+', '.join(list(np.array(calibration_runs['Bstar'])[:,1])))
    else:
        print('\nNo Bstar observations were identified.')
    print('\nThe following science observations were identified: '+', '.join(list(science_runs.keys())))

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
        if ccd == 1: vdarc_file = './VeloceReduction/veloce_reference_data/vdarc_azzurro_230915.txt'
        if ccd == 2: vdarc_file = './VeloceReduction/veloce_reference_data/vdarc_verde_230920.txt'
        if ccd == 3: vdarc_file = './VeloceReduction/veloce_reference_data/vdarc_rosso_230919.txt'

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

    # Loop through each repeated observation
    for repeated_observation in repeated_observations.keys():
        print('\nMonitoring RV for '+repeated_observation)
        
        # We will read out UTMJD, VRAD, and E_VRAD
        utmjd = []
        vrad = []
        e_vrad = []
        
        for run in repeated_observations[repeated_observation]:
            try:
                with fits.open('./reduced_data/'+date+'/'+repeated_observation+'_'+run+'/veloce_spectra_'+repeated_observation+'_'+run+'_'+date+'.fits') as file:
                    utmjd.append(file[0].header['UTMJD'])
                    vrad.append(file[0].header['VRAD'])
                    e_vrad.append(file[0].header['E_VRAD'])
            except:
                print('\nCould not read '+repeated_observation+'_'+run)
                print('Expected path was: reduced_data/'+date+'/'+repeated_observation+'_'+run+'/veloce_spectra_'+repeated_observation+'_'+run+'_'+date+'.fits')

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
            ax.axhline(np.mean(vrad),c = 'C3',lw=2,ls='dashed',label = r'$\leftangle v_\mathrm{rad} \rightangle = '+"{:.2f}".format(np.round(np.mean(vrad),2))+' \pm '+"{:.2f}".format(np.round(np.std(vrad),2))+r'\,\mathrm{km\,s^{-1}}$')
            ax.axhline(np.mean(vrad)-np.std(vrad),c = 'C1',lw=1,ls='dashed')
            ax.axhline(np.mean(vrad)+np.std(vrad),c = 'C1',lw=1,ls='dashed')
            ax.legend()
            plt.savefig('./reduced_data/'+date+'/'+repeated_observation+'_vrad_monitoring.pdf')
            plt.show()
            plt.close()
        else:
            print('Less than two observations could be read in for '+repeated_observation)
            print('Skipping plotting for '+repeated_observation)