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

def voigt_absorption_profile(wavelength, line_centre, line_depth, sigma, gamma):
    """
    Returns the Voigt line shape at wavelengths `wavelength` for an absorption line with a continuum.

    Parameters:
        wavelength : array-like
            Wavelength array over which to compute the Voigt profile.
        line_centre : float
            Central wavelength of the absorption line.
        line_depth : float
            The depth of the absorption relative to the continuum.
        sigma : float
            Gaussian standard deviation.
        gamma : float
            Lorentzian half-width at half-maximum (HWHM).
    """
    z = ((wavelength - line_centre) + 1j*gamma) / (sigma*np.sqrt(2))
    profile = wofz(z).real
    profile = profile / max(profile)  # Normalize profile to max 1
    return 1 - line_depth * profile

def fit_voigt_absorption_profile(wavelength, flux, initial_guess):
    """
    Fit a Gaussian to a spectrum.

    :param wavelength: Wavelength array.
    :param flux: Flux array.
    :param initial_guess: Initial guess for the Gaussian fit.

    :return: Fitted Gaussian parameters.
    """

    # Fit a Gaussian to the spectrum
    popt, pcov = curve_fit(voigt_absorption_profile, wavelength, flux, p0=initial_guess)
    return (popt, pcov)

def gaussian_absorption_profile(wavelength, line_centre, line_depth, line_sigma):
    """
    Gaussian function.

    :param wavelength: Wavelength array.
    :param line_centre: Centre of the Gaussian line.
    :param line_sigma: Sigma of the Gaussian line.
    :param line_depth: Depth of the Gaussian line.

    :return: Gaussian line.
    """
    return 1 - line_depth * np.exp(-0.5 * ((wavelength - line_centre) / line_sigma) ** 2)

def fit_gaussian_absorption_profile(wavelength, flux, initial_guess):
    """
    Fit a Gaussian to a spectrum.

    :param wavelength: Wavelength array.
    :param flux: Flux array.
    :param initial_guess: Initial guess for the Gaussian fit.

    :return: Fitted Gaussian parameters.
    """

    # Fit a Gaussian to the spectrum
    popt, pcov = curve_fit(gaussian_absorption_profile, wavelength, flux, p0=initial_guess)
    return (popt, pcov)

def calculate_barycentric_velocity_correction(science_header):
    """
    Calculate the barycentric velocity correction for a given observation (Ra, Dec, UT MJD) at Siding Spring Observatory (SSO).

    :param science_header: Header of the science FITS file.

    :return: Barycentric velocity correction in km/s.
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
    Shift wavelength array with rv_value in km/s.
    :param velocity_in_kms: Radial velocity in km/s.
    :param wavelength_array: Wavelength array to be shifted.
    :return: Shifted wavelength array.
    """
    return(wavelength_array / (1.+velocity_in_kms/299792.458))

def match_month_to_date(date):
    """
    Match the month to the date.

    :param date: Date in the format 'YYYYMMDD'.

    :return: Month in string format.
    """
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    
    return(months[int(date[2:-2])-1])

def polynomial_function(x, *coeffs):
    """
    Polynomial function.

    :param x: Independent variable (wavelength or pixel number).
    :param coeffs: Coefficients of the polynomial.

    :return: Calculated y values.
    """
    y = np.zeros_like(x,dtype=float)
    for i, coeff in enumerate(coeffs):
        y += coeff * x**i
    return y

def read_veloce_fits_image_and_metadata(file_path):
    """
    Read the relevant information from a FITS file.

    :param file_path: Path to the FITS file.
    
    :return: Full image and metadata.
    """

    # Read relevant information from FITS file
    metadata = dict()
    
    fits_file = fits.open(file_path)

    full_image = fits_file[0].data
    for key in ['OBJECT','UTMJD','MEANRA','MEANDEC','EXPTIME']:
        metadata[key] = fits_file[0].header[key]
    if 'DETA3X' in fits_file[0].header:
        readout_mode = '4Amp'
        metadata['READOUT'] = '4Amp'
    else:
        readout_mode = '2Amp'
        metadata['READOUT'] = '2Amp'
    fits_file.close()

    return(full_image, metadata)

def identify_calibration_and_science_runs(date, raw_data_dir):
    """
    Identify calibration and science runs from the log file.

    :param date: Date in the format 'YYYYMMDD'.
    :param raw_data_dir: Path to the raw data directory.
    
    :return: Dictionaries with calibration and science runs.
    """
    
    print('\n=============================================')
    print('\nIdentifying calibration and science runs now\n')

    raw_file_path = raw_data_dir+'/'+date+'/'

    log_file_path = glob.glob(raw_file_path+'*.log')
    if len(log_file_path) == 0:
        raise ValueError('No Log file present')
    else:
        if len(log_file_path) > 1:
            print('More than 1 Log file present, continuing with '+log_file_path[0]+'\n')
        else:
            print('Found Log file '+log_file_path[0]+'\n')
        log_file_path = log_file_path[0]

        log_file = open(log_file_path, "r")
        log_file_text = log_file.read()
        log_file.close()
        log_file_text = log_file_text.split('\n')

    # Now go through the log_file_text and read out all important information

    # Collect information about runs from log file.
    # We classify calibration_runs and science_runs
    calibration_runs = dict()
    calibration_runs['FibTh_15.0'] = []
    calibration_runs['FibTh_60.0'] = []
    calibration_runs['FibTh_180.0'] = []
    calibration_runs['SimTh_15.0'] = []
    calibration_runs['SimTh_60.0'] = []
    calibration_runs['SimTh_180.0'] = []
    calibration_runs['SimLC'] = []
    calibration_runs['Flat_0.1'] = []
    calibration_runs['Flat_1.0'] = []
    calibration_runs['Flat_10.0'] = []
    calibration_runs['Flat_60.0'] = []
    calibration_runs['Bstar'] = []
    # 'Dark' to be added depending on exposure times

    science_runs = dict()

    for line in log_file_text:
        # split line to read out specific information
        line_split = line.split(' ')

        # Identify runs via their numeric value
        run = line[:4]
        if not run.isnumeric():
            pass
        else:

            ccd = line[6]
            run_object = line[8:25].strip()
            utc = line[25:33].strip()
            exposure_time = line[35:42].strip()
            snr_noise = line[42:48].strip()
            snr_photons = line[48:53].strip()
            seeing = line[55:59].strip()
            lc_status = line[60:62].strip()
            thxe_status = line[63:67].strip()
            read_noise = line[70:85].strip()
            airmass = line[87:91].strip()
            overscan = line[97:].split()[0]
            comments = line[98+len(overscan):]
            if len(comments) != 0:
                if run_object != 'FlatField-Quartz':
                    print('Warning for '+run_object+' (run '+run+'): '+comments)

            # Read in type of observation from CCD3 info (since Rosso should always be available)
            if ccd == '3':
                if run_object == 'SimLC':
                    calibration_runs['SimLC'].append(run)
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
                elif run_object in ['56139','105435','127972']:
                    calibration_runs['Bstar'].append(run)            
                else:
                    if run_object in science_runs.keys():
                        science_runs[run_object].append(run)
                    else:
                        science_runs[run_object] = [run]
                        
    print('\nThe following science observations were identified: '+', '.join(list(science_runs.keys())))

    return(calibration_runs, science_runs)

def interpolate_spectrum(wavelength, flux, target_wavelength):
    """
    Interpolate a spectrum to a target wavelength array.

    :param wavelength: Wavelength array of the input spectrum.
    :param flux: Flux array of the input spectrum.
    :param target_wavelength: Target wavelength array.
    
    :return: Interpolated flux array
    """

    interpolation_function = interp1d(wavelength, flux, bounds_error=False, fill_value=(1.0,1.0), kind='cubic')
    return interpolation_function(target_wavelength)

