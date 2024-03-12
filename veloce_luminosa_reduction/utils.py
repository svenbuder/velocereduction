from . import config

import numpy as np
import glob
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from astropy.io import fits

def match_month_to_date(date):
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    
    return(months[int(date[2:-2])-1])

def read_veloce_fits_image_and_metadata(file_path):

    # Read relevant information from FITS file
    metadata = dict()
    
    fits_file = fits.open(file_path)

    full_image = fits_file[0].data
    for key in ['UTMJD','MEANRA','MEANDEC','EXPTIME']:
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
                        
    return(calibration_runs, science_runs)

def interpolate_spectrum(wavelength, flux, target_wavelength):
    """Interpolate the spectrum to a new wavelength grid."""
    interpolation_function = interp1d(wavelength, flux, bounds_error=False, fill_value="extrapolate")
    return interpolation_function(target_wavelength)


def wavelength_to_rgb(wavelength, gamma=1.0):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A = 1.
    if wavelength < 380:
        wavelength = 380.
#     if wavelength >750:
#         wavelength = 751.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    elif wavelength >= 750 and wavelength <= 800:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B,A)

def create_rainbow_colormap():
    clim=(350,780)
    cmap_norm = plt.Normalize(*clim)
    wl = np.arange(clim[0],clim[1]+1,1)
    colorlist = list(zip(cmap_norm(wl),[wavelength_to_rgb(w) for w in wl]))
    colormap = LinearSegmentedColormap.from_list("spectrum", colorlist)
    return(colormap)

def create_transparent_greyscale_colormap():
    colors = [(0, 0, 0, 1), (0.5, 0.5, 0.5, 0)]  # RGBA colors from transparent to black
    cmap_name = 'transparent'
    colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    return(colormap)
