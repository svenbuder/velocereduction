import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter

from . import config
from .utils import read_veloce_fits_image_and_metadata, match_month_to_date, polynomial_function, calculate_barycentric_velocity_correction, phase_correlation_shift

def substract_overscan(full_image, metadata, debug_overscan = False):
    """
    Subtracts the overscan from a given full astronomical image to correct for the CCD readout bias. This function 
    utilizes metadata to identify the overscan region and calculates its median value and RMS (Root Mean Square) 
    to adjust the image data accordingly. The corrected image is then trimmed to remove the overscan regions.

    If the debug_overscan flag is set to True, debug plots showing the overscan region, the calculated median overscan,
    and its effect on the image before and after subtraction will be displayed for visual inspection.

    Parameters:
        full_image (ndarray):   A 2D numpy array representing the full CCD image including overscan regions.
        metadata (dict):        A dictionary containing metadata of the image, which should include keys for overscan 
                                region coordinates and other necessary CCD characteristics.
        debug_overscan (bool):  A boolean flag that, when set to True, enables the display of debug plots.

    Returns:
        tuple: A tuple containing:
            - trimmed_image (ndarray):  The image after overscan subtraction, with overscan regions removed.
            - median_overscan (float):  The median value of the overscan region used for the correction.
            - overscan_rms (float):     The root mean square of the overscan region, indicating noise level.
            - readout_mode (str):       The readout mode of the CCD as extracted from the metadata, indicating how the 
                                        image data was read from the sensor.
    """

    # Identify overscan region and subtract overscan while reporting median overscan and overscan root-mean-square
    overscan_median = dict()
    overscan_rms = dict()
    
    if debug_overscan:
        plt.figure(figsize=(10,10))
        s = plt.imshow(full_image, vmin=975, vmax = 1025)
        plt.colorbar(s)
        if 'ipykernel' in sys.modules: plt.show()
        plt.close()

    if metadata['READOUT'] == '4Amp':

        # We report the median overscan
        # And we calculate a robust standard deviation, i.e.,
        # half the difference between 16th and 84th percentile

        overscan_size = 32

        # Lower-left Quadrant 1: 2120: and :2112
        quadrant1 = np.array(full_image[2120+32:-32,32:2112-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[2120:, :2112][:overscan_size, :] = True  # Top edge
        overscan[2120:, :2112][:, :overscan_size] = True  # Left edge
        overscan[2120:, :2112][-overscan_size:, :] = True  # Bottom edge
        overscan[2120:, :2112][:, -overscan_size:] = True  # Right edge
        overscan = full_image[overscan]

        if debug_overscan:
            plt.figure()
            plt.hist(overscan.flatten(),bins=np.arange(975,1050),label = 'q1', histtype='step', ls='dashed')

        overscan_median['q1'] = int(np.median(overscan))
        overscan_rms['q1'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant1 = (quadrant1 - overscan_median['q1'])

        # Upper-left Quadrant 2: :2120 and :2112
        quadrant2 = np.array(full_image[32:2120-32,32:2112-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[:2120, :2112][:overscan_size, :] = True  # Top edge
        overscan[:2120, :2112][:, :overscan_size] = True  # Left edge
        overscan[:2120, :2112][-overscan_size:, :] = True  # Bottom edge
        overscan[:2120, :2112][:, -overscan_size:] = True  # Right edge
        overscan = full_image[overscan]

        if debug_overscan:
            plt.hist(overscan.flatten(),bins=np.arange(975,1050),label = 'q2', histtype='step', ls='dashed')
        
        overscan_median['q2'] = int(np.median(overscan))
        overscan_rms['q2'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant2 -= overscan_median['q2']

        # Upper-right Quadrant 3: :2120 and 2112:
        quadrant3 = np.array(full_image[32:2120-32,2112+32:-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[:2120, 2112:][:overscan_size, :] = True  # Top edge
        overscan[:2120, 2112:][:, :overscan_size] = True  # Left edge
        overscan[:2120, 2112:][-overscan_size:, :] = True  # Bottom edge
        overscan[:2120, 2112:][:, -overscan_size:] = True  # Right edge
        overscan = full_image[overscan]

        if debug_overscan:
            plt.hist(overscan.flatten(),bins=np.arange(975,1050),label = 'q3', histtype='step', ls='dashed')

        overscan_median['q3'] = int(np.median(overscan))
        overscan_rms['q3'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant3 -= overscan_median['q3']

        # Lower-right Quadrant 4: 2120: and 2112:
        quadrant4 = np.array(full_image[2120+32:-32,2112+32:-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[2120:, 2112:][:overscan_size, :] = True  # Top edge
        overscan[2120:, 2112:][:, :overscan_size] = True  # Left edge
        overscan[2120:, 2112:][-overscan_size:, :] = True  # Bottom edge
        overscan[2120:, 2112:][:, -overscan_size:] = True  # Right edge
        overscan = full_image[overscan]

        overscan_median['q4'] = int(np.median(overscan))
        overscan_rms['q4'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant4 -= overscan_median['q4']

        if debug_overscan:
            plt.hist(overscan.flatten(),bins=np.arange(975,1050),label = 'q4', histtype='step', ls='dashed')
            plt.legend()
            if 'ipykernel' in sys.modules: plt.show()
            plt.close()

        trimmed_image = np.hstack([np.vstack([quadrant2,quadrant1]),np.vstack([quadrant3,quadrant4])]).clip(min=0.0)

    if metadata['READOUT'] == '2Amp':

        overscan_size = 32

        # Quadrant 1: :2088 and :
        quadrant1 = np.array(full_image[32:-32,32:2112-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[:, :2112][:overscan_size, :] = True  # Top edge
        overscan[:, :2112][-overscan_size:, :] = True  # Bottom edge
        overscan[:, :2112][:, :overscan_size] = True  # Left edge
        overscan[:, :2112][:, -overscan_size:] = True  # Inner edge near split
        overscan = full_image[overscan]

        overscan_median['q1'] = int(np.median(overscan))
        overscan_rms['q1'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant1 -= overscan_median['q1']

        # Quadrant 2: 2088: and :2112
        quadrant2 = np.array(full_image[32:-32,2112+32:-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[:, 2112:][:overscan_size, :] = True  # Top edge
        overscan[:, 2112:][-overscan_size:, :] = True  # Bottom edge
        overscan[:, 2112:][:, :overscan_size] = True  # Inner edge near split
        overscan[:, 2112:][:, -overscan_size:] = True  # Right edge
        overscan = full_image[overscan]

        overscan_median['q2'] = int(np.median(overscan))
        overscan_rms['q2'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant2 = (quadrant2 - overscan_median['q2'])

        trimmed_image = np.hstack([quadrant1,quadrant2]).clip(min=0.0)

    if debug_overscan:
        plt.figure(figsize=(10,10))
        s = plt.imshow(trimmed_image, vmin = -5, vmax = 100)
        plt.colorbar(s)
        if 'ipykernel' in sys.modules: plt.show()
        plt.close()
        
    if debug_overscan: print('      -->',overscan_median, overscan_rms, metadata['READOUT'])
        
    return(trimmed_image, overscan_median, overscan_rms, metadata['READOUT'])

def estimate_ccd_pixel_shifts_wrt_reference(calibration_runs):
    """
    Estimate pixel shifts in x and y directions with respect to reference frames (from 001122).
    The following reference frames are used:
    - CCD1: SimTh_180.0
    - CCD2: SimTh_60.0  and SimLC
    - CCD3: SimTh_15.0  and SimLC

    Neglecting FibTh frames, because while SimTh and SimLC shifts were similar, FibTh differed.
    """
    pixel_shifts_wrt_reference = {}

    for ccd in [1,2,3]:

        pixel_shift_x = []
        pixel_shift_y = []

        if ccd == 1:
            calibrations_to_compare = [
                # ['FibTh_180.0','0047'],
                ['SimTh_180.0','0003']
            ]
        elif ccd == 2:
            calibrations_to_compare = [
                # ['FibTh_60.0','0042'],
                ['SimTh_60.0','0057'],
                ['SimLC','0159']
            ]
        elif ccd == 3:
            calibrations_to_compare = [
                # ['FibTh_15.0','0037'],
                ['SimTh_15.0','0062'],
                ['SimLC','0159']
            ]
        
        for calibration_type, reference_run in calibrations_to_compare:
            if len(calibration_runs[calibration_type]) == 0:
                print(f"No calibration runs found for {calibration_type} on CCD{ccd}. Skipping shift estimate for this type.")
            else:
                # a) Reference frame
                full_image, metadata = read_veloce_fits_image_and_metadata(
                    config.working_directory+
                    'observations/001122/ccd_'+str(ccd)+'/22nov'+str(ccd)+reference_run+'.fits'
                )
                reference_image, _, _, _ = substract_overscan(full_image, metadata, debug_overscan=False)

                # b) Median of frames of the night
                images = []
                for run in calibration_runs[calibration_type]:
                    frame_path = config.working_directory+'observations/'+config.date+'/ccd_'+str(ccd)+'/'+config.date[-2:]+match_month_to_date(config.date)+str(ccd)+run+'.fits'
                    image, metadata = read_veloce_fits_image_and_metadata(frame_path)
                    trimmed_image, _, _, _ = substract_overscan(image, metadata, debug_overscan=False)
                    images.append(trimmed_image)
                median_image = np.median(np.array(images), axis=0)

                # c) Estimate shifts
                dx, dy, error = phase_correlation_shift(reference_image, median_image)
                print(f"Shift CCD{ccd} (from {calibration_type}):", dx, dy, error)
                pixel_shift_x.append(dx)
                pixel_shift_y.append(dy)

        if len(pixel_shift_x) > 0:
            pixel_shift_x_mean = np.mean(np.array(pixel_shift_x))
            if len(pixel_shift_x) > 1:
                pixel_shift_x_std  = np.std(np.array(pixel_shift_x))
                if pixel_shift_x_std > 0.5:
                    print(f'  --> dX estimated to be {pixel_shift_x_mean} +/- {pixel_shift_x_std} pixels for CCD{ccd}. Large scatter!')
            else:
                pixel_shift_x_std = np.nan
        else:
            pixel_shift_x_mean = 0.0
            pixel_shift_x_std = np.nan
            print('  --> Could not estimate pixel shift in X for CCD{ccd}. Setting to 0.0')

        if len(pixel_shift_y) > 0:
            pixel_shift_y_mean = np.mean(np.array(pixel_shift_y))
            if len(pixel_shift_y) > 1:
                pixel_shift_y_std  = np.std(np.array(pixel_shift_y))
                if pixel_shift_y_std > 0.5:
                    print(f'  --> dY estimated to be {pixel_shift_y_mean} +/- {pixel_shift_y_std} pixels for CCD{ccd}. Large scatter!')
            else:
                pixel_shift_y_std = np.nan
        else:
            pixel_shift_y_mean = 0.0
            pixel_shift_y_std = np.nan
            print('  --> Could not estimate pixel shift in Y for CCD{ccd}. Setting to 0.0')
        pixel_shifts_wrt_reference[ccd] = (np.round(pixel_shift_x_mean,2), np.round(pixel_shift_y_mean,2))

        print(f'Shifts for CCD{ccd}: {pixel_shift_x_mean:+.2f} +/- {pixel_shift_x_std:.2f} and {pixel_shift_y_mean:+.2f} +/- {pixel_shift_y_std:.2f}')

    return(pixel_shifts_wrt_reference)

def read_in_order_tramlines_tinney():
    """
    Reads in the optimized tramline information for each spectroscopic order from C. Tinney's data files.
    The tramline information specifies the pixel locations at the beginning and end of each order on the CCDs.

    CCD Files:
        - CCD1 (Azzurro): Orders 138-167 are read from 'azzurro-th-m138-167-all.txt'
        - CCD2 (Verde): Orders 104-139 are read from 'verde-th-m104-139-all.txt'
        - CCD3 (Rosso): Orders 65-104 are read from 'rosso-th-m65-104-all.txt'

    Each order's tramline information is stored in two dictionaries:
        - order_tramline_beginnings: Contains the beginning pixel of each order.
        - order_tramline_endings: Contains the ending pixel of each order.
        
    The keys for these dictionaries are formatted as 'ccd_{ccd}_{order}', where:
        - {ccd} is the CCD identifier (1, 2, or 3).
        - {order} is the spectral order number.

    Returns:
        tuple: A tuple containing three dictionaries:
            - order_tramline_ranges: Contains the range (start and end) of tramlines for each order.
            - order_tramline_beginnings: Dictionary with starting tramline positions.
            - order_tramline_endings: Dictionary with ending tramline positions.
    """
    order_ranges = dict()
    order_beginning_coefficients = dict()
    order_ending_coefficients = dict()

    with open(Path(__file__).resolve().parent / 'veloce_reference_data' / 'azzurro-th-m138-167-all.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if cnt % 4 == 0:
                split_lines = line[:-1].split(' ')
                order = int(split_lines[0])
                order_ranges['ccd_1_order_'+ str(order)] = np.arange(int(split_lines[1]),int(split_lines[2]))
            if cnt % 4 == 1:
                order_beginning_coefficients['ccd_1_order_'+ str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
                order_beginning_coefficients['ccd_1_order_'+ str(order)][0] -= 45
                order_ending_coefficients['ccd_1_order_'+ str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
            line = fp.readline()
            cnt += 1

    with open(Path(__file__).resolve().parent / 'veloce_reference_data' / 'verde-th-m104-139-all.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if cnt % 4 == 0:
                split_lines = line[:-1].split(' ')
                order = int(split_lines[0])
                order_ranges['ccd_2_order_'+ str(order)] = np.arange(int(split_lines[1]), int(split_lines[2]))
            if cnt % 4 == 1:
                order_beginning_coefficients['ccd_2_order_'+ str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
                order_beginning_coefficients['ccd_2_order_'+ str(order)][0] -= 45
                order_ending_coefficients['ccd_2_order_'+ str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
            line = fp.readline()
            cnt += 1

    with open(Path(__file__).resolve().parent / 'veloce_reference_data' / 'rosso-th-m65-104-all.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if cnt % 4 == 0:
                split_lines = line[:-1].split(' ')
                order = int(split_lines[0])
                order_ranges['ccd_3_order_'+ str(order)] = np.arange(int(split_lines[1]), int(split_lines[2]))
            if cnt % 4 == 1:
                order_beginning_coefficients['ccd_3_order_'+ str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
                order_beginning_coefficients['ccd_3_order_'+ str(order)][0] -= 45
                order_ending_coefficients['ccd_3_order_'+ str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
            line = fp.readline()
            cnt += 1

    # We could also use the laser comb position.
    # with open('./VeloceReduction/veloce_reference_data/verde-lc-m104-135-all.txt') as fp:
    # with open('./VeloceReduction/veloce_reference_data/rosso-lc-m65-104-all.txt') as fp:
    # For now, we simply assume that the laser comb position is just slightly offset from the order_ending
    # (so that we can simply use the order_ending_coefficients).

    order_ranges_sorted = dict()
    order_beginning_coefficients_sorted = dict()
    order_ending_coefficients_sorted = dict()

    for ccd in ['1','2','3']:
        if ccd == '1': orders = np.arange(167, 138-1, -1)
        if ccd == '2': orders = np.arange(140, 103-1, -1)
        if ccd == '3': orders = np.arange(104,  65-1, -1)

        for order in orders:
            order_ranges_sorted['ccd_'+ccd+'_order_'+str(order)] = order_ranges['ccd_'+ccd+'_order_'+str(order)]
            order_beginning_coefficients_sorted['ccd_'+ccd+'_order_'+str(order)] = order_beginning_coefficients['ccd_'+ccd+'_order_'+str(order)]
            order_ending_coefficients_sorted['ccd_'+ccd+'_order_'+str(order)] = order_ending_coefficients['ccd_'+ccd+'_order_'+str(order)]

    return(order_ranges_sorted, order_beginning_coefficients_sorted, order_ending_coefficients_sorted)

def read_in_order_tramlines(use_default = False):
    """
    Reads in optimized tramline information specifying the pixel positions for the beginning and ending of each 
    spectroscopic order across three CCDs. The data is read from text files and used to populate three dictionaries 
    with pixel information for each order.

    Parameters:
        use_default (bool): Set to True to use the default tramline information and not version from the night.

    Returns:
        tuple: Contains three dictionaries:
            - order_tramline_ranges (dict): Mapping of each spectroscopic order to its full pixel range.
            - order_tramline_beginning_coefficients (dict): Mapping of each order to the beginning pixel positions of its tramlines.
            - order_tramline_ending_coefficients (dict): Mapping of each order to the ending pixel positions of its tramlines.

    Each dictionary key is formatted as 'ccd_{ccd}_{order}', where '{ccd}' is the CCD number (1, 2, or 3),
    and '{order}' is the specific order number on that CCD.

    CCD Orders Handled:
        - CCD1 handles orders 167 to 138.
        - CCD2 handles orders 140 to 103.
        - CCD3 handles orders 104 to 65.

    Each order's data is loaded from a corresponding file in the format:
    './VeloceReduction/tramline_information/tramlines_begin_end_ccd_{ccd}_order_{order}.txt'

    The function constructs three dictionaries:
        - order_tramline_ranges: Maps 'ccd_{ccd}_{order}' to a range of pixel indices (0 to 4111).
        - order_tramline_beginning_coefficients: Coefficients for 5th order polynomial for starting pixel positions for tramlines.
        - order_tramline_ending_coefficients: Coefficients for 5th order polynomial for ending pixel positions for tramlines.
    """

    order_tramline_ranges = dict()
    order_tramline_beginning_coefficients = dict()
    order_tramline_ending_coefficients = dict()

    for ccd in ['1','2','3']:
        if ccd == '1': orders = np.arange(167, 138-1, -1)
        if ccd == '2': orders = np.arange(140, 103-1, -1)
        if ccd == '3': orders = np.arange(104,  65-1, -1)

        for order in orders:
            order_tramline_ranges['ccd_'+ccd+'_order_'+str(order)] = np.arange(4112)

            if use_default:
                tramline_information = np.loadtxt(Path(__file__).resolve().parent / 'tramline_information' / f'tramlines_begin_end_ccd_{ccd}_order_{order}.txt')
            else:
                try:
                    tramline_information = np.loadtxt(config.working_directory+'/reduced_data/'+config.date+f'/_tramline_information/tramlines_begin_end_ccd_{ccd}_order_{order}.txt')
                except:
                    tramline_information = np.loadtxt(Path(__file__).resolve().parent / 'tramline_information' / f'tramlines_begin_end_ccd_{ccd}_order_{order}.txt')
            order_tramline_beginning_coefficients['ccd_'+ccd+'_order_'+str(order)] = tramline_information[0,:-1] # neglecting the buffer info in last cell
            order_tramline_ending_coefficients['ccd_'+ccd+'_order_'+str(order)]    = tramline_information[1,:-1] # neglecting the buffer info in last cell

    return(order_tramline_ranges, order_tramline_beginning_coefficients, order_tramline_ending_coefficients)

def get_master_dark(runs, archival=False):
    """
    Read in the dark runs (assuming they are the same exposure time) and combine them to a master dark dictionary with keys for the three CCDs.

    Parameters:
        runs (list): List of observation runs for dark frames.

    Returns:
        dict: A dictionary containing the master dark image for the three CCDs.
    
    """

    # Extract Images from CCDs 1-3
    images = dict()
    
    # Read in, overscan subtract and append images to array
    for ccd in [1,2,3]:
        images['ccd_'+str(ccd)] = []

        # If we use acrhival dark frames, we use frames 0224-0226 from date 001122.
        if not archival: date = config.date
        else:
            runs = ['0224']
            date = '001122'

        for run in runs:
            full_image, metadata = read_veloce_fits_image_and_metadata(config.working_directory+'observations/'+date+'/ccd_'+str(ccd)+'/'+date[-2:]+match_month_to_date(date)+str(ccd)+run+'.fits')
            trimmed_image, _, _, _ = substract_overscan(full_image, metadata)
            images['ccd_'+str(ccd)].append(trimmed_image)
        
        # Calculate median across all runs
        images['ccd_'+str(ccd)] = np.array(np.median(images['ccd_'+str(ccd)],axis=0),dtype=float)
    return(images)

def convert_bstar_to_telluric(bstar_flux_in_orders, filter_kernel_size=51, debug=False):
    """
    Convert B-star flux to telluric flux by normalising the B-star flux to unity
    using a smoothed Bstar spectrum (smoothed via median_filter).

    Parameters:
        bstar_flux_in_orders (np.array): An array of B-star flux in the orders.
        filter_kernel_size (int): The kernel size for the median filter. 51 default.
        debug (bool): Set to True to display debug plots.

    Returns:
        np.array: An array of telluric flux in the orders.
    """

    telluric_flux_in_orders = []

    # Calculate the median of the B-star flux in each order
    for order in range(len(bstar_flux_in_orders)):

        smooth_bstar_flux_in_order = median_filter(bstar_flux_in_orders[order], size=filter_kernel_size)

        bstar_flux_for_tellurics = bstar_flux_in_orders[order] / smooth_bstar_flux_in_order
        bstar_flux_for_tellurics[np.isnan(bstar_flux_for_tellurics)] = 1.0
        
        # Let's make sure we have reasonable telluric features that we can divide with.
        telluric_flux_in_order = bstar_flux_for_tellurics.clip(min = 0.01, max = 1.0)
        
        # cut first and last 100 pixels of telluric (~2.5% of the order each left and right).
        telluric_flux_in_order[:100] = 1.0
        telluric_flux_in_order[-100:] = 1.0

        # We do not expect any telluric lines in a lot of orders.
        if (order < 47) | (order in [51,52,53,54,55,56,60,61]): telluric_flux_in_order = 1.0 * np.ones(len(telluric_flux_in_order))
        
        # Specific cuts left and right
        if order in [47,48,49,50]: telluric_flux_in_order[:200] = 1.0
        if order in [65,66,67]: telluric_flux_in_order[-200:] = 1.0
        if order == 107: telluric_flux_in_order[2250:] = 1.0

        # Specific cuts for too strong absorption (50%) where we do not expect it
        if order <= 82: telluric_flux_in_order[telluric_flux_in_order < 0.5] = 1.0

        telluric_flux_in_orders.append(telluric_flux_in_order)

        # Plot the bstar flux divided by the smooth flux
        if debug:
            plt.figure(figsize=(10,5))
            plt.title(order)
            plt.plot(bstar_flux_for_tellurics, label = 'B-star flux divided by smooth flux', lw = 0.5, c='k')
            plt.plot(telluric_flux_in_order, label = 'Final telluric flux', lw = 1, c='C0')
            plt.legend(ncol=4)
            plt.ylim(-0.1, 1.5)
            if 'ipykernel' in sys.modules: plt.show()
            plt.close()

    return(np.array(telluric_flux_in_orders))

def get_tellurics_from_bstar(bstar_information, master_flat_images, debug=False):
    """
    Extract telluric orders from a B-star observation.

    Parameters:
        bstar_information (list): A list in the format [bstar_id, run, obsering_time]
        master_flat_images (dict): The master flat field images
        debug (bool): Set to True to display debug plots.

    Returns:
        telluric_flux_in_orders (np.array): An array of telluric flux in the orders.
        utmjd (float): The modified Julian date of the telluric/BStar observation.
    """

    bstar_id, run, time = bstar_information

    if debug:
        print('Starting Order Extraction')

    # Extract the B-star flux in the orders and the metadata
    bstar_flux_in_orders, metadata = extract_orders(
        ccd1_runs = [run],
        ccd2_runs = [run],
        ccd3_runs = [run],
        Bstar = True,
        master_flat_images = master_flat_images,
        debug_overscan = debug
    )

    if debug:
        print('Starting Telluric Flux Conversion')

    # Convert the B-star flux to telluric flux by normalising the B-star flux to unity
    telluric_flux_in_orders = convert_bstar_to_telluric(bstar_flux_in_orders, debug=debug)

    return(telluric_flux_in_orders, metadata['UTMJD'])
    

def extract_orders(ccd1_runs, ccd2_runs, ccd3_runs, Flat = False, update_tramlines_based_on_flat = False, LC = False, Bstar = False, Science = False, ThXe = False, master_darks = None, master_flat_images = None, exposure_time_threshold_darks = 300, use_tinney_ranges = False, debug_tramlines = False, debug_rows = False, debug_overscan=False):
    """
    Extracts spectroscopic orders from CCD images for various types of Veloce CCD images
    using predefined tramline ranges and providing detailed debug information.

    Parameters:
        ccd1_runs (list): List of observation runs for CCD 1.
        ccd2_runs (list): List of observation runs for CCD 2.
        ccd3_runs (list): List of observation runs for CCD 3.
        Flat (bool): Set to True to extract orders for flat field images.
        update_tramlines_based_on_flat (bool): Set to True to update tramline information based on flat field images. Can only be activated if Flat == True.
        LC (bool): Set to True to extract orders for laser comb calibration images.
        Bstar (bool): Set to True to extract orders for B-star calibration images.
        Science (bool): Set to True to extract orders for science observations.
        ThXe (bool): Set to True to extract orders for ThXe calibration images.
        master_darks (dict): A dictionary containing master dark images for the three CCDs.
        master_flat_images (dict): A dictionary containing master flat images for the three CCDs.
        exposure_time_threshold_darks (int, float): The threshold exposure time for applying master darks to science images in seconds. Default is 300 (seconds, i.e. 5 minutes).
        use_tinney_ranges (bool): Set to True to use tramline ranges specified by Chris Tinney.
        debug_tramlines (bool): Set to True to display debug plots for tramline extraction.
        debug_rows (bool): Set to True to display debug plots for row tracing.
        debug_overscan (bool): Set to True to display debug plots for overscan correction.

    Returns:
        tuple: A tuple containing:
            - counts_in_orders (np.array): An array of extracted counts in the orders.
            - noise_in_orders (np.array): An array of noise measurements in the orders.
            - metadata (dict, optional): Metadata related to the science observations, included only if `Science` is True.

    Depending on the flag settings, this function processes CCD data differently:
        - `Flat` affects the data normalization methods.
        - `LC` determines the calibration regime applied.
        - `Science` enables additional metadata extraction.
    """

    # Check if Flat, LC, Bstar, Science, ThXe are all False
    if not any([Flat, LC, Bstar, Science, ThXe]): raise ValueError('To extract orders appropriately, at least one of the following flags must be set to True: Flat, LC, Bstar, Science, ThXe.')

    # Raise ValueError if we try to update tramlines based on flat field images without Flat being True
    if (not Flat) & (update_tramlines_based_on_flat): raise ValueError('Can only update tramlines based on flat field images (not possible if Flat is False).')
    
    # Check if exposure_time_threshold_darks is a float or int
    if not isinstance(exposure_time_threshold_darks, (int, float)): raise ValueError('Exposure_time_threshold_darks must be a float.')

    # Raise warning if we use Science exposures but do not provide master darks.
    if (Science) & (master_darks is None): print('     --> Warning: Note using any dark subtraction.')

    order_ranges, order_beginning_coefficients, order_ending_coefficients = read_in_order_tramlines()

    # Extract initial order ranges and coefficients
    if use_tinney_ranges: order_ranges, order_beginning_coefficients, order_ending_coefficients = read_in_order_tramlines_tinney()    

    # Extract Images from CCDs 1-3
    images = dict()
    images_noise = dict()
    
    # Read in, overscan subtract and append images to array
    for ccd in [1,2,3]:
        
        images['ccd_'+str(ccd)] = []
        images_noise['ccd_'+str(ccd)] = []
        if ccd == 1: runs = ccd1_runs
        if ccd == 2: runs = ccd2_runs
        if ccd == 3: runs = ccd3_runs
        
        # Residual from implementing CURE mirror monitoring
        #if Flat | ThXe:
        #    f, ax = plt.subplots()

        for run in runs:
            full_image, metadata = read_veloce_fits_image_and_metadata(config.working_directory+'observations/'+config.date+'/ccd_'+str(ccd)+'/'+config.date[-2:]+match_month_to_date(config.date)+str(ccd)+run+'.fits')
            trimmed_image, os_median, os_rms, readout_mode = substract_overscan(full_image, metadata, debug_overscan)
            
            # Let's apply a reasonable dark subtraction
            if (Science) & (master_darks is not None):

                exp_time_science = float(metadata['EXPTIME'])

                shape_darkframe = np.shape(master_darks[list(master_darks.keys())[-1]]['ccd_'+str(ccd)])

                # Let's check if the science exposure is actually long enough to necessitate dark subtraction
                if (exp_time_science < exposure_time_threshold_darks):
                    if ccd == 1:
                        print('  --> Science exposure time ('+str(exp_time_science)+' seconds) is less than threshold of '+str(exposure_time_threshold_darks)+' seconds to apply dark subtraction.')
                        print('      Adjust kwarg exposure_time_threshold_darks to change this threshold.')
                
                # Ensure that the Darks and Science frames have the same shape (as we may use an archival DarkFrame)
                elif shape_darkframe != np.shape(trimmed_image):
                    print('Shapes of DarkFrame and Science do not match: ',shape_darkframe, np.shape(trimmed_image))
                    print('Skipping DarkFrame subtraction.')

                # If the science exposure is long enough, apply dark subtraction
                else:
                    # Let's find the best matching dark frame (just above the exposure time) and apply it based on the exposure time ratio of science and said dark frame.
                    exp_times_dark = np.array(list(master_darks.keys()),dtype=float)
                    # If possible: Select only the dark frames that are equal or longer than the science exposure time
                    if len(np.where(exp_times_dark-exp_time_science >= 0.0)[0]) > 0:
                        exp_times_dark = exp_times_dark[exp_times_dark-exp_time_science >= 0.0]
                    else:
                        print('  --> Warning: No DarkFrame > Science exposure time ('+str(exp_time_science)+'s) found. Using closest DarkFrame.')

                    # Now find the clostest one of those
                    best_matching_dark = exp_times_dark[np.argmin(np.abs(exp_times_dark-exp_time_science))]
                    exp_times_ratio_science_to_dark = float(exp_time_science / best_matching_dark)

                    # Calculate an exposure time adjusted dark frame, which has no negative entries.
                    adjusted_dark = (np.array(master_darks[str(best_matching_dark)]['ccd_'+str(ccd)], dtype=float) * exp_times_ratio_science_to_dark).clip(min=0.0)

                    if (ccd == 1): print('  --> Subtracting '+str(best_matching_dark)+'s Dark from Science exposure '+str(run)+' (D='+str(best_matching_dark)+'s vs. S='+str(exp_time_science)+'s, S/D = '+"{:.2f}".format(exp_times_ratio_science_to_dark)+' ~ '+str(int(np.median(adjusted_dark.flatten())))+' counts).')

                    # Let's check that the dark and science frames have the same dimenions.
                    # This may fail if the archival 2Amp dark is used for a 4Amp science frame.
                    if np.shape(adjusted_dark) != np.shape(trimmed_image): raise ValueError('Dark frame ('+str(np.shape(adjusted_dark)[0])+','+str(np.shape(adjusted_dark)[1])+') and science frame ('+str(np.shape(trimmed_image)[0])+','+str(np.shape(trimmed_image)[1])+') have different shapes (this is likely because of a 4Amp science vs. 2Amp archivel dark)!')

                    trimmed_image -= adjusted_dark

            # Add check if CURE mirror folded in: We expect strong signals for Flat and ThXe.
            use_this_image = True
            if Flat:
                # Residual from implementing CURE mirror monitoring
                #ax.hist(trimmed_image.flatten(),bins = np.linspace(0,65535,100), histtype='step', ls='dashed', label = 'Flat '+str(run))
                nanmed = np.percentile(trimmed_image,99)
                expectation = 5000
                if nanmed < expectation:
                    print('  --> Flat image '+str(run)+' for CCD '+str(ccd)+' has not enough signal ('+str(nanmed)+'<'+str(expectation)+'). Ignoring. Was CURE mirror maybe not folded in?')
                    use_this_image = False
            elif ThXe:
                # Residual from implementing CURE mirror monitoring
                #ax.hist(trimmed_image.flatten(),bins = np.linspace(0,65535,100), histtype='step', ls='dashed', label = 'ThXe '+str(run))
                nanmed = np.percentile(trimmed_image,99)
                if ccd == 1:
                    expectation = 80
                elif ccd == 2:
                    expectation = 200
                else:
                    expectation = 500
                if nanmed < expectation:
                    print('  --> ThXe image '+str(run)+' for CCD '+str(ccd)+' has not enough signal ('+str(nanmed)+'<'+str(expectation)+'). Ignoring. Was CURE mirror maybe not folded in?')
                    use_this_image = False

            # If we should have a master flat
            if (not Flat) & (master_flat_images is not None):
                # Use the master flat for correction
                trimmed_image /= master_flat_images['ccd_'+str(ccd)]

                if Science:
                    # Calculate noise:
                    # sqrt( flux + read-out-noise^2), so flux variance = flux + read-out-noise^2
                    # with read-out-noise = max(os_rms) / master_flat
                    # Note: We assume the maximum RMS in the quadrants to be the relevant RMS
                    trimmed_image_variance = trimmed_image + (np.ones(np.shape(trimmed_image)) * np.max([os_rms[quadrant] for quadrant in os_rms.keys()]) / master_flat_images['ccd_'+str(ccd)])**2
                    images_noise['ccd_'+str(ccd)].append(trimmed_image_variance)
            elif (not Flat) & (ccd == 1):
                print('     --> Warning: No flat-field correction applied')

            if use_this_image:
                images['ccd_'+str(ccd)].append(trimmed_image)
            elif (run == runs[-1]) & len(images['ccd_'+str(ccd)]) == 0:
                print('  --> No good Flat or ThXe available for CCD '+str(ccd)+'. Be careful!')
                images['ccd_'+str(ccd)].append(trimmed_image)

        # Residual from implementing CURE mirror monitoring
        #if Flat | ThXe:
        #    ax.set_xlabel('Counts')
        #    ax.set_ylabel('Number of Pixels')
        #    ax.legend(ncol=2,fontsize=8)
        #    ax.set_yscale('log')
        #    plt.show()
        #    plt.close()
        
        # For science: sum the flux and calculate the calculate median counts (we previously use the co-adding, but found the median to be more robust)
        if Science:
            images['ccd_'+str(ccd)] = np.array(np.sum(images['ccd_'+str(ccd)],axis=0),dtype=float)
            images_noise['ccd_'+str(ccd)] = np.array(np.sqrt(np.sum(images_noise['ccd_'+str(ccd)],axis=0)),dtype=float)
        # For calibration: calculate median counts
        else: images['ccd_'+str(ccd)] = np.array(np.median(images['ccd_'+str(ccd)],axis=0),dtype=float)

        if Flat:
            # Normalise so that maximum response = 1
            images['ccd_'+str(ccd)] /= np.nanmax(images['ccd_'+str(ccd)])
            # Ensure that Flat pixels without value are still available as 1.0
            images['ccd_'+str(ccd)][np.isnan(images['ccd_'+str(ccd)])] = 1.0
            # Ensure that Flat pixels with negative value or 0.0 exactly are reset to 1.0
            images['ccd_'+str(ccd)][np.where(images['ccd_'+str(ccd)] <= 0.0)] = 1.0

            # Identify the rough (too wide) tramline ranges for each order as reported by default fit or C.Tinney (with slight adjustments).
            try:
                order_ranges, order_beginning_coeffs, order_ending_coeffs = read_in_order_tramlines(use_default=True)
            except:
                order_ranges, order_beginning_coeffs, order_ending_coeffs = read_in_order_tramlines_tinney()

            # Update tramlines if this was requested
            if update_tramlines_based_on_flat:
                if ccd == 1:
                    print('  --> Optimising tramlines based on Flat images (saving at reduced_data/YYMMDD/_tramline_information/).')
                    print('      Check reduced_data/YYMMDD/_debug/debug_tramlines_flat.pdf for results.')
                for order in list(order_beginning_coefficients):
                    if order[4] == str(ccd):
                        optimise_tramline_polynomial(
                            overscan_subtracted_images = images['ccd_'+str(ccd)], 
                            order = order,
                            readout_mode = readout_mode,
                            order_ranges = order_ranges,
                            order_beginning_coeffs = order_beginning_coeffs,
                            order_ending_coeffs = order_ending_coeffs,
                            overwrite = False,
                            debug = debug_rows
                        )

    # Read in the overwritten tramline information for the night
    order_ranges, order_beginning_coefficients, order_ending_coefficients = read_in_order_tramlines()

    counts_in_orders = []
    if Science:
        noise_in_orders = []
    
    # Create the debug_tramlines plot if we are debugging or fitting the tramlines
    if debug_tramlines | (update_tramlines_based_on_flat & Flat):
        f, gs = plt.subplots(1,3,figsize=(15,4))
        for panel_index in [0,1,2]:

            if Science | ThXe:
                s = gs[panel_index].imshow(np.log10(images['ccd_'+str(panel_index+1)]), vmin=0.1, vmax=np.nanpercentile(np.log10(images['ccd_'+str(panel_index+1)]),95), cmap='Greys')
                cbar_label = r'$\log_{10}(\mathrm{Counts})$'
            else:
                if Flat: vmin = 0; vmax = 0.1
                elif LC: vmin = 1; vmax = 10
                else: vmin = 1; vmax = 50
                s = gs[panel_index].imshow(images['ccd_'+str(panel_index+1)], vmin=vmin, vmax=vmax, cmap='Greys')
                if Flat: cbar_label = r'Normalised Counts'
                else: cbar_label = r'Counts'

            gs[panel_index].set_title('CCD '+str(panel_index+1))
            cbar = plt.colorbar(s, ax=gs[panel_index-1],extend='both')
            cbar.set_label(cbar_label)
            gs[panel_index].set_xlabel('X Pixel')
            gs[panel_index].set_ylabel('Y Pixel')
            gs[panel_index].set_xlim(0,np.shape(images['ccd_'+str(panel_index+1)])[1])
            gs[panel_index].set_ylim(np.shape(images['ccd_'+str(panel_index+1)])[0],0)
    
    for order in order_beginning_coefficients.keys():
        ccd = order[4]

        # Prepare to the flux from each tramlines in a row; give NaN values to regions without flux
        order_counts = np.zeros(np.shape(images['ccd_'+str(ccd)])[1]); order_counts[:] = np.nan
        if Science:
            order_noise = np.zeros(np.shape(images_noise['ccd_'+str(ccd)])[1]); order_noise[:] = np.nan

        order_xrange_begin = np.array(polynomial_function(np.arange(np.shape(images['ccd_'+str(ccd)])[0]),*order_beginning_coefficients[order])-1,dtype=int)
        order_xrange_end   = np.array(polynomial_function(np.arange(np.shape(images['ccd_'+str(ccd)])[0]),*order_ending_coefficients[order])+1,dtype=int)

        # If we are using the LC, use the region 11+-6 pixels to the right of the end of the main tramline
        if LC:
            order_xrange_begin = np.array(polynomial_function(np.arange(np.shape(images['ccd_'+str(ccd)])[0]),*order_ending_coefficients[order])+5,dtype=int)
            order_xrange_end   = np.array(polynomial_function(np.arange(np.shape(images['ccd_'+str(ccd)])[0]),*order_ending_coefficients[order])+17,dtype=int)

        if debug_tramlines | (update_tramlines_based_on_flat & Flat):
            if order == list(order_beginning_coefficients.keys())[0]:
                label_left = 'Tramline left edges'
                label_right = 'Tramline right edges'
            else:
                label_left = '_nolegend_'
                label_right = '_nolegend_'
            gs[int(ccd)-1].plot(order_xrange_begin,np.arange(len(order_xrange_begin)),c='C0',lw=0.1,label=label_left)
            gs[int(ccd)-1].plot(order_xrange_end,np.arange(len(order_xrange_begin)),c='C1',lw=0.1,label=label_right)
        
        # Because of the extended overscan region in 4Amplifier readout mode, we have to adjust which region we are using the extract the orders from.
        if readout_mode == '2Amp': order_ranges_adjusted_for_readout_mode = order_ranges[order]
        elif readout_mode == '4Amp': order_ranges_adjusted_for_readout_mode = order_ranges[order][16:-16]
        else: raise ValueError('Cannot handle readout_mode other than 2Amp or 4Amp')

        # Let's loop over the tramlines
        for x_index, x in enumerate(order_ranges_adjusted_for_readout_mode):

            # For each tramline, find the relevant pixels and then sum across the rows
            counts_in_tramline = np.sum(images['ccd_'+str(ccd)][x,order_xrange_begin[x_index]:order_xrange_end[x_index]], axis=0)
            order_counts[order_ranges[order][0] + x_index] = counts_in_tramline

            # If we are working with the Science frame, also compute the noise:
            if Science:
                # For each tramline, find the relevant pixels and then sum across the rows
                order_noise[order_ranges[order][0] + x_index] = np.sqrt(np.sum(images_noise['ccd_'+str(ccd)][x,order_xrange_begin[x_index]:order_xrange_end[x_index]]**2, axis=0))

        counts_in_orders.append(order_counts)
        if Science:
            noise_in_orders.append(order_noise)

    if debug_tramlines | (update_tramlines_based_on_flat & Flat):
        if Flat: type='_flat'
        elif Science: type='_'+metadata['OBJECT']
        elif LC: type='_lc'
        elif ThXe: type='_thxe'
        else: raise ValueError('Unknown type of observation.')
        
        plt.tight_layout()

        Path(config.working_directory+'reduced_data/'+config.date+'/_debug').mkdir(parents=True, exist_ok=True)
        plt.savefig(config.working_directory+'reduced_data/'+config.date+f'/_debug/debug_tramlines{type}.pdf',bbox_inches='tight')
        if 'ipykernel' in sys.modules: plt.show()
        plt.close()
        
    if Science:
        return(np.array(counts_in_orders),np.array(noise_in_orders),metadata)
    elif Bstar:
        return(np.array(counts_in_orders), metadata)
    elif Flat:
        return(np.array(counts_in_orders), images)
    else:
        return(np.array(counts_in_orders))


def find_tramline_beginning_and_ending(order, x_index, x_pixels, previous_beginning, previous_ending, expected_tramline_width = 38, tolerance=2, tolerance_to_previous=3, debug=False):
    """
    Calculates the beginning and ending positions of a tramline for a specific row based on pixel intensity data that exceeds a certain threshold. 
    This function identifies significant gaps likely representing the space between the main tramline and outer fibers.

    Parameters:
        order (str): The order being analyzed, formatted as 'ccd_{ccd}_{order}'.
        x_index (int): Index of the row currently being analyzed.
        x_pixels (array): Array of pixel positions within the tramline region above a specified intensity threshold.
        previous_beginning (int): Beginning pixel index of the tramline in the previous row.
        previous_ending (int): Ending pixel index of the tramline in the previous row.
        expected_tramline_width (int, optional): Expected width of the tramline, typically ranging from 38 to 45 pixels. Default is 38.
        tolerance (int, optional): Tolerance level for identifying significant gaps between the main tramline and outer fibers, measured in pixels. Default is 2.
        tolerance_to_previous (int, optional): Tolerance for deviations from the previous row's tramline positions, measured in pixels. Default is 3.
        debug (bool, optional): If True, enables debug outputs for troubleshooting the tramline detection process.

    Returns:
        tuple (int, int): A tuple containing the beginning and ending pixel indices of the tramline for the current row.
                          Returns (np.nan, np.nan) if the calculated tramline positions are outside of the defined tolerances or if other validity tests fail.
    """

    # Calculate differences between pixels above the threshold (which wis used as input for x_pixels)
    differences = [x_pixels[i+1] - x_pixels[i] for i in range(len(x_pixels) - 1)]

    if debug:
        print('\n  --> Debugging Traminline Beginning/Ending for Order:',order)
        print('  --> x_index:',x_index)
        print('  --> x_pixels:',x_pixels[0],'...',x_pixels[-1])
        print('  --> Differences:',differences)

    # Initialise tramline_beginning and tramline_ending as nans    
    tramline_beginning = np.nan
    if len(x_pixels) < 1: return(np.nan, np.nan)
    elif x_pixels[-1] < 1: return(np.nan, np.nan)

    tramline_ending = np.nan
    
    # Identify segments above the tolerance, while allowing for gaps.
    current_gap = 0
    for i, diff in enumerate(differences):
        # if no gap, continue the sequence
        if diff == 1: current_gap = 0
        elif diff > 1:
            current_gap += diff  # Increment the gap count by the missing numbers
            # Check if the gap exceeds tolerance
            if current_gap > tolerance:
                if np.isnan(tramline_beginning): tramline_beginning = x_pixels[i+1]
                # Add tramline_ending, if it is ~expected_tramline_width, so < expected_tramline_width +- 4 from the tramline_beginning
                elif np.abs((x_pixels[i]+1 - tramline_beginning) - expected_tramline_width) <= 4:
                    tramline_ending = x_pixels[i]+1
                    current_gap = 0
                else:
                    if debug: print('  --> Not using: '+str(x_pixels[i]+1)+' with width '+str(x_pixels[i]+1 - tramline_beginning)+' because the following width was expected: '+str(expected_tramline_width-3)+'-'+str(expected_tramline_width+3))
                    current_gap = 0
                    
    if debug: print('  --> x_index Initial Beginning/End: ',x_index, tramline_beginning, tramline_ending)
                    
    # Force new beginning to be close to beginning of previous pixel within tolerance_to_previous
    if (np.abs(previous_beginning - tramline_beginning) > tolerance_to_previous):
        tramline_beginning = previous_beginning
    # Replace with previous, if we could not find a tramline_beginning
    # but only if the previous tramline_beginning is not too close to the left edge
    elif np.isnan(tramline_beginning):
        if previous_beginning > 2:
            if debug: print('  --> Correction 1: Setting tramline_beginning = previous_beginning')
            # For CCD1, we know that positions tend to decrease downwards -- but let's make sure it's not left of the search range!
            if order[4] == '1': tramline_beginning = np.max([x_pixels[0]+5,previous_beginning - 1])
            # For CCD2, the higher orders turn around pixel 600
            elif (order[4] == '2'):
                if (int(order[-3:]) > 115):
                    # We know that positions tend to increase downwards -- but let's make sure it's not right of the search range!
                    if x_index < 600: tramline_beginning = np.min([x_pixels[-1]-5,previous_beginning + 1])
                    # We know that positions tend to decrease downwards -- but let's make sure it's not left of the search range!
                    else: tramline_beginning = np.max([x_pixels[0]+5,previous_beginning - 1])
                else:
                    tramline_beginning = previous_beginning.clip(min = x_pixels[0]+5, max = x_pixels[-1]-5)
            # For most other orders, using the same pixel as the previous row is fine.
            else:
                tramline_beginning = previous_beginning.clip(min = x_pixels[0]+5, max = x_pixels[-1]-5)
        else:
            if debug: print('  --> Exception 1: Beginning too far left; returning (np.nan,np.nan)')
            return(np.nan, np.nan)

    # Force new ending to be close to ending of previous pixel within tolerance_to_previous
    if (np.abs(previous_ending - tramline_ending) > tolerance_to_previous) & (previous_ending - tramline_beginning < expected_tramline_width+3):
        if debug: print('  --> Correction 2: Setting tramline_ending = previous_ending, because difference previous_ending - tramline_ending above tolerance of '+str(tolerance_to_previous)+' and previous ending within expected_tramline_width')
        tramline_ending = previous_ending
    # Replace with previous, if we could not find a tramline_ending
    # but only if the previous tramline_ending is not too close to the left edge
    elif np.isnan(tramline_ending):
        if previous_ending > 2:
            if debug: print('  --> Correction 3: Too far off from previous ending. Setting tramline_ending = previous_ending.clip(min = x_pixels[0]+5, max = x_pixels[-1]-5)')
            tramline_ending = previous_ending.clip(min = x_pixels[0]+5, max = x_pixels[-1]-5)
        else:
            if debug: print('  --> Exception 2: Ending too far left 1, returning (np.nan,np.nan)')
            return(np.nan, np.nan)
    # If the tramline ending is too close to left edge, we return nans
    elif tramline_ending <= expected_tramline_width+3:
        if debug: print('  --> Exception 3: Ending too far left 2, returning (np.nan,np.nan)')
        return(np.nan, np.nan)
    
    # Make sure that the tramlines are reasonably wide.
    # We expect a tramline with width expected_tramline_width within tolerance.
    if tramline_ending - tramline_beginning < expected_tramline_width-4:
        if debug: print('  --> Exception 4: Tramline not wide enough: ', tramline_ending - tramline_beginning, ' (expexting ',expected_tramline_width,'). Returning (np.nan,np.nan)')
        return(np.nan, np.nan)

    if debug: print('  --> No Exception. Using: ',tramline_beginning, tramline_ending)
    
    return(tramline_beginning, tramline_ending)

def optimise_tramline_polynomial(overscan_subtracted_images, order, order_ranges, order_beginning_coeffs, order_ending_coeffs, readout_mode, overwrite=False, debug=False):
    """
    Optimizes the polynomial coefficients for defining the beginning and ending of tramlines in spectroscopic data
    for a given order and readout mode. This function fits polynomials to tramline boundaries based on
    overscan-subtracted images.

    Parameters:
        overscan_subtracted_images (list of ndarray): A list of 2D arrays, each representing an overscan-subtracted image.
        order (int or str): The spectral order to be processed.
        order_ranges (dict): A dictionary mapping each order to its full pixel range.
        order_beginning_coeffs (dict): A dictionary mapping each order to the beginning pixel positions of its tramlines.
        order_ending_coeffs (dict): A dictionary mapping each order to the ending pixel positions of its tramlines.
        readout_mode (str): The readout mode used during image acquisition, affecting the fitting process.
        overwrite (bool, optional): If True, overwrites the existing polynomial coefficient file located at 
            'VeloceReduction/tramline_information/tramlines_begin_end_{order}.txt'. Default is False.
        debug (bool, optional): If True, displays debug plots that illustrate the polynomial fitting process and 
            the derived tramline boundaries. Default is False.

    Returns:
        tuple of arrays: Returns a tuple containing two arrays:
            - The first array contains the polynomial coefficients for the tramline beginning.
            - The second array contains the polynomial coefficients for the tramline ending.
    """

    #if readout_mode != '2Amp': raise ValueError('Can only handle 2Amp readout mode')

    ccd = order[4]
    
    adjusted_order_pixel = []
    adjusted_order_beginning = []
    adjusted_order_ending = []

    image_dimensions = np.shape(overscan_subtracted_images)

    # leave option to adjust beginning and end of tramlines.
    # Set left and right adjustment to 15 for CCDs 1 and 2 and 10 for CCD 3 by default.
    # 20 would be too close to neighbouring tramlines. 15 is too broad for CCD3.
    tramline_buffer_left = -15
    tramline_buffer_right = 15
    if order[4] == '3':
        tramline_buffer_left = -10
        tramline_buffer_right = 10
    # Orders at the detector edges tend to need special treatment
    if order == 'ccd_3_order_104': tramline_buffer_left = -8

    order_xrange_begin = np.array(polynomial_function(np.arange(np.shape(overscan_subtracted_images)[0]),*order_beginning_coeffs[order])+tramline_buffer_left,dtype=int)
    order_xrange_end   = np.array(polynomial_function(np.arange(np.shape(overscan_subtracted_images)[0]),*order_ending_coeffs[order])+tramline_buffer_right,dtype=int)

    # Define buffer for beginning and ending of the CCD to avoid issues with tramlines at the edges
    buffer = dict()
    buffer['ccd_1_order_138'] = [200,320]
    buffer['ccd_1_order_139'] = [125,550]
    buffer['ccd_1_order_140'] = [125,350]
    buffer['ccd_1_order_141'] = [125,250]
    buffer['ccd_1_order_142'] = [125,250]
    buffer['ccd_1_order_143'] = [125,250]
    buffer['ccd_1_order_144'] = [125,250]
    buffer['ccd_1_order_145'] = [125,250]
    buffer['ccd_1_order_146'] = [125,250]
    buffer['ccd_1_order_147'] = [225,250]
    buffer['ccd_1_order_148'] = [150,250]
    buffer['ccd_1_order_149'] = [225,250]
    buffer['ccd_1_order_150'] = [325,250]
    buffer['ccd_1_order_151'] = [500,250]
    buffer['ccd_1_order_152'] = [800,250]
    buffer['ccd_1_order_153'] = [1150,250]
    buffer['ccd_1_order_154'] = [1350,800]
    buffer['ccd_1_order_155'] = [1530,850]
    buffer['ccd_1_order_156'] = [1650,1220]
    buffer['ccd_1_order_157'] = [1650,1300]
    buffer['ccd_1_order_158'] = [1500,1350]
    buffer['ccd_1_order_159'] = [1500,1200]
    buffer['ccd_1_order_160'] = [1500,1200]
    buffer['ccd_1_order_161'] = [1500,1200]
    buffer['ccd_1_order_162'] = [1500,1500]
    buffer['ccd_1_order_163'] = [1600,1200]
    buffer['ccd_1_order_164'] = [1600,1250]
    buffer['ccd_1_order_165'] = [1560,1400]
    buffer['ccd_1_order_166'] = [1600,1500]
    buffer['ccd_1_order_167'] = [1500,1200]
    buffer['ccd_2_order_103'] = [105,450]
    buffer['ccd_2_order_104'] = [110,100]
    buffer['ccd_2_order_105'] = [110,100]
    buffer['ccd_2_order_106'] = [110,100]
    buffer['ccd_2_order_107'] = [110,100]
    buffer['ccd_2_order_108'] = [110,100]
    buffer['ccd_2_order_109'] = [110,100]
    buffer['ccd_2_order_110'] = [110,100]
    buffer['ccd_2_order_111'] = [100,100]
    buffer['ccd_2_order_112'] = [100,100]
    buffer['ccd_2_order_113'] = [100,100]
    buffer['ccd_2_order_114'] = [100,100]
    buffer['ccd_2_order_115'] = [110,100]
    buffer['ccd_2_order_116'] = [130,100]
    buffer['ccd_2_order_117'] = [130,100]
    buffer['ccd_2_order_118'] = [130,100]
    buffer['ccd_2_order_119'] = [150,100]
    buffer['ccd_2_order_120'] = [200,120]
    buffer['ccd_2_order_121'] = [220,100]
    buffer['ccd_2_order_122'] = [220,120]
    buffer['ccd_2_order_123'] = [200,100]
    buffer['ccd_2_order_124'] = [200,100]
    buffer['ccd_2_order_125'] = [300,100]
    buffer['ccd_2_order_126'] = [200,100]
    buffer['ccd_2_order_127'] = [260,100]
    buffer['ccd_2_order_128'] = [240,100]
    buffer['ccd_2_order_129'] = [250,110]
    buffer['ccd_2_order_130'] = [250,120]
    buffer['ccd_2_order_131'] = [280,120]
    buffer['ccd_2_order_132'] = [260,110]
    buffer['ccd_2_order_133'] = [270,120]
    buffer['ccd_2_order_134'] = [330,125]
    buffer['ccd_2_order_135'] = [365,125]
    buffer['ccd_2_order_136'] = [300,145]
    buffer['ccd_2_order_137'] = [360,100]
    buffer['ccd_2_order_138'] = [500,170]
    buffer['ccd_2_order_139'] = [550,290]
    buffer['ccd_2_order_140'] = [1700,650]
    buffer['ccd_3_order_65'] = [105,50]
    buffer['ccd_3_order_66'] = [105,50]
    buffer['ccd_3_order_67'] = [105,70]
    buffer['ccd_3_order_68'] = [105,50]
    buffer['ccd_3_order_69'] = [105,50]
    buffer['ccd_3_order_70'] = [105,50]
    buffer['ccd_3_order_71'] = [125,50]
    buffer['ccd_3_order_72'] = [105,50]
    buffer['ccd_3_order_73'] = [105,50]
    buffer['ccd_3_order_74'] = [105,50]
    buffer['ccd_3_order_75'] = [105,50]
    buffer['ccd_3_order_76'] = [105,50]
    buffer['ccd_3_order_77'] = [105,50]
    buffer['ccd_3_order_78'] = [105,50]
    buffer['ccd_3_order_79'] = [105,50]
    buffer['ccd_3_order_80'] = [105,50]
    buffer['ccd_3_order_81'] = [105,50]
    buffer['ccd_3_order_82'] = [105,50]
    buffer['ccd_3_order_83'] = [115,50]
    buffer['ccd_3_order_84'] = [115,50]
    buffer['ccd_3_order_85'] = [130,50]
    buffer['ccd_3_order_86'] = [135,50]
    buffer['ccd_3_order_87'] = [135,50]
    buffer['ccd_3_order_88'] = [155,100]
    buffer['ccd_3_order_89'] = [190,100]
    buffer['ccd_3_order_90'] = [230,100]
    buffer['ccd_3_order_91'] = [230,100]
    buffer['ccd_3_order_92'] = [300,100]
    buffer['ccd_3_order_93'] = [310,100]
    buffer['ccd_3_order_94'] = [330,125]
    buffer['ccd_3_order_95'] = [350,125]
    buffer['ccd_3_order_96'] = [370,125]
    buffer['ccd_3_order_97'] = [360,125]
    buffer['ccd_3_order_98'] = [410,125]
    buffer['ccd_3_order_99'] = [370,150]
    buffer['ccd_3_order_100'] = [375,150]
    buffer['ccd_3_order_101'] = [580,150]
    buffer['ccd_3_order_102'] = [400,150]
    buffer['ccd_3_order_103'] = [710,150]
    buffer['ccd_3_order_104'] = [1600,150]

    # Because of the extended overscan region in 4Amplifier readout mode, we have to adjust which region we are using the extract the orders from.
    if readout_mode == '2Amp': order_ranges_adjusted_for_readout_mode = order_ranges[order]
    elif readout_mode == '4Amp': order_ranges_adjusted_for_readout_mode = order_ranges[order][16:-16]
    else: raise ValueError('Cannot handle readout_mode other than 2Amp or 4Amp')

    order_ranges_adjusted_for_readout_mode = np.arange(image_dimensions[0])

    for x_index, x in enumerate(order_ranges_adjusted_for_readout_mode):

        # The uppermost and lowermost ~100 pixels typically do no get a lot of exposure
        # For robustness, let's neglect these pixels
        if (x_index >= buffer[order][0]) & (x_index <= image_dimensions[0] - buffer[order][1]):
            x_pixels_to_be_tested_for_tramline = np.arange(order_xrange_begin[x_index],order_xrange_end[x_index])
            x_pixels_to_be_tested_for_tramline = x_pixels_to_be_tested_for_tramline[x_pixels_to_be_tested_for_tramline >= 0]
            x_pixels_to_be_tested_for_tramline = x_pixels_to_be_tested_for_tramline[x_pixels_to_be_tested_for_tramline < image_dimensions[1]]
            x_pixel_values_to_be_tested_for_tramline = overscan_subtracted_images[x,x_pixels_to_be_tested_for_tramline]

            if len(x_pixel_values_to_be_tested_for_tramline) > 0:

                # We assume that flat measurements should be ~halfway between minimum and maximum
                threshold = 0.5*(
                    np.nanmin(x_pixel_values_to_be_tested_for_tramline)+
                    np.nanmax(x_pixel_values_to_be_tested_for_tramline)
                )
                # The right side of CCD 2 with order 130-140 gets less dominant exposure -> lower threshold
                if (ccd == '2') & (order[-2] in ['3','4']):
                    threshold = 0.4*(
                        np.nanmin(x_pixel_values_to_be_tested_for_tramline)+
                        np.nanmax(x_pixel_values_to_be_tested_for_tramline)
                    )
                above_threshold = np.where(x_pixel_values_to_be_tested_for_tramline > threshold)[0]
                
                if debug & (x_index % 500 == 0): debug_find_tramline_row = True
                else: debug_find_tramline_row = False

                if debug_find_tramline_row:
                    f2, ax2 = plt.subplots()
                    ax2.set_title('x_index: '+str(x_index))
                    ax2.plot(
                        x_pixels_to_be_tested_for_tramline,
                        x_pixel_values_to_be_tested_for_tramline,
                        label = 'Flat'
                    )
                    ax2.plot(
                        x_pixels_to_be_tested_for_tramline[above_threshold],
                        threshold*np.ones(len(x_pixel_values_to_be_tested_for_tramline[above_threshold])),
                        c = 'C3', label = 'Threshold ('+"{:.1f}".format(threshold)+')'
                    )
                    ax2.set_xlabel('X Pixel')
                    ax2.set_ylabel('Counts')
                    ax2.set_ylim(0,1.1*np.max(x_pixel_values_to_be_tested_for_tramline))
                    plt.tight_layout()
                    if 'ipykernel' in sys.modules: plt.show()
                    plt.close(f2)
                
                # We expect slightly different widths for each tramlines in the different CCDs
                if ccd == '3': expected_tramline_width = 38
                elif ccd == '2': expected_tramline_width = 44
                elif ccd == '1': expected_tramline_width = 45
        
                if x_index == buffer[order][0]:
                    tramline_beginning = np.nan
                    tramline_ending = np.nan

                tramline_beginning, tramline_ending = find_tramline_beginning_and_ending(
                    order,
                    x_index,
                    x_pixels_to_be_tested_for_tramline[above_threshold],
                    previous_beginning = tramline_beginning,
                    previous_ending = tramline_ending,
                    expected_tramline_width = expected_tramline_width,
                    debug = debug_find_tramline_row
                )
                
                if debug_find_tramline_row: print('  --> x_index, beginning, ending, width: ',x_index, tramline_beginning, tramline_ending, tramline_ending-tramline_beginning)

                x_pixels_tramline = []
                if np.isfinite(tramline_beginning) & np.isfinite(tramline_ending):
                    x_pixels_tramline = np.arange(tramline_beginning,tramline_ending)
                    adjusted_order_pixel.append(x_index)
                    adjusted_order_beginning.append(tramline_beginning)
                    adjusted_order_ending.append(tramline_ending)

    adjusted_order_pixel     = np.array(adjusted_order_pixel)
    adjusted_order_beginning = np.array(adjusted_order_beginning)
    adjusted_order_ending    = np.array(adjusted_order_ending)

    old_order_beginning, old_order_ending = np.loadtxt(Path(__file__).resolve().parent / 'tramline_information' / f'tramlines_begin_end_{order}.txt')
    old_buffer = [old_order_beginning[-1],old_order_ending[-1]]
    old_order_beginning = old_order_beginning[:-1]
    old_order_ending = old_order_ending[:-1]
   
    if order not in ['ccd_1_order_167','ccd_1_order_166']:
        try:
            order_beginning_fit = curve_fit(
                polynomial_function,
                adjusted_order_pixel,
                adjusted_order_beginning,
                p0 = old_order_beginning,
                bounds=(
                    [old_order_beginning[0]-10,old_order_beginning[1]-0.01,old_order_beginning[2]-1e-06,old_order_beginning[3]-1e-11,old_order_beginning[4]-1e-13],
                    [old_order_beginning[0]+10,old_order_beginning[1]+0.01,old_order_beginning[2]+1e-06,old_order_beginning[3]+1e-11,0]
                    )
            )[0]
        except:
            if debug: print('Could not fit beginning of '+order+'. Using old beginning.')
            order_beginning_fit = old_order_beginning

        try:
            order_ending_fit = curve_fit(
                polynomial_function,
                adjusted_order_pixel,
                adjusted_order_ending,
                p0 = old_order_ending,
                bounds=(
                    [old_order_ending[0]-10,old_order_ending[1]-0.01,old_order_ending[2]-1e-06,old_order_ending[3]-1e-11,old_order_ending[4]-1e-13],
                    [old_order_ending[0]+10,old_order_ending[1]+0.01,old_order_ending[2]+1e-06,old_order_ending[3]+1e-11,0]
                )
            )[0]
        except:
            if debug: print('  --> Could not fit end of order '+order+'. Using old ending.')
            order_ending_fit = old_order_ending
    else:
        if debug: print('  --> Skipping fitting for '+order+'. Using old values.')
        order_beginning_fit = old_order_beginning
        order_ending_fit = old_order_ending

    Path(config.working_directory+'reduced_data/'+config.date+'/_tramline_information').mkdir(parents=True, exist_ok=True)
    np.savetxt(config.working_directory+'reduced_data/'+config.date+f'/_tramline_information/tramlines_begin_end_{order}.txt',
        np.array([
            ['#c0', 'c1', 'c2', 'c3', 'c4','buffer_pixel'],
            np.concatenate((order_beginning_fit,[buffer[order][0]])),
            np.concatenate((order_ending_fit,[buffer[order][1]]))
        ]),
        fmt='%s'
    )

    # Print/plot tramline extraction diagnostics
    if debug:
        f, ax = plt.subplots(figsize=(15,15))
        ax.set_title('Tramline Extraction for '+order, fontsize=20)
        if order in ['ccd_1_order_167','ccd_1_order_166']:
            ax.imshow(np.log10(overscan_subtracted_images),cmap='Greys', vmax = np.nanpercentile(np.log10(overscan_subtracted_images.flatten()),68), label = 'Flat Exposure')
        else:
            ax.imshow(np.log10(overscan_subtracted_images),cmap='Greys', label = 'Flat Exposure')
        ax.plot(order_xrange_begin-tramline_buffer_left,np.arange(len(order_xrange_begin)),c='C3',lw=0.5, ls = 'dashed', label = 'Initial Tramline Region')
        ax.plot(order_xrange_end-tramline_buffer_right,np.arange(len(order_xrange_begin)),c='C3',lw=0.5, ls = 'dashed', label = '_nolegend_')
        ax.plot(order_xrange_begin,np.arange(len(order_xrange_begin)),c='C3',lw=0.5, label = 'Initial Search Region')
        ax.plot(order_xrange_end,np.arange(len(order_xrange_begin)),c='C3',lw=0.5, label = '_nolegend_')
        ax.axhline(buffer[order][0], label = 'Edge Buffer Region')
        ax.axhline(image_dimensions[0]-buffer[order][1], label = '_nolegend_')
        ax.set_aspect(1/10)

        ax.plot(
            adjusted_order_beginning,
            adjusted_order_pixel,
            c = 'C1',
            label = 'Identified Tramline Beginning/Ending'
        )
        ax.plot(
            np.round(polynomial_function(np.arange(image_dimensions[0]), *order_beginning_fit),0),
            np.arange(image_dimensions[0]),
            c = 'C0',
            ls = 'dashed',
            label = 'Polynomial Fit to Tramline Beginning/Ending'
        )
        ax.plot(
            adjusted_order_ending,
            adjusted_order_pixel,
            c = 'C1',
            label = '_nolegend_'
        )
        ax.plot(
            np.round(polynomial_function(np.arange(image_dimensions[0]), *order_ending_fit),0),
            np.arange(image_dimensions[0]),
            c = 'C0',
            ls = 'dashed',
            label = '_nolegend_'
        )

        ax.set_xlim(
            np.nanmax([np.nanmin(order_xrange_begin) - 20,-20]), 
            np.nanmin([np.nanmax(order_xrange_end) + 20, image_dimensions[1] + 20])
        )

        print('  --> Old vs New with old/new buffers: ',old_buffer, buffer[order])
        print('  --> Beginning:')
        print('      --> Old: ',[f"{number:.4e}" for number in old_order_beginning])
        print('      --> New: ',[f"{number:.4e}" for number in order_beginning_fit])
        print('  --> Ending:')
        print('      --> Old: ',[f"{number:.4e}" for number in old_order_ending])
        print('      --> New: ',[f"{number:.4e}" for number in order_ending_fit])
        
        ax.set_xlabel('X Pixels (Zoom)',fontsize=15)
        ax.set_ylabel('Y Pixels',fontsize=15)
        ax.legend(loc = 'upper left',fontsize=15)
        if (order == 'ccd_1_order_141') & overwrite: plt.savefig(Path(__file__).resolve().parent / 'joss_paper' / f'tramline_extraction_example_{order}.png',dpi=100,bbox_inches='tight')
        if 'ipykernel' in sys.modules: plt.show()
        plt.close(f)
        
    return(order_beginning_fit, order_ending_fit)