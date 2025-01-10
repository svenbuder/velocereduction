import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob

from . import config
from .utils import read_veloce_fits_image_and_metadata, match_month_to_date, polynomial_function

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
        plt.figure()
        plt.hist(full_image.flatten(),bins=np.linspace(975,1025,100))
        plt.show()
        plt.close()

        plt.figure(figsize=(10,10))
        s = plt.imshow(full_image, vmin=975, vmax = 1025)
        plt.colorbar(s)
        plt.show()
        plt.close()

    if metadata['READOUT'] == '4Amp':

        # Quadrant 1: :2120 and :2112
        quadrant1 = np.array(full_image[32:2120-32,32:2112-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[:32,:2112] = True
        overscan[:2120,:32] = True
        overscan[:32,2112:2112+32] = True
        overscan[:2120,2112:2112+32] = True
        overscan = full_image[overscan]

        # We report the median overscan
        overscan_median['q1'] = int(np.median(overscan))
        # And we calculate a robust standard deviation, i.e.,
        # half the difference between 16th and 84th percentile
        overscan_rms['q1'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant1 -= overscan_median['q1']

        # Quadrant 2: 2120: and :2112
        quadrant2 = np.array(full_image[2120+32:-32,32:2112-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[2120:2120+32,:2112] = True
        overscan[2120:,:32] = True
        overscan[-32:,:2112] = True
        overscan[2120:,2112:2112+32] = True
        overscan = full_image[overscan]

        overscan_median['q2'] = int(np.median(overscan))
        overscan_rms['q2'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant2 = (quadrant2 - overscan_median['q2'])

        # Quadrant 3: 2120: and 2112:
        quadrant3 = np.array(full_image[2120+32:-32,2112+32:-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[2120:2120+32,2112:] = True
        overscan[2120:,2120:2120+32] = True
        overscan[-32:,2112:] = True
        overscan[2120:,-32:] = True
        overscan = full_image[overscan]

        overscan_median['q3'] = int(np.median(overscan))
        overscan_rms['q3'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant3 -= overscan_median['q3']

        # Quadrant 4: :2120 and 2112:
        quadrant4 = np.array(full_image[32:2120-32,2112+32:-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[:32,2112:] = True
        overscan[2120-32:2120,2112:] = True
        overscan[:2120,-32:] = True
        overscan[:2120,2112:2112+32] = True
        overscan = full_image[overscan]

        overscan_median['q4'] = int(np.median(overscan))
        overscan_rms['q4'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant4 -= overscan_median['q4']

        trimmed_image = np.hstack([np.vstack([quadrant1,quadrant2]),np.vstack([quadrant4,quadrant3])]).clip(min=0.0)

    if metadata['READOUT'] == '2Amp':

        # Quadrant 1: :2088 and :
        quadrant1 = np.array(full_image[32:-32,32:2112-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[:32,:2112] = True # top
        overscan[-32:,:2112] = True # bottom
        overscan[:,:32] = True # left
        overscan[:,2112-32:2112] = True # right
        overscan = full_image[overscan]
        overscan_median['q1'] = int(np.median(overscan))
        overscan_rms['q1'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant1 -= overscan_median['q1']

        # Quadrant 2: 2088: and :2112
        quadrant2 = np.array(full_image[32:-32,2112+32:],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[:32,2112:] = True # top
        overscan[-32:,2112:] = True # bottom
        overscan[:,2112:2112+32] = True # left
        overscan[:,-32:] = True # left        
        overscan = full_image[overscan]
        overscan_median['q2'] = int(np.median(overscan))
        overscan_rms['q2'] = np.diff(np.percentile(overscan,q=[16,84]))/2
        quadrant2 = (quadrant2 - overscan_median['q2'])

        trimmed_image = np.hstack([quadrant1,quadrant2]).clip(min=0.0)

    if debug_overscan:
        plt.figure(figsize=(10,10))
        s = plt.imshow(trimmed_image, vmin = -5, vmax = 100)
        plt.colorbar(s)
        plt.show()
        plt.close()
        
    if debug_overscan:
        print(overscan_median, overscan_rms, metadata['READOUT'])
        
    return(trimmed_image, overscan_median, overscan_rms, metadata['READOUT'])

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

    with open('./VeloceReduction/veloce_reference_data/azzurro-th-m138-167-all.txt') as fp:
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

    with open('./VeloceReduction/veloce_reference_data/verde-th-m104-139-all.txt') as fp:
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

    with open('./VeloceReduction/veloce_reference_data/rosso-th-m65-104-all.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if cnt % 4 == 0:
                split_lines = line[:-1].split(' ')
                order = int(split_lines[0])
                if order < 100:
                    order_str = '0'+str(order)
                else:
                    order_str = str(order)
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

def read_in_order_tramlines():
    """
    Reads in optimized tramline information specifying the pixel positions for the beginning and ending of each 
    spectroscopic order across three CCDs. The data is read from text files and used to populate three dictionaries 
    with pixel information for each order.

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

    Returns:
        tuple: Contains three dictionaries:
            - order_tramline_ranges (dict): Mapping of each spectroscopic order to its full pixel range.
            - order_tramline_beginning_coefficients (dict): Mapping of each order to the beginning pixel positions of its tramlines.
            - order_tramline_ending_coefficients (dict): Mapping of each order to the ending pixel positions of its tramlines.

    Each dictionary key is formatted as 'ccd_{ccd}_{order}', where '{ccd}' is the CCD number (1, 2, or 3),
    and '{order}' is the specific order number on that CCD.
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

            tramline_information = np.loadtxt('./VeloceReduction/tramline_information/tramlines_begin_end_ccd_'+ccd+'_order_'+str(order)+'.txt')
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
        for run in runs:
            if not archival:
                full_image, metadata = read_veloce_fits_image_and_metadata(config.working_directory+'observations/'+config.date+'/ccd_'+str(ccd)+'/'+config.date[-2:]+match_month_to_date(config.date)+str(ccd)+run+'.fits')
            else:
                # Use the archival data from 001122
                full_image, metadata = read_veloce_fits_image_and_metadata(config.working_directory+'observations/001122/ccd_'+str(ccd)+'/22nov'+str(ccd)+'0224.fits')
            trimmed_image, _, _, _ = substract_overscan(full_image, metadata)
            images['ccd_'+str(ccd)].append(trimmed_image)
        
        # Calculate median across all runs
        images['ccd_'+str(ccd)] = np.array(np.median(images['ccd_'+str(ccd)],axis=0),dtype=float)
    return(images)


def extract_orders(ccd1_runs, ccd2_runs, ccd3_runs, Flat = False, update_tramlines_based_on_flat = False, LC = False, Science = False, master_darks = None, exposure_time_threshold_darks = 300, use_tinney_ranges = False, debug_tramlines = False, debug_overscan=False):
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
        Science (bool): Set to True to extract orders for science observations.
        master_darks (dict): A dictionary containing master dark images for the three CCDs.
        exposure_time_threshold_darks (int, float): The threshold exposure time for applying master darks to science images in seconds. Default is 300 (seconds, i.e. 5 minutes).
        use_tinney_ranges (bool): Set to True to use tramline ranges specified by Chris Tinney.
        debug_tramlines (bool): Set to True to display debug plots for tramline extraction.
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

    # Raise ValueError if we try to update tramlines based on flat field images without Flat being True
    if (not Flat) & (update_tramlines_based_on_flat):
        raise ValueError('Cannot update tramlines based on flat field images if Flat is False')
    
    # Check if exposure_time_threshold_darks is a float or int
    if not isinstance(exposure_time_threshold_darks, (int, float)):
        raise ValueError('exposure_time_threshold_darks must be a float.')

    # Raise warning if we use Science exposures but do not provide master darks.
    if (Science) & (master_darks is None):
        print('  -> Warning: Note using any dark subtraction.')

    order_ranges, order_beginning_coefficients, order_ending_coefficients = read_in_order_tramlines()

    if use_tinney_ranges:
        # Extract initial order ranges and coefficients
        order_ranges, order_beginning_coefficients, order_ending_coefficients = read_in_order_tramlines_tinney()    

    # Extract Images from CCDs 1-3
    images = dict()
    
    # Read in, overscan subtract and append images to array
    for ccd in [1,2,3]:
        
        images['ccd_'+str(ccd)] = []
        if ccd == 1: runs = ccd1_runs
        if ccd == 2: runs = ccd2_runs
        if ccd == 3: runs = ccd3_runs
        
        for run in runs:
            full_image, metadata = read_veloce_fits_image_and_metadata(config.working_directory+'observations/'+config.date+'/ccd_'+str(ccd)+'/'+config.date[-2:]+match_month_to_date(config.date)+str(ccd)+run+'.fits')
            trimmed_image, os_median, os_rms, readout_mode = substract_overscan(full_image, metadata, debug_overscan)
            
            # Let's apply a reasonable dark subtraction
            if (Science) & (master_darks is not None):

                exp_time_science = float(metadata['EXPTIME'])

                # Let's check if the science exposure is actually long enough to necessitate dark subtraction
                if (ccd == 1) & (exp_time_science < exposure_time_threshold_darks):
                    print('  --> Science exposure time ('+str(exp_time_science)+' seconds) is less than threshold of '+str(exposure_time_threshold_darks)+' seconds to apply dark subtraction.')
                    print('      Adjust kwarg exposure_time_threshold_darks to change this threshold.')
                
                # If the science exposure is long enough, apply dark subtraction
                else:
                    # Let's find the best matching dark frame (just above the exposure time) and apply it based on the exposure time ratio of science and said dark frame.
                    exp_times_dark = np.array(list(master_darks.keys()),dtype=float)
                    # If possible: Select only the dark frames that are equal or longer than the science exposure time
                    if len(np.where(exp_times_dark-exp_time_science >= 0.0)[0]) > 0:
                        exp_times_dark = exp_times_dark[exp_times_dark-exp_time_science >= 0.0]
                    else:
                        print('  --> Warning: No DarkFrame with exposure time longer than Science exposure time ('+str(exp_time_science)+'s) found. Using closest DarkFrame.')

                    # Now find the smallest one of those
                    best_matching_dark = exp_times_dark[np.argmin(exp_times_dark-exp_time_science)]
                    exp_times_ratio_science_to_dark = float(exp_time_science / best_matching_dark)

                    # Calculate an exposure time adjusted dark frame, which has no negative entries.
                    adjusted_dark = (np.array(master_darks[str(best_matching_dark)]['ccd_'+str(ccd)], dtype=float) * exp_times_ratio_science_to_dark).clip(min=0.0)

                    if (ccd == 1):
                        print('  --> Subtracting '+str(best_matching_dark)+'s dark frame from Science exposure '+str(run)+' (D='+str(best_matching_dark)+'s vs. S='+str(exp_time_science)+'s, S/D = '+"{:.2f}".format(exp_times_ratio_science_to_dark)+' ~ '+str(int(np.median(adjusted_dark.flatten())))+' counts).')

                    # Let's check that the dark and science frames have the same dimenions.
                    # This may fail if the archival 2Amp dark is used for a 4Amp science frame.
                    if np.shape(adjusted_dark) != np.shape(trimmed_image):
                        raise ValueError('Dark frame ('+str(np.shape(adjusted_dark)[0])+','+str(np.shape(adjusted_dark)[1])+') and science frame ('+str(np.shape(trimmed_image)[0])+','+str(np.shape(trimmed_image)[1])+') have different shapes (this is likely because of a 4Amp science vs. 2Amp archivel dark)!')

                    trimmed_image -= adjusted_dark

            images['ccd_'+str(ccd)].append(trimmed_image)
        
        # For science: sum counts
        if Science:
            images['ccd_'+str(ccd)] = np.array(np.median(images['ccd_'+str(ccd)],axis=0),dtype=float)
        # For calibration: calculate median
        else:
            images['ccd_'+str(ccd)] = np.array(np.median(images['ccd_'+str(ccd)],axis=0),dtype=float)

        if Flat:
            # Normalise so that maximum response = 1
            images['ccd_'+str(ccd)] /= np.nanmax(images['ccd_'+str(ccd)])
            # Ensure that Flat pixels without value are still available as 1.0
            images['ccd_'+str(ccd)][np.isnan(images['ccd_'+str(ccd)])] = 1.0
            # Ensure that Flat pixels with negative value or 0.0 exactly are reset to 1.0
            images['ccd_'+str(ccd)][np.where(images['ccd_'+str(ccd)] <= 0.0)] = 1.0

            if update_tramlines_based_on_flat:
                for order in list(order_beginning_coefficients):
                    optimise_tramline_polynomial(
                        overscan_subtracted_images = images['ccd_'+str(ccd)], 
                        order = order,
                        readout_mode = readout_mode,
                        overwrite = True,
                        debug = False
                    )
                # Read in the overwritten tramline information
                order_ranges, order_beginning_coefficients, order_ending_coefficients = read_in_order_tramlines()

    counts_in_orders = []
    noise_in_orders = []
    
    if debug_tramlines:
        f, gs = plt.subplots(1,3,figsize=(12,4))
        for panel_index in [0,1,2]:
            if Flat: vmin = 0; vmax = 0.1
            elif LC: vmin = 1; vmax = 10
            else: vmin = 1; vmax = 50
            s = gs[panel_index].imshow(images['ccd_'+str(panel_index+1)], vmin=vmin, vmax=vmax, cmap='Greys')
            gs[panel_index].set_title('CCD '+str(panel_index+1))
            plt.colorbar(s, ax=gs[panel_index-1])
            gs[panel_index].set_xlim(0,np.shape(images['ccd_'+str(panel_index+1)])[1])
            gs[panel_index].set_ylim(np.shape(images['ccd_'+str(panel_index+1)])[0],0)
    
    for order in order_beginning_coefficients.keys():
        ccd = order[4]

        # Prepare to the flux from each tramlines in a row; give NaN values to regions without flux
        order_counts = np.zeros(np.shape(images['ccd_'+str(ccd)])[1]); order_counts[:] = np.nan
        order_noise = np.zeros(np.shape(images['ccd_'+str(ccd)])[1]); order_noise[:] = np.nan

        order_xrange_begin = np.array(polynomial_function(np.arange(np.shape(images['ccd_'+str(ccd)])[0]),*order_beginning_coefficients[order]),dtype=int)
        order_xrange_end   = np.array(polynomial_function(np.arange(np.shape(images['ccd_'+str(ccd)])[0]),*order_ending_coefficients[order]),dtype=int)

        # If we are using the LC, use the region 11+-6 pixels to the right of the end of the main tramline
        if LC:
            order_xrange_begin = np.array(polynomial_function(np.arange(np.shape(images['ccd_'+str(ccd)])[0]),*order_ending_coefficients[order])+5,dtype=int)
            order_xrange_end   = np.array(polynomial_function(np.arange(np.shape(images['ccd_'+str(ccd)])[0]),*order_ending_coefficients[order])+17,dtype=int)

        if debug_tramlines:
            gs[int(ccd)-1].plot(order_xrange_begin,np.arange(len(order_xrange_begin)),c='C0',lw=0.1)
            gs[int(ccd)-1].plot(order_xrange_end,np.arange(len(order_xrange_begin)),c='C1',lw=0.1)
        
        # Because of the extended overscan region in 4Amplifier readout mode, we have to adjust which region we are using the extract the orders from.
        if readout_mode == '2Amp':
            order_ranges_adjusted_for_readout_mode = order_ranges[order]
        elif readout_mode == '4Amp':
            order_ranges_adjusted_for_readout_mode = order_ranges[order][16:-16]
        else:
            raise ValueError('Cannot handle readout_mode other than 2Amp or 4Amp')

        for x_index, x in enumerate(order_ranges_adjusted_for_readout_mode):
            
            counts_in_pixels_to_be_summed = images['ccd_'+str(ccd)][x,order_xrange_begin[x_index]:order_xrange_end[x_index]]
            
            order_counts[order_ranges[order][0] + x_index] = np.sum(counts_in_pixels_to_be_summed,axis=0)
            
            # We are making the quick assumption that the read noise is simply the maximum overscan RMS
            total_read_noise = np.max([os_rms[region] for region in os_rms.keys()])*np.sqrt(len(counts_in_pixels_to_be_summed))

            # For science: multiply read noise with nr of runs,
            # since we are coadding frames, rather than using median
            if Science:
                total_read_noise *= np.sqrt(len(runs))

            # if debug_tramlines & Science & (x_index == 2000):
            #     print(order)
            #     print('x_index:      ',2000)
            #     print('sum(counts):  ',np.sum(counts_in_pixels_to_be_summed,axis=0))
            #     print('sqrt(counts): ',np.sqrt(np.sum(counts_in_pixels_to_be_summed,axis=0)))
            #     print('read noise:   ',total_read_noise)
            #     print('counts:       ',counts_in_pixels_to_be_summed)
            #     plt.figure()
            #     plt.title(order)
            #     plt.plot(counts_in_pixels_to_be_summed)
            #     plt.show()
            #     plt.close()
                
            # noise = sqrt(flux + pixel_read_noise**2 * nr of pixels * nr of exposures)
            order_noise[order_ranges[order][0] + x_index] = np.sqrt(np.sum(counts_in_pixels_to_be_summed,axis=0) + total_read_noise**2)

        counts_in_orders.append(order_counts)
        noise_in_orders.append(order_noise)

    if debug_tramlines:
        if Flat:
            type='_flat'
        elif Science:
            type='_science'
        elif LC:
            type='_lc'
        else:
            type=''
        
        plt.tight_layout()
        plt.savefig('./VeloceReduction/tramline_information/debug_tramlines'+type+'.pdf',dpi=400,bb_inches='tight')
        plt.show()
        plt.close()
        
    if Science:
        return(np.array(counts_in_orders),np.array(noise_in_orders),metadata)
    else:
        return(np.array(counts_in_orders),np.array(noise_in_orders))


def find_tramline_beginning_and_ending(x_index, x_pixels, previous_beginning, previous_ending, expected_tramline_width = 38, tolerance=2, tolerance_to_previous=3, debug=False):
    """
    Calculates the beginning and ending positions of a tramline for a specific row based on pixel intensity data that exceeds a certain threshold. 
    This function identifies significant gaps likely representing the space between the main tramline and outer fibers.

    Parameters:
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
        print('x_index:')
        print(x_index)
        print('x_pixels:')
        print(x_pixels)
        print('differences')
        print(differences)

    # Initialise tramline_beginning and tramline_ending as nans    
    tramline_beginning = np.nan
    if len(x_pixels) < 1:
        return(np.nan, np.nan)
    elif x_pixels[-1] < 1:
        return(np.nan, np.nan)
    tramline_ending = np.nan
    
    # Identify segments above the tolerance, while allowing for gaps.
    current_gap = 0
    for i, diff in enumerate(differences):
        if diff == 1:
            # No gap, continue the sequence
            current_gap = 0
        elif diff > 1:
            current_gap += diff  # Increment the gap count by the missing numbers
            # Check if the gap exceeds tolerance
            if current_gap > tolerance:
                if np.isnan(tramline_beginning):
                    tramline_beginning = x_pixels[i+1]
                # Add tramline_ending, if it is ~expected_tramline_width, so < expected_tramline_width +- 4 from the tramline_beginning
                elif np.abs((x_pixels[i]+1 - tramline_beginning) - expected_tramline_width) <= 4:
                    tramline_ending = x_pixels[i]+1
                    current_gap = 0
                else:
                    if debug:
                        print('Not using: ',x_pixels[i]+1, x_pixels[i]+1 - tramline_beginning, 'expected: ',expected_tramline_width-3,expected_tramline_width+3)
                    current_gap = 0
                    
    if debug:
        print('x_index Initial Beginning/End')
        print(x_index, tramline_beginning, tramline_ending)
                    
    # Force new beginning to be close to beginning of previous pixel within tolerance_to_previous
    if (np.abs(previous_beginning - tramline_beginning) > tolerance_to_previous):
        tramline_beginning = previous_beginning
    # Replace with previous, if we could not find a tramline_beginning
    # but only if the previous tramline_beginning is not too close to the left edge
    elif np.isnan(tramline_beginning):
        if previous_beginning > 2:
            if debug:
                print('-> tramline_beginning = previous_beginning')
            tramline_beginning = previous_beginning
        else:
            if debug:
                print('-> beginning too far left')
            return(np.nan, np.nan)

    # Force new ending to be close to ending of previous pixel within tolerance_to_previous
    if (np.abs(previous_ending - tramline_ending) > tolerance_to_previous) & (previous_ending - tramline_beginning < expected_tramline_width+3):
        if debug:
            print('Difference previous_ending - tramline_ending above tolerance of '+str(tolerance_to_previous)+' and previous ending within expected_tramline_width')
        tramline_ending = previous_ending
    # Replace with previous, if we could not find a tramline_ending
    # but only if the previous tramline_ending is not too close to the left edge
    elif np.isnan(tramline_ending):
        if previous_ending > 2:
            if debug:
                print('-> tramline_ending = previous_ending')
            tramline_ending = previous_ending
        else:
            if debug:
                print('-> ending too far left 1')
            return(np.nan, np.nan)
    # If the tramline ending is too close to left edge, we return nans
    elif tramline_ending <= expected_tramline_width+3:
        if debug:
            print('-> ending too far left 2')
        return(np.nan, np.nan)
    
    # Make sure that the tramlines are reasonably wide.
    # We expect a tramline with width expected_tramline_width within tolerance.
    if tramline_ending - tramline_beginning < expected_tramline_width-4:
        if debug:
            print('-> not wide enough', tramline_ending - tramline_beginning)
        return(np.nan, np.nan)

    if debug:
        print('-> end, ',tramline_beginning, tramline_ending)
    
    return(tramline_beginning, tramline_ending)

def optimise_tramline_polynomial(overscan_subtracted_images, order, readout_mode, overwrite=False, debug=False):
    """
    Optimizes the polynomial coefficients for defining the beginning and ending of tramlines in spectroscopic data
    for a given order and readout mode. This function fits polynomials to tramline boundaries based on
    overscan-subtracted images.

    Parameters:
        overscan_subtracted_images (list of ndarray): A list of 2D arrays, each representing an overscan-subtracted image.
        order (int or str): The spectral order to be processed.
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

    if readout_mode != '2Amp':
        raise ValueError('Can only handle 2Amp readout mode')

    ccd = order[4]
    
    adjusted_order_pixel = []
    adjusted_order_beginning = []
    adjusted_order_ending = []

    image_dimensions = np.shape(overscan_subtracted_images['ccd_'+str(ccd)])

    # Identify the rough (too wide) tramline ranges for each order as reported by C.Tinney (with slight adjustments).
    order_ranges, order_beginning_coeffs, order_ending_coeffs = read_in_order_tramlines_tinney()

    # leave option to adjust beginning and end of tramlines.
    # Set left and right adjustment to 0 by default
    left = 0
    right = 0

    order_xrange_begin = np.array(polynomial_function(np.arange(np.shape(overscan_subtracted_images['ccd_'+str(ccd)])[0]),*order_beginning_coeffs[order])+left,dtype=int)
    order_xrange_end   = np.array(polynomial_function(np.arange(np.shape(overscan_subtracted_images['ccd_'+str(ccd)])[0]),*order_ending_coeffs[order])+right,dtype=int)

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
    buffer['ccd_1_order_165'] = [1560,1350]
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
    buffer['ccd_2_order_120'] = [190,100]
    buffer['ccd_2_order_121'] = [220,100]
    buffer['ccd_2_order_122'] = [170,100]
    buffer['ccd_2_order_123'] = [200,100]
    buffer['ccd_2_order_124'] = [200,100]
    buffer['ccd_2_order_125'] = [220,100]
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

    f, ax = plt.subplots(figsize=(15,15))
    ax.set_title('Tramline Extraction for '+order, fontsize=20)
    ax.imshow(np.log10(overscan_subtracted_images['ccd_'+str(ccd)]),cmap='Greys', label = 'Flat Exposure')
    ax.plot(order_xrange_begin,np.arange(len(order_xrange_begin)),c='C3',lw=0.5, label = 'Initial Search Region')
    ax.plot(order_xrange_end,np.arange(len(order_xrange_begin)),c='C3',lw=0.5, label = '_nolegend_')
    ax.axhline(buffer[order][0], label = 'Edge Buffer Region')
    ax.axhline(image_dimensions[0]-buffer[order][1], label = '_nolegend_')
    ax.set_aspect(1/10)

    # Because of the extended overscan region in 4Amplifier readout mode, we have to adjust which region we are using the extract the orders from.
    if readout_mode == '2Amp':
        order_ranges_adjusted_for_readout_mode = order_ranges[order]
    elif readout_mode == '4Amp':
        order_ranges_adjusted_for_readout_mode = order_ranges[order][16:-16]
    else:
        raise ValueError('Cannot handle readout_mode other than 2Amp or 4Amp')

    order_ranges_adjusted_for_readout_mode = np.arange(image_dimensions[0])

    for x_index, x in enumerate(order_ranges_adjusted_for_readout_mode):

        # The uppermost and lowermost ~100 pixels typically do no get a lot of exposure
        # For robustness, let's neglect these pixels
        if (x_index >= buffer[order][0]) & (x_index <= image_dimensions[0] - buffer[order][1]):
            x_pixels_to_be_tested_for_tramline = np.arange(order_xrange_begin[x_index],order_xrange_end[x_index])
            x_pixels_to_be_tested_for_tramline = x_pixels_to_be_tested_for_tramline[x_pixels_to_be_tested_for_tramline >= 0]
            x_pixels_to_be_tested_for_tramline = x_pixels_to_be_tested_for_tramline[x_pixels_to_be_tested_for_tramline < image_dimensions[1]]
            x_pixel_values_to_be_tested_for_tramline = overscan_subtracted_images['ccd_'+str(ccd)][x,x_pixels_to_be_tested_for_tramline]

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
                
                if debug:
                    f2, ax2 = plt.subplots()
                    ax2.plot(
                        x_pixels_to_be_tested_for_tramline,
                        x_pixel_values_to_be_tested_for_tramline
                    )
                    ax2.plot(
                        x_pixels_to_be_tested_for_tramline[above_threshold],
                        threshold*np.ones(len(x_pixel_values_to_be_tested_for_tramline[above_threshold])),
                        c = 'C3'
                    )
                    ax2.set_ylim(0,1.1*np.max(x_pixel_values_to_be_tested_for_tramline))
                    plt.tight_layout()
                    plt.show()
                
                if debug:
                    debug_find_tramline_row = True
                else:
                    debug_find_tramline_row = False

                # We expect slightly different widths for each tramlines in the different CCDs
                if ccd == '3':
                    expected_tramline_width = 38
                elif ccd == '2':
                    expected_tramline_width = 44
                elif ccd == '1':
                    expected_tramline_width = 45
        
                if x_index == buffer[order][0]:
                    tramline_beginning = np.nan
                    tramline_ending = np.nan
                tramline_beginning, tramline_ending = find_tramline_beginning_and_ending(
                    x_index,
                    x_pixels_to_be_tested_for_tramline[above_threshold],
                    previous_beginning = tramline_beginning,
                    previous_ending = tramline_ending,
                    expected_tramline_width = expected_tramline_width,
                    debug = debug_find_tramline_row
                )
                
                if debug_find_tramline_row:
                    print(x_index, tramline_beginning, tramline_ending, tramline_ending-tramline_beginning)

                x_pixels_tramline = []
                if np.isfinite(tramline_beginning) & np.isfinite(tramline_ending):
                    x_pixels_tramline = np.arange(tramline_beginning,tramline_ending)
                    adjusted_order_pixel.append(x_index)
                    adjusted_order_beginning.append(tramline_beginning)
                    adjusted_order_ending.append(tramline_ending)

    adjusted_order_pixel     = np.array(adjusted_order_pixel)
    adjusted_order_beginning = np.array(adjusted_order_beginning)
    adjusted_order_ending    = np.array(adjusted_order_ending)

    order_beginning_fit = curve_fit(
        polynomial_function,
        adjusted_order_pixel,
        adjusted_order_beginning,
        p0 = [np.median(adjusted_order_beginning),1.3e-02,-2.8e-05,9.3e-11,-2.7e-14],
        bounds=([0,0,-1e-4,0,-6e13],[image_dimensions[0],0.125,0,3e-9,0])
    )[0]
    order_ending_fit = curve_fit(
        polynomial_function,
        adjusted_order_pixel,
        adjusted_order_ending,
        p0 = [np.median(adjusted_order_ending),1.3e-02,-2.8e-05,9.3e-11,-2.7e-14],
        bounds=([0,0,-1e-4,0,-6e13],[image_dimensions[0],0.125,0,3e-9,0])
    )[0]
    
    if overwrite:
        np.savetxt('./VeloceReduction/tramline_information/tramlines_begin_end_'+order+'.txt',
                np.array([
                    ['#c0', 'c1', 'c2', 'c3', 'c4','buffer_pixel'],
                    np.concatenate((order_beginning_fit,[buffer[order][0]])),
                    np.concatenate((order_ending_fit,[buffer[order][1]]))
                ]),
                fmt='%s')
    else:
        try:
            old_order_beginning, old_order_ending = np.loadtxt('./VeloceReduction/tramline_information/tramlines_begin_end_'+order+'.txt')
            old_buffer = [old_order_beginning[-1],old_order_ending[-1]]
            old_order_beginning = old_order_beginning[:-1]
            old_order_ending = old_order_ending[:-1]

            print('Old vs New:')
            print('Buffer:')
            print(old_buffer, buffer[order])
            print('Beginning:')
            print(old_order_beginning, order_beginning_fit)
            print('Ending:')
            print(old_order_ending, order_ending_fit)

        except:
            print('No old tramline information found for '+order)

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

    try:
        ax.set_xlim(
            np.nanmax([np.nanmin([np.nanmin(order_xrange_begin),np.nanmin(adjusted_order_beginning)]) - 20,-20]), 
            np.nanmin([np.nanmax([np.nanmax(order_xrange_end),np.nanmax(adjusted_order_ending)]) + 20, image_dimensions[1] + 20])
            )
    except:
        pass

    print(order)
    print(order_beginning_fit)
    print(order_ending_fit)
    
    plt.xlabel('X Pixels (Zoom)',fontsize=15)
    plt.ylabel('Y Pixels',fontsize=15)
    plt.legend(loc = 'upper left',fontsize=15)
    if (order == 'ccd_1_order_141') & overwrite:
        plt.savefig('joss_paper/tramline_extraction_example_'+order+'.png',dpi=100,bbox_inches='tight')
    plt.show()
    plt.close()
    
    return(order_beginning_fit, order_ending_fit)