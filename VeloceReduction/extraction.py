import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from . import config
from .utils import read_veloce_fits_image_and_metadata, match_month_to_date, polynomial_function

def substract_overscan(full_image, metadata, debug_overscan = False):
    """
    Substract overscan from the full image.

    :param full_image: The full image.
    :param metadata: The metadata of the image.
    :param debug_overscan: Whether to show debug plots.
    
    :return: The trimmed image, the median overscan, the overscan RMS, and the readout mode.
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

        trimmed_image = np.hstack([np.vstack([quadrant1,quadrant2]),np.vstack([quadrant4,quadrant3])]).clip(min=0)

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

        trimmed_image = np.hstack([quadrant1,quadrant2]).clip(min=0)

    if debug_overscan:
        plt.figure(figsize=(10,10))
        s = plt.imshow(trimmed_image, vmin = -5, vmax = 100)
        plt.colorbar(s)
        plt.show()
        plt.close()
        
    if debug_overscan:
        print(overscan_median, overscan_rms, metadata['READOUT'])

    return(trimmed_image, overscan_median, overscan_rms, metadata['READOUT'])

def extract_initial_order_ranges_and_coeffs():
    """
    Extract the initial order ranges and coefficients from the reference data.

    :return: Dictionaries with initial order ranges and coefficients.
    """

    initial_order_ranges = dict()
    initial_order_coeffs = dict()

    with open('./VeloceReduction/veloce_reference_data/azzurro-th-m138-167-all.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if cnt % 4 == 0:
                split_lines = line[:-1].split(' ')
                order = int(split_lines[0])
                initial_order_ranges['ccd_1_order_'+ str(order)] = np.arange(int(split_lines[1]),int(split_lines[2]))
            if cnt % 4 == 1:
                initial_order_coeffs['ccd_1_order_'+ str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
            line = fp.readline()
            cnt += 1

    with open('./VeloceReduction/veloce_reference_data/verde-th-m104-139-all.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if cnt % 4 == 0:
                split_lines = line[:-1].split(' ')
                order = int(split_lines[0])
                initial_order_ranges['ccd_2_order_'+ str(order)] = np.arange(int(split_lines[1]), int(split_lines[2]))
            if cnt % 4 == 1:
                initial_order_coeffs['ccd_2_order_'+ str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
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
                initial_order_ranges['ccd_3_order_'+ str(order)] = np.arange(int(split_lines[1]), int(split_lines[2]))
            if cnt % 4 == 1:
                initial_order_coeffs['ccd_3_order_'+ str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
            line = fp.readline()
            cnt += 1

    # To Do: also read in information for laser comb.
    # with open('./VeloceReduction/veloce_reference_data/verde-lc-m104-135-all.txt') as fp:
    # with open('./VeloceReduction/veloce_reference_data/rosso-lc-m65-104-all.txt') as fp:

    return(initial_order_ranges, initial_order_coeffs)

def extract_orders(ccd1_runs, ccd2_runs, ccd3_runs, Flat = False, LC = False, Science = False, debug_tramlines = False, debug_overscan=False):
    """
    Extract the orders from the CCDs.

    :param ccd1_runs: The runs for CCD 1.
    :param ccd2_runs: The runs for CCD 2.
    :param ccd3_runs: The runs for CCD 3.
    :param Flat: Whether to extract the orders for the flat.
    :param LC: Whether to extract the orders for the laser comb.
    :param Science: Whether to extract the orders for the science.
    :param debug_tramlines: Whether to show debug plots.
    :param debug_overscan: Whether to show debug plots.

    :return: The counts in the orders and the noise in the orders. If Science is True, also the metadata.
    """
    
    # Extract initial order ranges and coefficients
    initial_order_ranges, initial_order_coeffs = extract_initial_order_ranges_and_coeffs()


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

    counts_in_orders = []
    noise_in_orders = []
    
    if debug_tramlines:
        plt.figure(figsize=(15,15))
        s = plt.imshow(images['ccd_2'], vmin = 1, vmax = 20, cmap='Greys',aspect=5)
        plt.colorbar(s)
    
    for order in initial_order_coeffs:
        ccd = order[4]

        # Identify the tramline ranges for each order
        # initial_order_ranges[order] are the initial orders reported by C.Tinney.
        left = -45
        right = 0
        if LC & (ccd == '3'):
            left = 0
            right = 10
        if LC & (ccd == '2'):
            left = 8
            right = 20
        order_xrange_begin = np.array(polynomial_function(np.arange(np.shape(images['ccd_'+str(ccd)])[0]),*initial_order_coeffs[order])+left,dtype=int)
        order_xrange_end   = np.array(polynomial_function(np.arange(np.shape(images['ccd_'+str(ccd)])[0]),*initial_order_coeffs[order])+right,dtype=int)

        if debug_tramlines:
            if ccd == '2':
                plt.plot(order_xrange_begin,np.arange(len(order_xrange_begin)),c='C0',lw=0.5)
                plt.plot(order_xrange_end,np.arange(len(order_xrange_begin)),c='C1',lw=0.5)
        
        # Save the flux from each tramlines in a row; give NaN values to regions without flux
        order_counts = np.zeros(np.shape(images['ccd_'+str(ccd)])[1]); order_counts[:] = np.nan
        order_noise = np.zeros(np.shape(images['ccd_'+str(ccd)])[1]); order_noise[:] = np.nan

        # Because of the extended overscan region in 4Amplifier readout mode, we have to adjust which region we are using the extract the orders from.
        if readout_mode == '2Amp':
            order_ranges_adjusted_for_readout_mode = initial_order_ranges[order]
        elif readout_mode == '4Amp':
            order_ranges_adjusted_for_readout_mode = initial_order_ranges[order][16:-16]
        else:
            raise ValueError('Cannot handle readout_mode other than 2Amp or 4Amp')

        for x_index, x in enumerate(order_ranges_adjusted_for_readout_mode):
            
            counts_in_pixels_to_be_summed = images['ccd_'+str(ccd)][x,order_xrange_begin[x_index]:order_xrange_end[x_index]]
            
            order_counts[initial_order_ranges[order][0] + x_index] = np.sum(counts_in_pixels_to_be_summed,axis=0)
            
            # We are making the quick assumption that the read noise is simply the maximum overscan RMS
            total_read_noise = np.max([os_rms[region] for region in os_rms.keys()])*np.sqrt(len(counts_in_pixels_to_be_summed))

            # For science: multiply read noise with nr of runs,
            # since we are coadding frames, rather than using median
            if Science:
                total_read_noise *= np.sqrt(len(runs))

            if debug_tramlines & Science & (x_index == 2000):
                print(order)
                print('x_index:      ',2000)
                print('sum(counts):  ',np.sum(counts_in_pixels_to_be_summed,axis=0))
                print('sqrt(counts): ',np.sqrt(np.sum(counts_in_pixels_to_be_summed,axis=0)))
                print('read noise:   ',read_noise)
                print('counts:       ',counts_in_pixels_to_be_summed)
                plt.figure()
                plt.title(order)
                plt.plot(counts_in_pixels_to_be_summed)
                plt.show()
                plt.close()
                
            # noise = sqrt(flux + pixel_read_noise**2 * nr of pixels * nr of exposures)
            order_noise[initial_order_ranges[order][0] + x_index] = np.sqrt(np.sum(counts_in_pixels_to_be_summed,axis=0) + total_read_noise**2)

        counts_in_orders.append(order_counts)
        noise_in_orders.append(order_noise)

    if debug_tramlines:
        plt.xlim(2500,4200)
        plt.ylim(2000,2500)
        plt.show()
        plt.close()
        
    if Science:
        return(np.array(counts_in_orders),np.array(noise_in_orders),metadata)
    else:
        return(np.array(counts_in_orders),np.array(noise_in_orders))


def find_tramline_beginning_and_ending(x_index, x_pixels, previous_beginning, previous_ending, expected_tramline_width = 38, tolerance=2, tolerance_to_previous=3, debug=False):
    """
    Find the tramline beginning and ending for a given position of x_pixels above a threshold.
    Basically: Identify significant gaps that we expect to be those between the main tramline and the outer fibres left and right of it.
    
    :param x_index: Index of the row we are currently investigating
    :param x_pixels: The x pixels, that is, pixels within the region of the tramline above a certain threshold
    :param previous_beginning: The beginning of the previous row.
    :param previous_ending: The ending of the previous row.
    :expected_tramline_width: Expected tramline width (typically 38-45)
    :param tolerance: The tolerance for identifying a "gap" (between main tramline and the outer fibres).
    :param tolerance_to_previous: The tolerance for difference to previous row's beginning/ending.
    :param debug: Whether to show debug plots/prints.

    :return: The main tramline beginning and ending as a 2-tuple.
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
    Optimise the tramline polynomial for beginning and ending for a given order and readout mode.

    :param overscan_subtracted_images: The overscan subtracted images.
    :param order: The order.
    :param readout_mode: The readout mode.
    :param overwrite: Whether to overwrite a potentially existing file
        VeloceReduction/tramline_information/tramlines_begin_end_'+order+'.txt
    :param debug: Whether to show debug plots.

    :return: The tramline beginning and ending polynomial fit coefficients as 2 arrays.
    """

    if readout_mode != '2Amp':
        raise ValueError('Can only handle 2Amp readout mode')

    ccd = order[4]
    
    adjusted_order_pixel = []
    adjusted_order_beginning = []
    adjusted_order_ending = []

    image_dimensions = np.shape(overscan_subtracted_images['ccd_'+str(ccd)])

    # Identify the rough (too wide) tramline ranges for each order
    # initial_order_ranges[order] are the initial orders reported by C.Tinney with slight adjustments.
    initial_order_ranges, initial_order_coeffs = extract_initial_order_ranges_and_coeffs()

    if ccd == '3':
        left = -65
        right = 10
    elif ccd == '2':
        left = -60
        right = 15
    elif ccd == '1':
        left = -65
        right = 15

    order_xrange_begin = np.array(polynomial_function(np.arange(np.shape(overscan_subtracted_images['ccd_'+str(ccd)])[0]),*initial_order_coeffs[order])+left,dtype=int)
    order_xrange_end   = np.array(polynomial_function(np.arange(np.shape(overscan_subtracted_images['ccd_'+str(ccd)])[0]),*initial_order_coeffs[order])+right,dtype=int)

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
        order_ranges_adjusted_for_readout_mode = initial_order_ranges[order]
    elif readout_mode == '4Amp':
        order_ranges_adjusted_for_readout_mode = initial_order_ranges[order][16:-16]
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
        np.savetxt('VeloceReduction/tramline_information/tramlines_begin_end_'+order+'.txt',
                np.array([
                    ['#c0', 'c1', 'c2', 'c3', 'c4','buffer_pixel'],
                    np.concatenate((order_beginning_fit,[buffer[order][0]])),
                    np.concatenate((order_ending_fit,[buffer[order][1]]))
                ]),
                fmt='%s')
    else:
        try:
            old_order_beginning, old_order_ending = np.loadtxt('VeloceReduction/tramline_information/tramlines_begin_end_'+order+'.txt')
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