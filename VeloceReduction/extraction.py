import numpy as np
from . import config
from .utils import read_veloce_fits_image_and_metadata, match_month_to_date, polynomial_function

def substract_overscan(full_image,metadata):

    # Identify overscan region and subtract overscan while reporting median overscan and overscan root-mean-square
    overscan_median = dict()
    overscan_rms = dict()

    if metadata['READOUT'] == '4Amp':

        # Quadrant 1: :2120 and :2112
        quadrant1 = np.array(full_image[32:2120-32,32:2112-32],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[:32,:2112] = True
        overscan[:2120,:32] = True
        overscan[:32,2112:2112+32] = True
        overscan[:2120,2112:2112+32] = True
        overscan = full_image[overscan]

        overscan_median['q1'] = int(np.median(overscan))
        overscan_rms['q1'] = np.std(overscan - np.median(overscan))
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
        overscan_rms['q2'] = np.std(overscan - np.median(overscan))
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
        overscan_rms['q3'] = np.std(overscan - np.median(overscan))
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
        overscan_rms['q4'] = np.std(overscan - np.median(overscan))
        print(quadrant4)
        quadrant4 -= overscan_median['q4']
        print(quadrant4)

        science_image = np.hstack([np.vstack([quadrant1,quadrant2]),np.vstack([quadrant4,quadrant3])]).clip(min=0)

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
        overscan_rms['q1'] = np.std(overscan - np.median(overscan))
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
        overscan_rms['q2'] = np.std(overscan - np.median(overscan))
        quadrant2 = (quadrant2 - overscan_median['q2'])

        trimmed_image = np.hstack([quadrant1,quadrant2]).clip(min=0)

    # if config.debug:
    #     plt.figure(figsize=(10,10))
    #     s = plt.imshow(full_image,vmax=2200)
    #     plt.colorbar(s)
    #     plt.show()
    #     plt.close()

    #     plt.figure(figsize=(10,10))
    #     s = plt.imshow(trimmed_image,vmax=1200)
    #     plt.colorbar(s)
    #     plt.show()
    #     plt.close()

    return(trimmed_image, overscan_median, overscan_rms)

def extract_initial_order_ranges_and_coeffs():

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

def extract_orders(ccd1_runs, ccd2_runs, ccd3_runs, Flat = False, LC = False, Science = False, tramline_debug = False):
    
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
            trimmed_image, os_median, os_rms = substract_overscan(full_image, metadata)
            images['ccd_'+str(ccd)].append(trimmed_image)
        
        # For science: sum counts
        if Science:
            images['ccd_'+str(ccd)] = np.array(np.median(images['ccd_'+str(ccd)],axis=0),dtype=float)
        # For calibration: calculate median
        else:
            images['ccd_'+str(ccd)] = np.array(np.median(images['ccd_'+str(ccd)],axis=0),dtype=float)

        if Flat:
            images['ccd_'+str(ccd)] /= np.nanmax(images['ccd_'+str(ccd)])

    counts_in_orders = []
    noise_in_orders = []
    
    if tramline_debug:
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

        if tramline_debug:
            if ccd == '2':
                plt.plot(order_xrange_begin,np.arange(len(order_xrange_begin)),c='C0',lw=0.5)
                plt.plot(order_xrange_end,np.arange(len(order_xrange_begin)),c='C1',lw=0.5)
        
        # Save the flux from each tramlines in a row; give NaN values to regions without flux
        order_counts = np.zeros(np.shape(images['ccd_'+str(ccd)])[1]); order_counts[:] = np.nan
        order_noise = np.zeros(np.shape(images['ccd_'+str(ccd)])[1]); order_noise[:] = np.nan
        for x_index, x in enumerate(initial_order_ranges[order]):
            
            counts_in_pixels_to_be_summed = images['ccd_'+str(ccd)][x,order_xrange_begin[x_index]:order_xrange_end[x_index]]
            
            order_counts[initial_order_ranges[order][0] + x_index] = np.sum(counts_in_pixels_to_be_summed,axis=0)
            
            # We are making the quick assumption that the read noise is simply the maximum overscan RMS
            read_noise = np.max([os_rms[region] for region in os_rms.keys()])*np.sqrt(len(counts_in_pixels_to_be_summed))

            # For science: multiply read noise with nr of runs,
            # since we are coadding frames, rather than using median
            if Science:
                read_noise*np.sqrt(len(runs))

            if tramline_debug & Science & (x_index == 2000):
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
                
            # noise = sqrt(flux) + read_noise * nr of pixels * nr of runs/exposures
            order_noise[initial_order_ranges[order][0] + x_index] = np.sqrt(np.sum(counts_in_pixels_to_be_summed,axis=0)) + read_noise

        counts_in_orders.append(order_counts)
        noise_in_orders.append(order_noise)

    if tramline_debug:
        plt.xlim(2500,4200)
        plt.ylim(2000,2500)
        plt.show()
        plt.close()
        
    return(np.array(counts_in_orders),np.array(noise_in_orders))