from . import config

import numpy as np

if config.debug:
    import matplotlib.pyplot as plt

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

    with open('../veloce_luminosa_reduction/veloce_reference_data/azzurro-th-m138-167-all.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if cnt % 4 == 0:
                split_lines = line[:-1].split(' ')
                order = int(split_lines[0])
                initial_order_ranges['ccd_1_order_'+str(order)] = np.arange(int(split_lines[1]),int(split_lines[2]))
            if cnt % 4 == 1:
                initial_order_coeffs['ccd_1_order_'+str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
            line = fp.readline()
            cnt += 1

    with open('../veloce_luminosa_reduction/veloce_reference_data/verde-th-m104-139-all.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if cnt % 4 == 0:
                split_lines = line[:-1].split(' ')
                order = int(split_lines[0])
                initial_order_ranges['ccd_2_order_'+str(order)] = np.arange(int(split_lines[1]), int(split_lines[2]))
            if cnt % 4 == 1:
                initial_order_coeffs['ccd_2_order_'+str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
            line = fp.readline()
            cnt += 1

    with open('../veloce_luminosa_reduction/veloce_reference_data/rosso-lc-m65-104-all.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if cnt % 4 == 0:
                split_lines = line[:-1].split(' ')
                order = int(split_lines[0])
                initial_order_ranges['ccd_3_order_'+str(order)] = np.arange(int(split_lines[1]), int(split_lines[2]))
            if cnt % 4 == 1:
                initial_order_coeffs['ccd_3_order_'+str(order)] = [float(coeff) for coeff in line[10:-1].split(' ')]
            line = fp.readline()
            cnt += 1

    return(initial_order_ranges, initial_order_coeffs)

