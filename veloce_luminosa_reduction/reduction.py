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
        overscan[:32,:] = True
        overscan[:2088,:32] = True
        overscan[:32,-32:] = True
        overscan[:2088,-32:] = True
        overscan = full_image[overscan]

        overscan_median['q1'] = int(np.median(overscan))
        overscan_rms['q1'] = np.std(overscan - np.median(overscan))
        quadrant1 -= overscan_median['q1']

        # Quadrant 2: 2088: and :2112
        quadrant2 = np.array(full_image[32:-32,2112+32:],dtype=int)
        overscan = np.zeros(np.shape(full_image),dtype=bool)
        overscan[2088:2088+32,:2112] = True
        overscan[2088:,:32] = True
        overscan[-32:,:2112] = True
        overscan[2088:,2112:2112+32] = True
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
