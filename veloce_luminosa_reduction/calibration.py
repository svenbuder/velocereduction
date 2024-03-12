import numpy as np

def get_wavelength_coeffs_from_vdarc():
    
    wavelength_coeffs = dict()
    with open('../veloce_luminosa_reduction/veloce_reference_data/vdarc_azzurro_230915.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if line[:5] == 'START':
                order = int(line[6:-1])
            if line[:6] == 'COEFFS':
                wavelength_coeffs['ccd_1_order_'+str(order)] = np.array([float(coeff) for coeff in line[7:-1].split(' ')])
            line = fp.readline()
            cnt += 1

    with open('../veloce_luminosa_reduction/veloce_reference_data/vdarc_verde_230920.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if line[:5] == 'START':
                order = int(line[6:-1])
            if line[:6] == 'COEFFS':
                wavelength_coeffs['ccd_2_order_'+str(order)] = np.array([float(coeff) for coeff in line[7:-1].split(' ')])
            line = fp.readline()
            cnt += 1

    with open('../veloce_luminosa_reduction/veloce_reference_data/vdarc_rosso_230919.txt') as fp:
        line = fp.readline()
        cnt = 0
        while line:
            if line[:5] == 'START':
                order = int(line[6:-1])
            if line[:6] == 'COEFFS':
                wavelength_coeffs['ccd_3_order_'+str(order)] = np.array([float(coeff) for coeff in line[7:-1].split(' ')])
            line = fp.readline()
            cnt += 1
    return(wavelength_coeffs)
