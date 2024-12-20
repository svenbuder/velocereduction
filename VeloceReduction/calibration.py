import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy.optimize import curve_fit
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import config

from VeloceReduction.utils import polynomial_function, calculate_barycentric_velocity_correction, velocity_shift


def calibrate_single_order(file, order, science_object, overview_pdf=None, barycentric_velocity=None):
    """
    Calibrates a single order of the spectrum.

    :param order: The order to calibrate.
    :param overview_pdf: The PDF to save the overview plot to (if not None)
    :param barycentric_velocity: The barycentric velocity to correct the wavelength for (if not None)

    :return: None
    """

    order_name = file[order].header['EXTNAME'].lower()

    # Because of the different number of pixels for the 2Amp and 4Amp readout, we have to adjust the centre pixel
    # This is usually 2048 for 4Amp readout and 2064 for 2Amp readout.
    order_centre_pixel = int(len(file[order].data['WAVE_AIR'])/2)

    # try:

    # Use the initial pixel <-> wavelength information per order to fit a polynomial function to it.
    # Here wavelength is reported in vacuum and in units of nm, not Å.
    # So we will later multiply it with 10 to report wavelength in vacuum in Å.
    # We will also use a published form to convert from vacuum to air wavelength and report that for ease.
    thxe_pixels_and_wavelengths = np.array(np.loadtxt('./VeloceReduction/veloce_reference_data/thxe_pixels_and_positions/'+file[order].header['EXTNAME'].lower()+'_px_wl.txt'))
    wavelength_solution_vacuum_coefficients, x = curve_fit(polynomial_function,
        thxe_pixels_and_wavelengths[:,0] - order_centre_pixel,
        thxe_pixels_and_wavelengths[:,1],
        p0 = [np.median(thxe_pixels_and_wavelengths[:,1]), 0.05, 0.0, 0.0, 0.0]
    )

    wavelength_solution_vacuum = polynomial_function(np.arange(len(file[order].data['WAVE_VAC'])) - order_centre_pixel,*wavelength_solution_vacuum_coefficients)
    file[order].data['WAVE_VAC'] = wavelength_solution_vacuum*10. # report Å, not nm.

    if barycentric_velocity is not None:
        # Correct the wavelength for the barycentric velocity.
        file[order].data['WAVE_VAC'] = velocity_shift(velocity_in_kms=barycentric_velocity, wavelength_array=file[order].data['WAVE_VAC'])

    # Using conversion from Birch, K. P., & Downs, M. J. 1994, Metro, 31, 315
    # Consistent to the 2024 version of Korg (https://github.com/ajwheeler/Korg.jl)
    file[order].data['WAVE_AIR'] = file[order].data['WAVE_VAC'] / (1 + 0.0000834254 + 0.02406147 / (130 - (1e4/file[order].data['WAVE_VAC'])**2) + 0.00015998 / (38.9 - (1e4/file[order].data['WAVE_VAC'])**2))

    if overview_pdf is not None:
        plot_wavelength_calibrated_order_data(order, science_object, file, overview_pdf)

def plot_wavelength_calibrated_order_data(order, science_object, file, overview_pdf):
    """
    Plots the data for a single order with 5 panels:
    1. Science,
    2. Science signal-to-noise,
    3. Flat,
    4. ThXe emission lines for wavelength calibration, and
    5. Laser Comb (LC) for wavelength calibration.

    :param order: The order to plot.
    :param science_object: The name of the science object.
    :param file: The FITS file.
    :param pdf: The PDF to save the plot to.

    :return: None, but saves the plot as a new page in the provided pdf.
    """
    
    f, gs = plt.subplots(5,1,figsize=(15,10),sharex=True)
    f.suptitle(config.date+' '+science_object+' '+file[order].header['EXTNAME']+ ' Baryc. Vel. Correction: '+"{:.2f}".format(np.round(file[0].header['BARYVEL'],2))+' km/s')

    ax = gs[0]
    ax.plot(file[order].data['SCIENCE'], lw=1)
    ax.set_yscale('log')
    ax.set_ylabel('Science')

    ticks = np.arange(0,len(file[order].data['WAVE_AIR']),100)
    ax.set_xticks(ticks,labels=ticks,rotation=90)
    ax.set_xlim(ticks[0],ticks[-1])   
    ax2 = ax.twiny()
    ax2.set_xticks(ticks-ticks[0])
    ax2.set_xticklabels(np.round(file[order].data['WAVE_AIR'][ticks],1),rotation=90)
    ax2.set_xlabel(r'Air Wavelength $\lambda_\mathrm{air}~/~\mathrm{\AA}$')

    ax = gs[1]
    ax.plot(file[order].data['SCIENCE']/file[order].data['SCIENCE_NOISE'], lw=1)
    ax.set_ylabel('Science Signal-to-Noise')

    ax = gs[2]
    ax.plot(file[order].data['FLAT'], lw=1)
    ax.set_ylabel('Flat')

    ax = gs[3]
    ax.plot(file[order].data['THXE'], lw=1)
    ax.set_yscale('log')
    ax.set_ylabel('ThXe')

    ax = gs[4]
    ax.plot(file[order].data['LC'], lw=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('Lc')
    ax.set_xlabel('Pixel')
    ax.set_xticks(ticks,labels=ticks,rotation=90)

    overview_pdf.savefig()
    plt.close()

def calibrate_wavelength(science_object, correct_barycentric_velocity=True, create_overview_pdf=False):
    """
    The function to read in the FITS file of an extracted science_object
    and perform the wavelength calibration to overwrite placeholder wavelength arrays.
    
    :param science_object: The name of the science object.
    :param correct_barycentric_velocity: Whether to correct the wavelength for the barycentric velocity.
    :param create_overview_pdf: Whether to create an overview PDF of the reduced data.
    
    :return: None
    """

    print('Calibrating wavelength for '+science_object)

    # Directory where the reduced data for this science_object can be found
    input_output_directory = config.working_directory+'reduced_data/'+config.date+'/'+science_object

    # Read in FITS file and prepare it to be updated
    with fits.open(input_output_directory+'/veloce_spectra_'+science_object+'_'+config.date+'.fits', mode='update') as file:
    
        # Let user know if we are correcting for barycentric velocity and creating overview PDF
        if correct_barycentric_velocity:
            barycentric_velocity = calculate_barycentric_velocity_correction(science_header=file[0].header)
            print('  -> Correcting for barycentric velocity: '+"{:.2f}".format(np.round(barycentric_velocity,2))+' km/s')
            file[0].header['BARYVEL'] = barycentric_velocity
        else:
            print('  -> Not correcting for barycentric velocity.')
            barycentric_velocity = None
            file[0].header['BARYVEL'] = 'None'

        # Now loop through the FITS file extensions aka Veloce orders to apply the wavelength calibration (and potentially save an overview PDF).
        if create_overview_pdf:
            print('  -> Creating overview PDF.')
            with PdfPages(input_output_directory+'/veloce_spectra_'+science_object+'_'+config.date+'_overview.pdf') as pdf:
                for order in range(1,len(file)):
                    calibrate_single_order(file, order, science_object, overview_pdf=pdf, barycentric_velocity=barycentric_velocity)
        else:
            print('  -> Not creating overview PDF. This may take some time for the '+str(len(file))+' orders.')
            for order in range(1,len(file)):
                calibrate_single_order(file, order, science_object, overview_pdf=None, barycentric_velocity=barycentric_velocity)
