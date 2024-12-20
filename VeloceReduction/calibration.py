import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy.optimize import curve_fit
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import config

from VeloceReduction.utils import polynomial_function, calculate_barycentric_velocity_correction, velocity_shift, fit_voigt_absorption_profile, radial_velocity_from_line_shift, voigt_absorption_profile


def calibrate_single_order(file, order, science_object, barycentric_velocity=None):
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

    wavelength_to_plot = file[order].data['WAVE_AIR']

    f, gs = plt.subplots(5,1,figsize=(15,10),sharex=True)
    
    if isinstance(file[0].header['VRAD'], float):
        wavelength_to_plot = velocity_shift(velocity_in_kms=file[0].header['VRAD'], wavelength_array=wavelength_to_plot)
        f.suptitle(config.date+' '+science_object+' '+file[order].header['EXTNAME']+ r' $v_\mathrm{rad} = '+str(file[0].header['VRAD'])+r' \pm '+str(file[0].header['E_VRAD'])+'\,\mathrm{km\,s^{-1}}$, $v_\mathrm{bary} = '+"{:.2f}".format(np.round(file[0].header['BARYVEL'],2))+'\,\mathrm{km\,s^{-1}}$')
    else:
        f.suptitle(config.date+' '+science_object+' '+file[order].header['EXTNAME']+ r' $v_\mathrm{rad} = N/A, $v_\mathrm{bary} = '+"{:.2f}".format(np.round(file[0].header['BARYVEL'],2))+'\,\mathrm{km\,s^{-1}}$')

    ax = gs[0]
    ax.plot(file[order].data['SCIENCE'], lw=1)
    ax.set_yscale('log')
    ax.set_ylabel('Science')

    ticks = np.arange(0,len(wavelength_to_plot),100)
    ax.set_xticks(ticks,labels=ticks,rotation=90)
    ax.set_xlim(ticks[0],ticks[-1])   
    ax2 = ax.twiny()
    ax2.set_xticks(ticks-ticks[0])
    ax2.set_xticklabels(np.round(wavelength_to_plot[ticks],1),rotation=90)
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

def calibrate_wavelength(science_object, correct_barycentric_velocity=True, fit_voigt_for_rv=True, create_overview_pdf=False):
    """
    The function to read in the FITS file of an extracted science_object
    and perform the wavelength calibration to overwrite placeholder wavelength arrays.
    
    :param science_object: The name of the science object.
    :param correct_barycentric_velocity: Whether to correct the wavelength for the barycentric velocity.
    :param fit_voigt_for_rv: Whether to fit a Voigt profile to the Halpha and CaII triplet lines to estimate the radial velocity.
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
            # ToDo: Check why the barycentric velocity correction seems to be exactly off in the wrong direction?! For now, we will just flip the sign.
            barycentric_velocity = -calculate_barycentric_velocity_correction(science_header=file[0].header)
            print('  -> Correcting for barycentric velocity: '+"{:.2f}".format(np.round(barycentric_velocity,2))+' km/s')
            file[0].header['BARYVEL'] = barycentric_velocity
        else:
            print('  -> Not correcting for barycentric velocity.')
            barycentric_velocity = None
            file[0].header['BARYVEL'] = 'None'

        # Now loop through the FITS file extensions aka Veloce orders to apply the wavelength calibration
        for order in range(1,len(file)):
            calibrate_single_order(file, order, science_object, barycentric_velocity=barycentric_velocity)

        # If requested, fit a Voigt profile to the Halpha and CaII triplet lines to estimate the radial velocity
        if fit_voigt_for_rv:
            print('  -> Estimating rough Radial Velocity from Halpha and CaII triplet')

            # Define the lines and orders to fit a Voigt profile to.
            # Only fit 4 at the moment, as they deliver the most reliable results of ~3 km/s of literature values for the tests done for 240219.
            lines_air_and_orders_for_rv = [
                [r'$\mathrm{H_\alpha}$ 6562.7970',6562.7970,97,'CCD_3_ORDER_93'],
                [r'$\mathrm{H_\alpha}$ 6562.7970',6562.7970,98,'CCD_3_ORDER_94'],
                #[r'CaII triplet 8498.0230',8498.0230,76,'CCD_3_ORDER_72'],
                #[r'CaII triplet 8542.0910',8542.0910,76,'CCD_3_ORDER_72'],
                [r'CaII triplet 8542.0910',8542.0910,75,'CCD_3_ORDER_71'],
                [r'CaII triplet 8662.1410',8662.1410,75,'CCD_3_ORDER_71'],
                #[r'CaII triplet 8662.1410',8662.1410,74,'CCD_3_ORDER_70']
            ]

            f, gs = plt.subplots(1,len(lines_air_and_orders_for_rv),figsize=(15,4),sharey=True)
            gs[0].set_ylabel('Flux')

            rv_estimates = []
            rv_estimates_upper = []
            rv_estimates_lower = []

            for index, (line_name, line_centre, order, order_name) in enumerate(lines_air_and_orders_for_rv):
                ax = gs[index]
                ax.set_title(line_name+r'$\,\mathrm{\AA}$'+'\n'+order_name.replace('_',' '))
                ax.set_xlabel(r'Wavelength $\lambda_\mathrm{air}~/~\mathrm{\AA}$')

                if file[order].header['EXTNAME'] != order_name:
                    print('  -> Warning: '+file[order].header['EXTNAME']+' != '+order_name)

                close_to_line_centre = (
                    (file[order].data['WAVE_AIR'] > line_centre - 10) &
                    (file[order].data['WAVE_AIR'] < line_centre + 10)
                )

                wavelength_to_fit = file[order].data['WAVE_AIR'][close_to_line_centre]
                flux_to_fit = file[order].data['SCIENCE'][close_to_line_centre]
                flux_to_fit /= np.nanpercentile(flux_to_fit,q=99)

                ax.plot(
                    wavelength_to_fit,
                    flux_to_fit,
                    c = 'C0',
                    zorder = 1,
                    label = 'Veloce'
                )

                voigt_profile_parameters, voigt_profile_covariances = fit_voigt_absorption_profile(
                    wavelength_to_fit,
                    flux_to_fit,
                    # initial_guess: [line_centre, amplitude, sigma, gamma]
                    initial_guess = [line_centre, 0.5, 0.5, 0.5]
                )

                rv_voigt = radial_velocity_from_line_shift(voigt_profile_parameters[0], line_centre)
                e_line_centre = np.sqrt(np.diag(voigt_profile_covariances))[0]
                rv_upper_voigt = radial_velocity_from_line_shift(voigt_profile_parameters[0]+e_line_centre, line_centre)
                rv_lower_voigt = radial_velocity_from_line_shift(voigt_profile_parameters[0]-e_line_centre, line_centre)

                rv_estimates.append(rv_voigt)
                rv_estimates_upper.append(rv_upper_voigt)
                rv_estimates_lower.append(rv_lower_voigt)

                ax.plot(
                    wavelength_to_fit,
                    voigt_absorption_profile(wavelength_to_fit, *voigt_profile_parameters),
                    c = 'C1',
                    label = 'Voigt\n'+str(np.round(rv_voigt,1))+' km/s'
                )

                ax.axvline(line_centre, lw = 1, ls = 'dashed', c = 'C3', zorder = 3)
                ax.legend(fontsize=8,handlelength=1)
                ax.set_ylim(-0.1,1.1)

            rv_mean = np.round(np.mean(rv_estimates),1)
            rv_std  = np.round(np.std(rv_estimates),1)
            rv_unc  = np.round(np.median(np.array(rv_estimates_upper)-np.array(rv_estimates_lower)),1)
            print(r'  -> $v_\mathrm{rad}  = '+str(rv_mean)+' \pm '+str(rv_std)+' \pm '+str(rv_unc)+r'\,\mathrm{km\,s^{-1}}$ (mean, scatter, uncertainty)')
            
            file[0].header['VRAD'] = rv_mean
            file[0].header['E_VRAD'] = rv_std

            plt.tight_layout()
            plt.show()
            plt.close()

    # Let's create an overview PDF if requested
    if create_overview_pdf:
        with PdfPages(input_output_directory+'/veloce_spectra_'+science_object+'_'+config.date+'_overview.pdf') as overview_pdf:
            with fits.open(input_output_directory+'/veloce_spectra_'+science_object+'_'+config.date+'.fits') as file:
                print('  -> Creating overview PDF. This may take some time for the '+str(len(file))+' orders.')
                for order in range(1,len(file)):
                    plot_wavelength_calibrated_order_data(order, science_object, file, overview_pdf)
    else:
        print('  -> Not creating overview PDF.')