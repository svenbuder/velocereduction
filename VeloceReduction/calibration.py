import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy.optimize import curve_fit
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import config

from VeloceReduction.utils import polynomial_function, calculate_barycentric_velocity_correction, velocity_shift, fit_voigt_absorption_profile, radial_velocity_from_line_shift, voigt_absorption_profile


def calibrate_single_order(file, order, barycentric_velocity=None):
    """
    Calibrates a single spectral order by fitting a polynomial to the pixel-to-wavelength relationship 
    and optionally applying a barycentric velocity correction. The calibration updates the wavelength 
    data for both vacuum and air wavelengths directly within the provided FITS file object.

    Parameters:
        file (HDUList): An open FITS file object from astropy.io.fits, which should contain the spectral data.
                        The spectral data of each order should be accessible via indexing.
        order (int): The index of the order within the FITS file to calibrate. This specifies which HDU in the
                     FITS file is being calibrated.
        barycentric_velocity (float, optional): The barycentric radial velocity (in km/s) to correct for
                                                the Doppler shift due to Earth's motion. If provided, this velocity
                                                is used to adjust the calculated wavelengths for the motion of the Earth.
                                                If None, no correction is applied.

    Returns:
        None: The function modifies the 'file' object in-place, updating the wavelength data for the specified order.

    This function processes the spectral data by:
    - Determining the center pixel based on the number of pixels in the order, which varies by readout mode.
    - Loading a predefined pixel-to-wavelength calibration from a text file.
    - Fitting a polynomial to these data points to create a wavelength solution in vacuum.
    - Converting this vacuum wavelength solution to air wavelength using a standard conversion formula.
    - Optionally correcting for barycentric velocity if a value is provided.

    Note:
    - The polynomial fit and vacuum-to-air wavelength conversion rely on specific coefficients and formulae
      that are expected to be defined or included via external resources or modules.

    Example:
        >>> from astropy.io import fits
        >>> fits_file = fits.open('path_to_fits_file.fits')
        >>> calibrate_single_order(fits_file, 0, barycentric_velocity=30.5)
        >>> fits_file.close()  # Always close the FITS file after modification to ensure data integrity
    """

    # Because of the different number of pixels for the 2Amp and 4Amp readout, we have to adjust the centre pixel
    # This is usually 2048 for 4Amp readout and 2064 for 2Amp readout.
    order_centre_pixel = int(len(file[order].data['WAVE_AIR'])/2)
    
    # Use the initial pixel <-> wavelength information per order to fit a polynomial function to it.
    
    # Load calibration data for the order
    # Note: Wavelength is reported in vacuum and in units of nm, not Å.
    thxe_file_path = './VeloceReduction/veloce_reference_data/thxe_pixels_and_positions/' + file[order].header['EXTNAME'].lower() + '_px_wl.txt'
    thxe_pixels_and_wavelengths = np.array(np.loadtxt(thxe_file_path))
    
    # Fit a polynomial function to pixel and wavelength data
    wavelength_solution_vacuum_coefficients, _ = curve_fit(
        polynomial_function,
        thxe_pixels_and_wavelengths[:,0] - order_centre_pixel,
        thxe_pixels_and_wavelengths[:,1],
        p0=[np.median(thxe_pixels_and_wavelengths[:,1]), 0.05, 0.0, 0.0, 0.0]
    )
    
    # Calculate vacuum wavelengths and convert them to air wavelengths
    wavelength_solution_vacuum = polynomial_function(
        np.arange(len(file[order].data['WAVE_VAC'])) - order_centre_pixel,
        *wavelength_solution_vacuum_coefficients
    ) * 10  # Convert from nm to Å
    file[order].data['WAVE_VAC'] = wavelength_solution_vacuum

    if barycentric_velocity is not None:
        file[order].data['WAVE_VAC'] = velocity_shift(barycentric_velocity, file[order].data['WAVE_VAC'])

    # Using conversion from Birch, K. P., & Downs, M. J. 1994, Metro, 31, 315
    # Consistent to the 2024 version of Korg (https://github.com/ajwheeler/Korg.jl)
    file[order].data['WAVE_AIR'] = file[order].data['WAVE_VAC'] / (1 + 0.0000834254 + 0.02406147 / (130 - (1e4/file[order].data['WAVE_VAC'])**2) + 0.00015998 / (38.9 - (1e4/file[order].data['WAVE_VAC'])**2))

def plot_wavelength_calibrated_order_data(order, science_object, file, overview_pdf):
    """
    Generates a five-panel plot for a single calibrated spectral order and saves it to the provided PDF. The panels include:
    1. Science spectrum.
    2. Science signal-to-noise ratio.
    3. Flat-field data.
    4. ThXe emission lines used for wavelength calibration.
    5. Laser Comb (LC) data used for wavelength calibration.

    Parameters:
        order (int): The index of the spectral order to plot.
        science_object (str): The name of the science object associated with the data.
        file (HDUList): The FITS file object containing the spectral data and associated headers.
        overview_pdf (PdfPages): A PdfPages object where the generated plot will be saved as a new page.

    Returns:
        None: The function does not return anything but saves the generated plot as a new page in the provided PDF.

    This function performs the following:
    - Reads the air wavelength data and optionally applies a radial velocity correction based on the FITS header.
    - Creates a five-panel plot with shared x-axes showing:
        - Science spectrum with pixel and wavelength labels.
        - Signal-to-noise ratio of the science spectrum.
        - Flat-field spectrum.
        - ThXe calibration lamp spectrum (logarithmic scale).
        - Laser Comb calibration spectrum (logarithmic scale).
    - Saves the plot to the specified PDF and closes the plot to free resources.

    Example:
        >>> from matplotlib.backends.backend_pdf import PdfPages
        >>> from astropy.io import fits
        >>> fits_file = fits.open('path_to_fits_file.fits')
        >>> with PdfPages('overview.pdf') as pdf:
        ...     plot_wavelength_calibrated_order_data(0, 'HD12345', fits_file, pdf)
        >>> fits_file.close()
    """

    # Extract wavelength data to plot
    wavelength_to_plot = file[order].data['WAVE_AIR']

    # Create the figure and set up shared x-axis
    f, gs = plt.subplots(5,1,figsize=(15,10),sharex=True)
    
    # Plot title with radial and barycentric velocity information
    if isinstance(file[0].header['VRAD'], float):
        wavelength_to_plot = velocity_shift(velocity_in_kms=file[0].header['VRAD'], wavelength_array=wavelength_to_plot)
        f.suptitle(config.date+' '+science_object+' '+file[order].header['EXTNAME']+ r' $v_\mathrm{rad} = '+str(file[0].header['VRAD'])+r' \pm '+str(file[0].header['E_VRAD'])+'\,\mathrm{km\,s^{-1}}$, $v_\mathrm{bary} = '+"{:.2f}".format(np.round(file[0].header['BARYVEL'],2))+'\,\mathrm{km\,s^{-1}}$')
    else:
        f.suptitle(config.date+' '+science_object+' '+file[order].header['EXTNAME']+ r' $v_\mathrm{rad} = N/A, $v_\mathrm{bary} = '+"{:.2f}".format(np.round(file[0].header['BARYVEL'],2))+'\,\mathrm{km\,s^{-1}}$')

    # Panel 1: Science spectrum
    ax = gs[0]
    ax.plot(file[order].data['SCIENCE'], lw=1)
    ax.set_ylabel('Science')
    ax.set_ylim(-0.1, 1.2)
    ticks = np.arange(0, len(wavelength_to_plot), 100)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=90)
    ax.set_xlim(ticks[0], ticks[-1])
    ax2 = ax.twiny()
    ax2.set_xticks(ticks - ticks[0])
    ax2.set_xticklabels(np.round(wavelength_to_plot[ticks], 1), rotation=90)
    ax2.set_xlabel(r'Air Wavelength $\lambda_\mathrm{air}~/~\mathrm{\AA}$')

    # Panel 2: Signal-to-noise ratio
    ax = gs[1]
    ax.plot(file[order].data['SCIENCE'] / file[order].data['SCIENCE_NOISE'], lw=1)
    ax.set_ylabel('Science S/N')

    # Panel 3: Flat-field data
    ax = gs[2]
    ax.plot(file[order].data['FLAT'], lw=1)
    ax.set_ylabel('Flat')

    # Panel 4: ThXe emission lines
    ax = gs[3]
    ax.plot(file[order].data['THXE'], lw=1)
    ax.set_yscale('log')
    ax.set_ylabel('ThXe')

    # Panel 5: Laser Comb data
    ax = gs[4]
    ax.plot(file[order].data['LC'], lw=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('LC')
    ax.set_xlabel('Pixel')
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=90)

    # Save the plot to the provided PDF
    overview_pdf.savefig()
    plt.close()

def calibrate_wavelength(science_object, correct_barycentric_velocity=True, fit_voigt_for_rv=True, create_overview_pdf=False):
    """
    Calibrates the wavelength data for a given science object by applying a series of corrections and enhancements.
    This includes barycentric velocity correction, radial velocity estimation via Voigt profile fitting, and
    optionally, generating an overview PDF with plots of the calibrated data.

    Parameters:
        science_object (str): The identifier for the science object. This is used to locate the corresponding
                              FITS file and to label outputs appropriately.
        correct_barycentric_velocity (bool): If True, apply a correction for the barycentric velocity to the
                                             wavelength data based on header information.
        fit_voigt_for_rv (bool): If True, fits a Voigt profile to selected spectral lines (Halpha and CaII triplet)
                                 to estimate the radial velocity of the object.
        create_overview_pdf (bool): If True, generates a PDF file containing plots of the calibrated spectral data
                                    for each order in the FITS file.

    Returns:
        None: This function modifies the FITS files in-place and may generate output files (PDFs), but does not
              return any values.

    Workflow:
        - The function first identifies the appropriate directory and FITS file based on the provided science_object name.
        - If barycentric velocity correction is enabled, it calculates and applies this correction.
        - If Voigt profile fitting is enabled, it estimates the radial velocity for the object using selected spectral lines
          and updates the FITS header with these values.
        - Optionally, it can generate a comprehensive PDF overview of the calibrated spectral data.
        - The function handles updates to the FITS file in-place and ensures all changes are saved.

    Note:
        The function assumes that the directory structure and naming conventions are consistent with a predefined
        configuration, which should be verified in the actual implementation environment.

    Example:
        >>> calibrate_wavelength('HD12345', correct_barycentric_velocity=True, fit_voigt_for_rv=True, create_overview_pdf=True)
    """

    print('Calibrating wavelength for ' + science_object)

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
            calibrate_single_order(file, order, barycentric_velocity=barycentric_velocity)

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
            plt.savefig(input_output_directory+'/veloce_spectra_'+science_object+'_'+config.date+'_rough_rv_estimate.pdf',bbox_inches='tight')
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