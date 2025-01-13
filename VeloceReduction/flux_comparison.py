from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from astropy.table import Table
from scipy.optimize import minimize
from scipy.signal import medfilt
import matplotlib.pyplot as plt

from VeloceReduction.utils import polynomial_function, wavelength_vac_to_air, apply_velocity_shift_to_wavelength_array, degrade_spectral_resolution

def read_korg_syntheses():
    """
    Read synthetic spectra synthesised with the synthesis tool Korg (https://github.com/ajwheeler/Korg.jl).
    The code is run in julia and we thus use spectra precomputed and provided as part of this repository.

    Returns:
        korg_spectra (astropy.table.Table) with columns:
        - wavelength_air (Angstrom): wavelength in air,
        - wavelength_vac (Angstrom): wavelength in vacuum,
        - flux_sun (array):      normalised flux for the Sun,
        - flux_arcturus (array): normalised flux for Arcturus,
        - flux_61cyga (array):   normalised flux for 61 Cyg A,
        - flux_hd22879 (array):  normalised flux for HD 22879,
        - flux_18sco (array):    normalised flux for 18 Sco

    Details on how to recreate the spectra can be found in korg_flux/calculate_korg_flux.ipynb.    
    In short, spectra are synthesised:
    - for the Sun, Arcturus, 61 Cyg A, HD 22879 and 18 Sco with stellar parameters (Teff, logg, [Fe/H], vmic, vsini)
      from Jofre et al. (2017, http://adsabs.harvard.edu/abs/2017A%26A...601A..38J) for the first 4 and
      from Soubiran et al. (2024, https://ui.adsabs.harvard.edu/abs/2024A&A...682A.145S) for 18 Sco,
    - on a wavelength grid 3590:0.01:9510 Angstrom and downgraded to resolution R=80,000,
    - the default linelist of Korg based on an extraction of lines for the Sun from the VALD database.
    
    Warning:
    The linelist is not including lines between 9000 and 9510Å.
    It is further not including lines that are not visible in the Sun, but possibly in the cooler stars.

    Stellar parameters:
    Sun:      Teff=5771, logg=4.44, [Fe/H]= 0.03, vmic=1.06, vsini=np.sqrt(1.6**2 +4.2**2)
    Acruturs: Teff=4286, logg=1.64, [Fe/H]=-0.52, vmic=1.58, vsini=np.sqrt(3.8**2 +5.0**2)
    61 Cyg A: Teff=4373, logg=4.63, [Fe/H]=-0.33, vmic=1.07, vsini=np.sqrt(0.0**2 +4.2**2)
    HD 22879: Teff=5868, logg=4.27, [Fe/H]=-0.86, vmic=1.05, vsini=np.sqrt(4.4**2 +5.4**2)
    18 Sco:   Teff=5824, logg=4.42, [Fe/H]= 0.03, vmic=1.00, vsini=np.sqrt(2.03**2+3.7**2)

    """
    korg_spectra = Table.read(Path(__file__).resolve().parent / 'korg_flux' / 'korg_flux_sun_arcturus_61cyga_hd22879_18sco_R80000_3590_0.01_9510AA.fits')
    return(korg_spectra)

def read_korg_normalisation_buffers():
    """
    Read the left and right pixel buffers that will be used for the order-by-order comparison of the synthetic spectra.

    Returns:
        buffer_dict (dict) with keys for orders in the form 'ccd_'+ccd+'_order_'+order.
        
    Each order has an 2-element array with the left and right buffer.
    Right buffers are negative and buffers are to be applied as: flux_before_buffering[buffer[0]:buffer[1]].
    """
    buffer_text = np.loadtxt(Path(__file__).resolve().parent / 'korg_flux' / 'korg_order_comparison_buffers.txt', dtype=str)
    normalisation_buffers = dict()
    for buffer in buffer_text:
        normalisation_buffers[buffer[0]] = [int(buffer[1]),int(buffer[2])]
    return(normalisation_buffers)

    
def normalise_veloce_flux_via_smoothed_ratio_to_korg_flux(veloce_wavelength, veloce_flux, korg_flux, normalisation_buffers, filter_kernel_size = 501, debug=False):
    """
    Normalise the Veloce flux by dividing it by the Korg flux.
    
    The normalising 5-order Chebychev function is fitted to a median-smoothed ratio of the Veloce and Korg fluxes and repeated after 2-sigma outlier rejection and buffering.
    It is then applied to the whole Veloce spectrum array.

    Parameters:
        veloce_wavelength (array): wavelength of the Veloce spectrum,
        veloce_flux (array): flux of the Veloce spectrum,
        korg_flux (array): flux of the Korg synthetic spectrum,
        normalisation_buffers (array): left and right pixel buffers for the order-by-order comparison of the synthetic spectra.
        filter_kernel_size (int): size of the median filter kernel to smooth the flux ratio. Default is 501.
        debug (bool): print/plot debug information.

    Returns:
        veloce_flux_normalised (array): normalised flux of the Veloce spectrum.
    """

    # Calculate the Flux ratio of Veloce / Korg
    flux_ratio = veloce_flux / korg_flux

    # Identify absorption peaks above Xth percentile and neglect first/last Y pixels
    absorption_pixels = np.zeros(len(flux_ratio), dtype=bool)
    absorption_pixels[np.isnan(absorption_pixels)] = True
    absorption_pixels[0:normalisation_buffers[0]] = True
    absorption_pixels[normalisation_buffers[1]:-1] = True

    # Apply a broad median filter to estimate a smoothed ratio
    smooth_wavelength = veloce_wavelength[~absorption_pixels]
    smooth_flux_ratio = medfilt(flux_ratio[~absorption_pixels], kernel_size=filter_kernel_size)

    # Neglect 1/2 filer sizes left and right for more robust estimate
    smooth_flux_ratio[:filter_kernel_size//2] = np.nan
    smooth_flux_ratio[-filter_kernel_size//2:] = np.nan
    smooth_wavelength = smooth_wavelength[np.isfinite(smooth_flux_ratio)]
    smooth_flux_ratio = smooth_flux_ratio[np.isfinite(smooth_flux_ratio)]

    # Initial Chebyshev polynomial fit
    chebychev_fit_initial = Chebyshev.fit(smooth_wavelength, smooth_flux_ratio, deg=5)

    # Calculate residuals and identify outliers
    initial_fit_values = chebychev_fit_initial(smooth_wavelength)
    residuals = smooth_flux_ratio - initial_fit_values
    std_residuals = np.std(residuals)
    outlier_mask = np.abs(residuals) > 2 * std_residuals

    # Exclude outliers for the second fit
    refined_wavelength = smooth_wavelength[~outlier_mask]
    refined_flux_ratio = smooth_flux_ratio[~outlier_mask]

    # Final Chebyshev polynomial fit
    chebychev_fit_refined = Chebyshev.fit(refined_wavelength, refined_flux_ratio, deg=5)

    # Use the Chebychev polynomial to normalise the Veluce Flux
    veloce_flux_normalised_with_korg = veloce_flux / chebychev_fit_refined(veloce_wavelength)

    # Plot the normalisation process if debug is enabled
    if debug:
        f, ax = plt.subplots(figsize=(15,3))

        ax.scatter(
            refined_wavelength,
            refined_flux_ratio,
            c = 'C3', s = 1, label = 'Points Used for Refined Normalisation'
        )
        ax.plot(
            veloce_wavelength,
            flux_ratio,
            c = 'C0', lw = 0.5, label = 'Flux Ratio'
        )
        ax.plot(
            smooth_wavelength,
            smooth_flux_ratio,
            c = 'C1', lw = 0.5, label = 'Smoothed Flux Ratio'
        )
        ax.plot(
            veloce_wavelength,
            chebychev_fit_refined(veloce_wavelength),
            c = 'C3', lw = 0.5, label = 'Chebychev Fit to Smoothed Flux Ratio'
        )
        ax.set_xlabel(r'Wavelength $\lambda_\mathrm{vac}~/~\mathrm{\AA}$')
        ax.set_ylabel('Flux Ratio\n'+r'$f_\mathrm{Veloce}/f_\mathrm{Korg}$')
        ax.set_ylim(0.5,1.5)
        ax.legend(ncol=4, loc = 'upper center')


    return(veloce_flux_normalised_with_korg)

def make_veloce_and_korg_spectrum_compatible(wavelength_coefficients,veloce_science_flux,radial_velocity,barycentric_velocity,korg_wavelength_vac,korg_flux, normalisation_buffers, telluric_line_wavelengths = None, telluric_line_fluxes = None, debug=False):
    """
    This function makes the Veloce and Korg spectra compatible. It does so by:
    - calculating the Veloce wavelength based on a given set of wavelength coefficients.
    - applying the radial velocity and barycentric velocity to the Veloce wavelength.
    - identifying the overlapping wavelength range between the Veloce and Korg spectra.
    - interpolating the Korg spectrum to the Veloce wavelength grid.
    - incorporate the provided telluric spectrum.
    - calling the normalisation function to apply a robust normalisation to the Veloce spectrum
    - returning the residuals between the Veloce and Korg spectra.

    Parameters:
        wavelength_coefficients (array): coefficients to apply to the wavelength solution of the Veloce spectrum,
        veloce_science_flux (array): flux of the Veloce spectrum,
        radial_velocity (float): radial velocity in km/s,
        barycentric_velocity (float): barycentric velocity in km/s,
        korg_wavelength_vac (array): vacuum wavelengths of the Korg synthetic spectra,
        korg_flux (array): flux of the Korg synthetic spectra,
        normalisation_buffers (array): left and right pixel buffers for the order-by-order comparison of the synthetic spectra,
        telluric_line_wavelengths (array): wavelengths of telluric lines,
        telluric_line_fluxes (array): fluxes of telluric lines,
        debug (bool): print/plot debug information.

    Returns:
        normalised_veloce_science_flux (array): normalised flux of the Veloce spectrum.

    """

    # Calculate the Veloce wavelength solution based on the given coefficients.
    veloce_wavelength_vac = polynomial_function(np.arange(4128)-2064, *wavelength_coefficients)*10

    # Avoid infinite fluxes in the Veloce spectrum.
    veloce_wavelength_vac = veloce_wavelength_vac[np.isfinite(veloce_science_flux)]
    veloce_science_flux = veloce_science_flux[np.isfinite(veloce_science_flux)]

    # Apply the radial velocity and barycentric velocity to the Veloce wavelength.
    veloce_wavelength_vac_rv_shifted = apply_velocity_shift_to_wavelength_array(
        radial_velocity+barycentric_velocity,
        veloce_wavelength_vac
    )

    # Calculate the Veloce wavelength in air (necessary for interpolation with telluric line spectra with air wavelengths).
    veloce_wavelength_air_rv_shifted = wavelength_vac_to_air(veloce_wavelength_vac_rv_shifted)

    # Identify the pixels of the Korg spectrum that overlap with the Veloce Order region
    relevant_korg_spectrum_region = (
        (korg_wavelength_vac > veloce_wavelength_vac_rv_shifted[0]  - 5) &
        (korg_wavelength_vac < veloce_wavelength_vac_rv_shifted[-1] + 5)
    )
    relevant_korg_spectrum_region[np.isnan(np.array(korg_flux))] = False
    if len(np.where(relevant_korg_spectrum_region)[0]) == 0:
        print('Korg:   ',korg_wavelength_vac[0],korg_wavelength_vac[-1])
        print('Needed: ',veloce_wavelength_vac_rv_shifted[0]  - 5,veloce_wavelength_vac_rv_shifted[-1] + 5)
        raise ValueError('No relevant Korg spectrum available')

    # Interpolate the Korg spectrum onto the Veloce wavelength grid
    korg_flux_interpolated = np.interp(
        veloce_wavelength_vac_rv_shifted,
        korg_wavelength_vac[relevant_korg_spectrum_region],
        np.array(korg_flux)[relevant_korg_spectrum_region]
    )

    # Incorporate telluric lines into the Korg spectrum, if they were provided.
    if telluric_line_fluxes is not None:

        # Test if the telluric spectrum has a wavelength array associated with it, or we can assume it is the same as the Veloce spectrum
        if telluric_line_wavelengths is not None:
            # Shift telluric lines from their presumably rest-wavlength onto the Veloce wavelength
            # We expect the wavelength to be in air.
            telluric_flux_interpolated = np.interp(
                veloce_wavelength_air_rv_shifted,
                apply_velocity_shift_to_wavelength_array(
                    radial_velocity+barycentric_velocity,
                    telluric_line_wavelengths
                ),
                telluric_line_fluxes,
                # assume no telluric absorption lines outside the expected region
                left = 1.0,
                right = 1.0
            )
        else:
            # In this case we assume that the telluric flux is taken via Veloce itself during the same night
            # and is already shifted and interpolated onto the science spectrum wavelength grid.
            telluric_flux_interpolated = telluric_line_fluxes

        # Multiply the Korg spectrum with the telluric spectrum
        korg_flux_interpolated *= telluric_flux_interpolated

    veloce_flux_normalised_with_korg_flux = normalise_veloce_flux_via_smoothed_ratio_to_korg_flux(veloce_wavelength_vac_rv_shifted, veloce_science_flux, korg_flux_interpolated, normalisation_buffers, debug=debug)

    # Visualise the Veloce and Korg spectra if debug is enabled
    if debug:
        f, ax = plt.subplots(figsize=(15,3))
            
        # Show Korg Flux with Tellurics
        if telluric_line_wavelengths is None:
            label = 'Korg Flux with Veloce Tellurics'
        else:
            label = 'Korg Flux with Hinkle Tellurics'
        ax.plot(
            veloce_wavelength_vac_rv_shifted,
            korg_flux_interpolated,
            c = 'C1', lw = 0.5, label = label
        )

        # Show Telluric Flux
        if telluric_line_fluxes is not None:
            ax.plot(
                veloce_wavelength_air_rv_shifted,
                telluric_flux_interpolated,
                c = 'C2', lw = 0.5, label = 'Telluric Flux'
            )

        # Show Veloce Flux
        ax.plot(
            veloce_wavelength_vac_rv_shifted,
            veloce_flux_normalised_with_korg_flux,
            c = 'C0', lw = 0.5, label = 'Normalised Veloce Flux'
        )
        
        # Residuals
        ax.plot(
            veloce_wavelength_vac_rv_shifted,
            korg_flux_interpolated - veloce_flux_normalised_with_korg_flux,
            c = 'C4', lw = 0.5, label = 'Residuals Korg - Veloce'
        )

        ax.set_xlabel(r'Wavelength $\lambda_\mathrm{vac}~/~\mathrm{\AA}$')
        ax.set_ylabel('Flux\n'+r'$f_\lambda~/~\mathrm{norm.}$')
        ax.legend(ncol=4, loc = 'upper center')
        ax.set_ylim(-0.1,1.25)
        plt.tight_layout()
        plt.show()
        plt.close()

    return(veloce_flux_normalised_with_korg_flux, korg_flux_interpolated)

def calculate_absolute_residual_sum_between_veloce_and_korg_spectrum(wavelength_coefficients, veloce_science_flux, barycentric_velocity, radial_velocity, korg_wavelength_vac, korg_flux, normalisation_buffers, telluric_line_wavelengths = None, telluric_line_fluxes = None, debug=False):
    """
    This function provides a scipy.optimise function with the 1 value that will be optimised.

    Here, we calculate the sum of absolute residuals between the Veloce and Korg spectra.

    Parameters:
        wavelength_coefficients (array): coefficients to apply to the wavelength solution of the Veloce spectrum,
        veloce_science_flux (array): flux of the Veloce spectrum,
        barycentric_velocity (float): barycentric velocity in km/s,
        radial_velocity (float): radial velocity in km/s,
        korg_wavelength_vac (array): vacuum wavelengths of the Korg synthetic spectra,
        korg_flux (array): flux of the Korg synthetic spectra,
        normalisation_buffers (array): left and right pixel buffers for the order-by-order comparison of the synthetic spectra,
        telluric_line_wavelengths (array): wavelengths of telluric lines, default is None (assuming telluric lines would be on same wavelength as veloce_science_flux),
        telluric_line_fluxes (array): fluxes of telluric lines, default is None (assuming no telluric spectrum will be used),
        debug (bool): print/plot debug information.

    Returns:
        sum_abs_residuals (float): sum of absolute residuals between the Veloce and Korg spectra.
    
    """
    
    # Make Veloce and Korg fluxes compatible: normalised and interpolated onto the same wavelength array.
    normalised_veloce_science_flux, korg_flux_interpolated = make_veloce_and_korg_spectrum_compatible(
        wavelength_coefficients,
        veloce_science_flux,
        radial_velocity,
        barycentric_velocity,
        korg_wavelength_vac,
        korg_flux,
        normalisation_buffers,
        telluric_line_wavelengths,
        telluric_line_fluxes,
        debug
    )

    # Now calculate the sum of the absolute residuals between the Veloce and Korg spectra within the bugger regions.
    sum_abs_residuals = np.sum(np.abs(normalised_veloce_science_flux - korg_flux_interpolated)[normalisation_buffers[0]:normalisation_buffers[1]])

    if debug:
        print(sum_abs_residuals, wavelength_coefficients)

    return(sum_abs_residuals)

def fit_wavelength_solution_with_korg_spectrum(order, veloce_fits_file, radial_velocity, korg_wavelength_vac, korg_flux, normalisation_buffers, telluric_line_wavelengths = None, telluric_line_fluxes = None, debug=False):
    """
    Fit the wavelength solution of a Veloce spectrum to a Korg synthetic spectrum.

    Parameters:
        order (str): order to be fitted,
        veloce_fits_file (str): path to the Veloce spectrum to be compared.
        radial_velocity (float): radial velocity in km/s.
        korg_wavelength_vac (array): vacuum wavelengths of the Korg synthetic spectra.
        korg_flux (array): flux of the Korg synthetic spectra.
        normalisation_buffers (array): array with left and right pixel buffers for the order-by-order comparison of the synthetic spectra, e.g. [200:-200].
        telluric_line_wavelengths (array): wavelengths of telluric lines, default is None (assuming telluric lines would be on same wavelength as veloce_science_flux).
        telluric_line_fluxes (array): fluxes of telluric lines, default is None (assuming no telluric spectrum will be used).
        debug (bool): print/plot debug information.

    Returns:

    """

    if debug:
        print(f'  -> Fitting wavelength solution for order {order}.')

    # We will use the science spectrum and barycentric velocity from the veloce_fits_file:
    veloce_science_flux  = veloce_fits_file[order].data['science']
    barycentric_velocity = veloce_fits_file[0].header['BARYVEL']

    # If telluric lines have been observed for the given science spectrum via BStars, we will use them.
    if 'telluric' in veloce_fits_file[order].data.columns.names:
        telluric_line_wavelengths = None # They will are already on the same wavelength as the science spectrum
        telluric_line_fluxes = veloce_fits_file[order].data['telluric']

    # Use wavelength coefficients according to the following preference:
    # 1) Coefficients fitted with 18Sco and Korg synthesis
    # 2) Coefficients fitted with LC
    # 3) Coefficients fitted with ThXe
    try:
        initial_wavelength_coefficients = np.loadtxt(Path(__file__).resolve().parent / 'wavelength_coefficients' / f'wavelength_coefficients_{order}_korg.txt')
        if debug:
            print('  -> Using initial solution from Korg: ')
    except:
        try:
            initial_wavelength_coefficients = np.loadtxt(Path(__file__).resolve().parent / 'wavelength_coefficients' / f'wavelength_coefficients_{order}_lc.txt')
            if debug:
                print('  -> Using initial solution from LC:')
        except:
            initial_wavelength_coefficients = np.loadtxt(Path(__file__).resolve().parent / 'wavelength_coefficients' / f'wavelength_coefficients_{order}_thxe.txt')
            if debug:
                print('  -> Using initial solution from ThXe:')
    if debug:
        print(initial_wavelength_coefficients)

    wavelength_coefficients_minimum = minimize(
        calculate_absolute_residual_sum_between_veloce_and_korg_spectrum,
        initial_wavelength_coefficients,
        bounds = [
            # Allow wiggle-room within roughly 2% of wavelength solution (except for the centre wavelength, which we know within 2Å)
            (initial_wavelength_coefficients[0] - 0.2  , initial_wavelength_coefficients[0] + 0.2),
            (initial_wavelength_coefficients[1] - 2e-05, initial_wavelength_coefficients[1] + 2e-05),
            (initial_wavelength_coefficients[2] - 2e-09, initial_wavelength_coefficients[2] + 2e-09),
            (initial_wavelength_coefficients[3] - 2e-13, initial_wavelength_coefficients[3] + 2e-13),
            (initial_wavelength_coefficients[4] - 2e-15, initial_wavelength_coefficients[4] + 2e-15),
            (initial_wavelength_coefficients[5] - 2e-17, initial_wavelength_coefficients[5] + 2e-17)
        ],
        args=(veloce_science_flux, barycentric_velocity, radial_velocity, korg_wavelength_vac, korg_flux, normalisation_buffers, telluric_line_wavelengths, telluric_line_fluxes, debug)
    )

    # Once the fitting is finished, plot the Korg and Veloce spectra and how the latter was normalised
    normalised_veloce_science_flux, korg_flux_interpolated = make_veloce_and_korg_spectrum_compatible(
        wavelength_coefficients_minimum.x,
        veloce_science_flux,
        radial_velocity,
        barycentric_velocity,
        korg_wavelength_vac,
        korg_flux,
        normalisation_buffers,
        telluric_line_wavelengths,
        telluric_line_fluxes,
        debug=True
    )

def calculate_wavelength_coefficients_with_korg_synthesis(veloce_fits_file, korg_wavelength_vac, korg_flux, order_selection=None, enforce_vrad=None, debug=False):
    """
    Calculate the wavelength coefficients to match the Veloce spectra to the Korg synthetic spectra.

    Parameters:
        veloce_fits_file (str): path to the Veloce spectrum to be compared,
        korg_wavelength_vac (array): vacuum wavelengths of the Korg synthetic spectra,
        korg_flux (array): flux of the Korg synthetic spectra,
        order_selection (array): array to select orders to be used in the comparison,
        enforce_vrad (float): enforce a radial velocity shift in km/s.
        debug (bool): print/plot debug information.

    Returns:
        wavelength_coefficients (array): coefficients to apply to the wavelength solution of the Veloce spectrum.
    """

    if enforce_vrad is not None:
        radial_velocity = enforce_vrad
        print(f'Enforcing a radial velocity shift of {radial_velocity} km/s.')
    else:
        radial_velocity = veloce_fits_file[0].header['VRAD']
        print(f'Enforcing a radial velocity shift of {radial_velocity} km/s.')

    # To avoid reading in the archival telluric spectrum for each order, we will do it once here.
    # Unless we use the telluric spectrum observed, then we can simply skip this step.
    if 'telluric' not in veloce_fits_file[1].data.columns.names:
        hinkle_atlas = Table.read(Path(__file__).resolve().parent / 'hinkle_2000_atlas' / 'hinkle_2000_solar_arcturus_telluric_atlas.fits')
        telluric_line_wavelengths = np.array(hinkle_atlas['WAVELENGTH'])
        telluric_line_fluxes = degrade_spectral_resolution(
            telluric_line_wavelengths,
            np.array(hinkle_atlas['TELLURIC']),
            original_resolution = 150000.,
            target_resolution = 80000.
        )
    else:
        telluric_line_wavelengths = None
        telluric_line_fluxes = None
        
    if order_selection is not None:
        orders = order_selection
        print('Calculating coefficients for orders '+','.join(orders))
    else:
        orders = []
        for ccd in ['1','2','3']:
            if ccd == '1': orders.append(['ccd_1_order_'+str(x) for x in np.arange(167, 138-1, -1)])
            if ccd == '2': orders.append(['ccd_2_order_'+str(x) for x in np.arange(140, 103-1, -1)])
            if ccd == '3': orders.append(['ccd_3_order_'+str(x) for x in np.arange(104,  65-1, -1)])
        orders = np.concatenate((orders))
        print('Calculating coefficients for all orders (138-167 for CCD1, 103-140 for CCD2, and 65-104 for CCD3).')

    # Read the dictionary of left and right buffers that will be used when calculating the normalisation function between Veloce and Korg spectrum.
    normalisation_buffers = read_korg_normalisation_buffers()

    # Loop over all orders and fit the wavelength solution to the Korg synthetic spectrum.
    for order in orders:
        fit_wavelength_solution_with_korg_spectrum(order, veloce_fits_file, radial_velocity, korg_wavelength_vac, korg_flux, normalisation_buffers[order], telluric_line_wavelengths, telluric_line_fluxes, debug=debug)