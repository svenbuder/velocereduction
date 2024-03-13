import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from .utils import interpolate_spectrum

def interpolate_orders_and_merge(wavelengths, fluxes, uncertainties, linear_wavelengths):
    """Interpolate and merge echelle orders onto a linear wavelength grid.

    Parameters:
    - wavelengths: List of arrays, each with the wavelength scale of an order.
    - fluxes: List of arrays, each with the flux for an order.
    - uncertainties: List of arrays, each with the flux uncertainty for an order.
    - linear_wavelengths: Array of linearly spaced wavelengths for interpolation.

    Returns:
    - merged_flux: Array of the merged flux on the linear wavelength grid.
    - merged_uncertainty: Array of the merged flux uncertainty on the linear grid.
    """
    # Initialize arrays to store sum of weighted fluxes and sum of weights
    weighted_flux_sum = np.zeros_like(linear_wavelengths)
    weight_sum = np.zeros_like(linear_wavelengths)
    variance_sum = np.zeros_like(linear_wavelengths)

    for wave, flux, unc in zip(wavelengths, fluxes, uncertainties):
        # Create interpolation functions for flux and uncertainty
        flux_interp = interp1d(wave, flux, bounds_error=False, fill_value="extrapolate")
        unc_interp = interp1d(wave, unc, bounds_error=False, fill_value="extrapolate")

        # Interpolate flux and uncertainty onto the linear wavelength grid
        interp_flux = flux_interp(linear_wavelengths)
        interp_unc = unc_interp(linear_wavelengths)

        # Calculate weights (inverse square of uncertainty)
        weights = 1 / interp_unc**2

        # Sum the weighted fluxes and the weights for averaging
        weighted_flux_sum += interp_flux * weights
        weight_sum += weights

        # Propagate variance
        variance_sum += (interp_flux**2) * (interp_unc**2)

    # Calculate the merged flux and its uncertainty
    merged_flux = weighted_flux_sum / weight_sum
    merged_uncertainty = np.sqrt(variance_sum) / weight_sum

    return merged_flux, merged_uncertainty


def degrade_resolution_with_uncertainty(wavelength, flux, flux_uncertainty, original_resolution, target_resolution):
    """
    Degrade the spectral resolution of flux and estimate the resulting flux uncertainty.

    Parameters:
    wavelength (numpy.ndarray): Wavelength array of the spectrum.
    flux (numpy.ndarray): Flux array of the spectrum.
    flux_uncertainty (numpy.ndarray): Flux uncertainty array of the spectrum.
    original_resolution (float): Original spectral resolution.
    target_resolution (float): Target spectral resolution to degrade to.

    Returns:
    numpy.ndarray: The degraded flux array.
    numpy.ndarray: The estimated flux uncertainty of the degraded spectrum.
    """
    # Calculate the FWHM at the original and target resolutions
    fwhm_original = wavelength / original_resolution
    fwhm_target = wavelength / target_resolution
    
    # Calculate the required broadening in FWHM
    fwhm_diff = np.sqrt(np.maximum(fwhm_target**2 - fwhm_original**2, 0))
    
    # Convert FWHM difference to sigma for the Gaussian kernel
    sigma = fwhm_diff / (2.0 * np.sqrt(2 * np.log(2)))
    
    # Assume sigma varies across wavelength; use mean sigma for simplicity in this example
    mean_sigma = np.mean(sigma)
    
    # Convolve flux with Gaussian kernel to degrade resolution
    degraded_flux = gaussian_filter1d(flux, mean_sigma)
    
    # Convolve squared flux uncertainty with Gaussian kernel and take square root to estimate new uncertainty
    degraded_flux_uncertainty = np.sqrt(gaussian_filter1d(flux_uncertainty**2, mean_sigma))
    
    return degraded_flux, degraded_flux_uncertainty

def coadd_spectra(spectra, common_wavelength):
    """Co-add multiple spectra onto a common wavelength grid."""
    interpolated_fluxes = []
    total_weights = np.zeros_like(common_wavelength)

    for spec in spectra:
        wavelength, flux, flux_uncertainty = spec['wavelength'], spec['flux'], spec['flux_uncertainty']
        interpolated_flux = interpolate_spectrum(wavelength, flux, common_wavelength)
        weight = 1 / flux_uncertainty**2

        # Summing weighted fluxes and weights for later averaging
        interpolated_fluxes.append(interpolated_flux * weight)
        total_weights += weight

    # Averaging the weighted sum of fluxes by the total weight to get the final co-added flux
    coadded_flux = np.sum(interpolated_fluxes, axis=0) / total_weights
    coadded_flux_uncertainty = np.sqrt(1 / total_weights)

    return common_wavelength, coadded_flux, coadded_flux_uncertainty

