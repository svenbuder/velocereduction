import numpy as np
from scipy.ndimage import gaussian_filter

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
