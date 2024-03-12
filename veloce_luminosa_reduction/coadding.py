import numpy as np
from utils import interpolate_spectrum

def co_add_spectra(spectra, common_wavelength):
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
