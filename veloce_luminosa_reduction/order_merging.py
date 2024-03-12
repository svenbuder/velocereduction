import numpy as np
from scipy.interpolate import interp1d

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
