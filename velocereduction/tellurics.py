import numpy as np


def interval_mask(wavelength_nm, intervals):
    wavelength_nm = np.asarray(wavelength_nm, float); mask = np.zeros(wavelength_nm.shape, bool)
    for lo, hi in intervals: mask |= (wavelength_nm >= lo) & (wavelength_nm <= hi)
    return mask


def apply_mask(flux, variance, mask):
    flux, variance = np.asarray(flux, float).copy(), np.asarray(variance, float).copy()
    flux[mask] = np.nan; variance[mask] = np.nan
    return flux, variance
