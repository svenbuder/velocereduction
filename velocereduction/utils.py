from math import comb
import numpy as np

from .constants import (
    SLIT_ORDER as slit_order, SCIENCE_FIBRES as science_fibres,
    SKY_FIBRES as sky_fibres, FIBRE_COMPONENTS as science_sky_slit_order,
    C_ANGSTROM_GHZ, LC_REPEAT_GHZ, LC_OFFSET_GHZ,
)
from .config import ReductionConfig, ReductionPaths, prepare_reduction, setup_logging


def is_science_fibre(fibre):
    return isinstance(fibre, int)


def is_sky_fibre(fibre):
    return isinstance(fibre, str) and fibre.startswith("S")


fibre_to_index = {f: i for i, f in enumerate(science_sky_slit_order)}


def robust_sigma(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return 1.4826 * np.nanmedian(np.abs(values - np.nanmedian(values)))


def shifted_coefficients(coeffs, dx=0.0, dy=0.0):
    coeffs = np.asarray(coeffs, float)
    shifted = np.zeros_like(coeffs)
    for k, ck in enumerate(coeffs):
        for j in range(k + 1):
            shifted[j] += ck * comb(k, j) * (-dx) ** (k - j)
    shifted[0] += dy
    return shifted


def lasercomb_wavelength_from_numbers(n, repeat_frequency_ghz=LC_REPEAT_GHZ, offset_frequency_ghz=LC_OFFSET_GHZ):
    return C_ANGSTROM_GHZ / (np.asarray(n) * repeat_frequency_ghz + offset_frequency_ghz)


def lasercomb_numbers_from_wavelength(wavelength_angstrom, repeat_frequency_ghz=LC_REPEAT_GHZ, offset_frequency_ghz=LC_OFFSET_GHZ):
    return (C_ANGSTROM_GHZ / np.asarray(wavelength_angstrom) - offset_frequency_ghz) / repeat_frequency_ghz


def wavelength_vac_to_air(wavelength_vac):
    w = np.asarray(wavelength_vac, float)
    return w / (1 + 0.0000834254 + 0.02406147 / (130 - (1e4 / w) ** 2) + 0.00015998 / (38.9 - (1e4 / w) ** 2))


def wavelength_air_to_vac(wavelength_air):
    w = np.asarray(wavelength_air, float)
    return w * (1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - (1e4 / w) ** 2)
                + 0.0001599740894897 / (38.92568793293 - (1e4 / w) ** 2))
