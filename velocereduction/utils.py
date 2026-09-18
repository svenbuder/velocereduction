from math import comb
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, stdtr

from .constants import (
    SLIT_ORDER as slit_order, SCIENCE_FIBRES as science_fibres,
    SKY_FIBRES as sky_fibres, FIBRE_COMPONENTS as science_sky_slit_order,
    C_ANGSTROM_GHZ, LC_REPEAT_GHZ, LC_OFFSET_GHZ,
)
from .config import ReductionConfig, ReductionPaths, prepare_reduction, setup_logging


GAUSSIAN_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def nanmedian_profile(data):
    """Column median without warnings for entirely invalid columns."""
    data = np.asarray(data, float)
    result = np.full(data.shape[1], np.nan)
    use = np.any(np.isfinite(data), axis=0)
    result[use] = np.nanmedian(data[:, use], axis=0)
    return result


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


# -----------------------------------------------------------------------------
# Pixel-integrated line-spread functions
# -----------------------------------------------------------------------------


def gaussian_sigma_from_fwhm(fwhm):
    return np.asarray(fwhm, float) / GAUSSIAN_FWHM_FACTOR


def pixel_integrated_gaussian(y, integrated_counts, y_center, sigma):
    """Gaussian with total ``integrated_counts`` integrated over unit pixels."""
    y = np.asarray(y, float)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.full_like(y, np.nan, dtype=float)
    lo = (y - 0.5 - float(y_center)) / sigma
    hi = (y + 0.5 - float(y_center)) / sigma
    return float(integrated_counts) * (ndtr(hi) - ndtr(lo))


def pixel_integrated_gaussian_lsf(offset, fwhm):
    """Unit-area pixel-integrated Gaussian parameterised by intrinsic FWHM."""
    offset = np.asarray(offset, float)
    sigma = float(gaussian_sigma_from_fwhm(fwhm))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.full_like(offset, np.nan, dtype=float)
    return ndtr((offset + 0.5) / sigma) - ndtr((offset - 0.5) / sigma)


def moffat_alpha_from_fwhm(fwhm, one_over_beta):
    """Return Moffat alpha for intrinsic FWHM and ``one_over_beta=1/beta``."""
    fwhm = float(fwhm)
    q = float(one_over_beta)
    if not np.isfinite(fwhm) or fwhm <= 0 or not np.isfinite(q) or q <= 0:
        return np.inf
    if q >= 2.0:  # beta <= 1/2 is not normalisable in one dimension.
        return np.nan
    return fwhm / (2.0 * np.sqrt(np.expm1(np.log(2.0) * q)))


def pixel_integrated_moffat_lsf(offset, fwhm, one_over_beta):
    """Unit-area pixel-integrated Moffat LSF.

    ``one_over_beta=0`` is evaluated as the exact Gaussian limit.  This makes
    the adopted SimLC parameterisation continuous and avoids very large beta
    values for nearly Gaussian profiles.
    """
    offset = np.asarray(offset, float)
    fwhm = float(fwhm)
    q = float(one_over_beta)
    if not np.isfinite(fwhm) or fwhm <= 0 or not np.isfinite(q) or q < 0 or q >= 2:
        return np.full_like(offset, np.nan, dtype=float)
    if q <= 1e-8:
        return pixel_integrated_gaussian_lsf(offset, fwhm)

    beta = 1.0 / q
    alpha = moffat_alpha_from_fwhm(fwhm, q)
    nu = 2.0 * beta - 1.0
    scale = alpha / np.sqrt(nu)
    return stdtr(nu, (offset + 0.5) / scale) - stdtr(nu, (offset - 0.5) / scale)


def core_wing_sigmas_from_fwhm(fwhm, wing_fraction, wing_sigma_ratio):
    """Convert mixture FWHM to core/wing sigmas for two concentric Gaussians.

    ``wing_fraction`` is the integrated-flux fraction in the broader Gaussian;
    ``wing_sigma_ratio`` is sigma_wing / sigma_core (>1).  The returned sigmas
    produce the requested *intrinsic mixture* FWHM.
    """
    fwhm = float(fwhm)
    q = float(wing_fraction)
    ratio = float(wing_sigma_ratio)
    if (
        not np.isfinite(fwhm) or fwhm <= 0
        or not np.isfinite(q) or q < 0 or q >= 1
        or not np.isfinite(ratio) or ratio <= 1
    ):
        return np.nan, np.nan
    if q <= 1e-10:
        sigma = fwhm / GAUSSIAN_FWHM_FACTOR
        return sigma, ratio * sigma

    # With sigma_core=1 the normalized continuous mixture at x is
    # (1-q) exp(-x^2/2) + q/ratio exp(-x^2/(2 ratio^2)), apart from a common
    # normalization.  Solve for the positive half-maximum location once.
    peak = (1.0 - q) + q / ratio

    def half_max(x):
        value = (
            (1.0 - q) * np.exp(-0.5 * x * x)
            + (q / ratio) * np.exp(-0.5 * (x / ratio) ** 2)
        )
        return value - 0.5 * peak

    x_half = brentq(half_max, 0.0, 10.0 * ratio)
    sigma_core = fwhm / (2.0 * x_half)
    return sigma_core, ratio * sigma_core


def pixel_integrated_core_wing_gaussians_lsf(
    offset, fwhm, wing_fraction, wing_sigma_ratio,
):
    """Unit-area pixel-integrated concentric core+wing Gaussian mixture."""
    offset = np.asarray(offset, float)
    q = float(wing_fraction)
    sigma_core, sigma_wing = core_wing_sigmas_from_fwhm(
        fwhm, q, wing_sigma_ratio
    )
    if not np.isfinite(sigma_core) or not np.isfinite(sigma_wing):
        return np.full_like(offset, np.nan, dtype=float)
    core = ndtr((offset + 0.5) / sigma_core) - ndtr((offset - 0.5) / sigma_core)
    wing = ndtr((offset + 0.5) / sigma_wing) - ndtr((offset - 0.5) / sigma_wing)
    return (1.0 - q) * core + q * wing


def pixel_integrated_lsf(
    offset,
    shape,
    *,
    fwhm,
    one_over_beta=0.0,
    wing_fraction=0.0,
    wing_sigma_ratio=2.0,
):
    """Evaluate one of the supported normalized pixel-integrated LSFs."""
    shape = str(shape).lower()
    if shape == "gaussian":
        return pixel_integrated_gaussian_lsf(offset, fwhm)
    if shape == "moffat":
        return pixel_integrated_moffat_lsf(offset, fwhm, one_over_beta)
    if shape in {"core_wing_gaussians", "core_wing", "core+wing"}:
        return pixel_integrated_core_wing_gaussians_lsf(
            offset, fwhm, wing_fraction, wing_sigma_ratio
        )
    raise ValueError(
        "shape must be 'moffat', 'gaussian', or 'core_wing_gaussians'"
    )


# -----------------------------------------------------------------------------
# Wavelength / comb utility functions retained for backwards compatibility
# -----------------------------------------------------------------------------


def lasercomb_wavelength_from_numbers(
    n, repeat_frequency_ghz=LC_REPEAT_GHZ, offset_frequency_ghz=LC_OFFSET_GHZ
):
    return C_ANGSTROM_GHZ / (
        np.asarray(n) * repeat_frequency_ghz + offset_frequency_ghz
    )


def lasercomb_numbers_from_wavelength(
    wavelength_angstrom,
    repeat_frequency_ghz=LC_REPEAT_GHZ,
    offset_frequency_ghz=LC_OFFSET_GHZ,
):
    return (
        C_ANGSTROM_GHZ / np.asarray(wavelength_angstrom) - offset_frequency_ghz
    ) / repeat_frequency_ghz


def wavelength_vac_to_air(wavelength_vac):
    w = np.asarray(wavelength_vac, float)
    return w / (
        1
        + 0.0000834254
        + 0.02406147 / (130 - (1e4 / w) ** 2)
        + 0.00015998 / (38.9 - (1e4 / w) ** 2)
    )


def wavelength_air_to_vac(wavelength_air):
    w = np.asarray(wavelength_air, float)
    return w * (
        1
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - (1e4 / w) ** 2)
        + 0.0001599740894897 / (38.92568793293 - (1e4 / w) ** 2)
    )
