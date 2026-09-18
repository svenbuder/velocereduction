"""SimLC line measurement with a smooth, spatially varying LSF.

The SimLC workflow deliberately separates line detection from precision
centroiding:

1. detect/seed every usable peak down to S/N=3 with the generic integrated
   Gaussian calibration fitter;
2. fit the requested LSF shape independently to the detected peaks to obtain
   noisy local shape measurements and their uncertainties;
3. infer an uncertainty-aware smooth FWHM(y, m) field and smooth order-only
   dimensionless shape parameters;
4. remeasure every detected line centre with that smooth LSF fixed;
5. assign exact comb modes from the static wavelength reference using the
   optimized centroids.

The production default is ``lsf_shape='moffat'``.  ``'gaussian'`` and
``'core_wing_gaussians'`` are retained as useful alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits
from astropy.table import Table
from numpy.polynomial.legendre import legval, legval2d, legvander
from scipy.optimize import least_squares

from .calibration import (
    CalibrationLineSet,
    CalibrationPeakConfig,
    CalibrationPeakFlag,
    calibration_quality_summary,
    _clear_identification,
    _match_peak_table_to_predicted_lines,
    _predict_reference_lines_for_order,
    _refresh_used_for_wavelength_fit,
    measure_calibration_peaks,
)
from .models import ExtractedExposure, LineSpreadFunctionModel
from .utils import pixel_integrated_lsf, robust_sigma
from .wavelength import WAVELENGTH_Y_BOUNDS, read_wavelength_solution_fits


SIMLC_REPEAT_FREQUENCY_HZ = 25.0e9
SIMLC_OFFSET_FREQUENCY_HZ = 9.56e9
SIMLC_INITIAL_DETECTION_SNR = 5.0
SIMLC_DEFAULT_MINIMUM_SNR = 15.0
SPEED_OF_LIGHT_MPS = 299_792_458.0
SPEED_OF_LIGHT_NM_S = SPEED_OF_LIGHT_MPS * 1e9

SUPPORTED_LSF_SHAPES = ("moffat", "gaussian", "core_wing_gaussians")


@dataclass(frozen=True)
class SimLCLSFConfig:
    """Tunable settings for two-pass SimLC LSF inference."""

    initial_detection_snr: float = SIMLC_INITIAL_DETECTION_SNR
    fit_half_width: int = 3
    maximum_centroid_shift: float = 0.55

    # Local shape bounds.
    minimum_fwhm: float = 0.35
    maximum_fwhm: float = 3.0
    maximum_one_over_beta: float = 1.5
    maximum_wing_fraction: float = 0.70
    minimum_wing_sigma_ratio: float = 1.05
    maximum_wing_sigma_ratio: float = 4.0

    # Shape-model line selection.  Choose the highest threshold that still
    # gives enough lines and dispersion-direction coverage in an order.
    shape_snr_grid: tuple[float, ...] = (
        100.0, 75.0, 50.0, 40.0, 30.0, 25.0, 20.0, 15.0,
        12.0, 10.0, 8.0, 6.0, 5.0
    )
    target_shape_lines_per_order: int = 20
    minimum_shape_lines_per_order: int = 10
    minimum_y_coverage_fraction: float = 0.65
    y_coverage_bins: int = 5
    minimum_populated_y_bins: int = 4
    minimum_neighbour_distance: float = 7.2

    # Smooth LSF field.  FWHM needs both y and order dependence; the remaining
    # dimensionless shape terms are deliberately smoother and order-only.
    fwhm_y_degree: int = 4
    fwhm_order_degree: int = 3
    order_parameter_degree: int = 3
    smooth_clip_sigma: float = 4.5
    smooth_max_iterations: int = 6

    # Final centroid quality; the final S/N cut is supplied explicitly to
    # measure_simlc_lines because it can vary substantially between nights.
    maximum_y_uncertainty: float = 0.30


# -----------------------------------------------------------------------------
# Comb frequencies and mode identification
# -----------------------------------------------------------------------------


def make_comb_reference_table(
    wavelength_min_nm,
    wavelength_max_nm,
    *,
    repetition_rate_hz=SIMLC_REPEAT_FREQUENCY_HZ,
    offset_frequency_hz=SIMLC_OFFSET_FREQUENCY_HZ,
):
    """Generate exact SimLC modes spanning a wavelength interval."""
    if repetition_rate_hz <= 0:
        raise ValueError("repetition_rate_hz must be positive")
    lo, hi = sorted((float(wavelength_min_nm), float(wavelength_max_nm)))
    fmin = SPEED_OF_LIGHT_NM_S / hi
    fmax = SPEED_OF_LIGHT_NM_S / lo
    nmin = int(np.ceil((fmin - offset_frequency_hz) / repetition_rate_hz))
    nmax = int(np.floor((fmax - offset_frequency_hz) / repetition_rate_hz))
    modes = np.arange(nmin, nmax + 1, dtype=np.int64)
    frequency = offset_frequency_hz + modes * repetition_rate_hz
    wavelength = SPEED_OF_LIGHT_NM_S / frequency
    idx = np.argsort(wavelength)
    modes, frequency, wavelength = modes[idx], frequency[idx], wavelength[idx]
    return Table(dict(
        reference_id=np.asarray([f"LC_{n:d}" for n in modes], dtype="U32"),
        wavelength_nm=wavelength.astype(float),
        wavelength_uncertainty_nm=np.zeros(len(modes), dtype=float),
        reference_intensity=np.full(len(modes), np.nan),
        species=np.full(len(modes), "SimLC", dtype="U16"),
        comb_mode=modes,
        comb_frequency_hz=frequency.astype(float),
    ))


def make_comb_reference_table_from_frequencies(frequencies_hz, *, mode_numbers=None):
    """Create exact SimLC references from explicit frequencies."""
    frequency = np.asarray(frequencies_hz, dtype=float)
    finite = np.isfinite(frequency) & (frequency > 0)
    frequency = frequency[finite]
    if mode_numbers is None:
        modes = np.arange(len(frequency), dtype=np.int64)
    else:
        modes = np.asarray(mode_numbers, dtype=np.int64)[finite]
    wavelength = SPEED_OF_LIGHT_NM_S / frequency
    idx = np.argsort(wavelength)
    frequency, wavelength, modes = frequency[idx], wavelength[idx], modes[idx]
    return Table(dict(
        reference_id=np.asarray([f"LC_{n:d}" for n in modes], dtype="U32"),
        wavelength_nm=wavelength.astype(float),
        wavelength_uncertainty_nm=np.zeros(len(modes), dtype=float),
        reference_intensity=np.full(len(modes), np.nan),
        species=np.full(len(modes), "SimLC", dtype="U16"),
        comb_mode=modes,
        comb_frequency_hz=frequency.astype(float),
    ))


def identify_simlc_lines(
    peak_table,
    *,
    reference_wavelength_function,
    detector_shift_y=0.0,
    y_bounds=WAVELENGTH_Y_BOUNDS,
    comb_reference_table=None,
    repetition_rate_hz=SIMLC_REPEAT_FREQUENCY_HZ,
    offset_frequency_hz=SIMLC_OFFSET_FREQUENCY_HZ,
    initial_matching_radius_pixel=2.5,
    final_matching_radius_pixel=1.2,
):
    """Assign exact LFC modes using a detector-shifted static solution."""
    peak_table = peak_table.copy(copy_data=True)
    _clear_identification(peak_table)
    predicted_by_order = {}
    for order in np.unique(np.asarray(peak_table["order"], dtype=int)):
        edge_wave = np.asarray([
            reference_wavelength_function(y_bounds[0], int(order)),
            reference_wavelength_function(y_bounds[1], int(order)),
        ], dtype=float)
        if comb_reference_table is None:
            reference = make_comb_reference_table(
                np.nanmin(edge_wave), np.nanmax(edge_wave),
                repetition_rate_hz=repetition_rate_hz,
                offset_frequency_hz=offset_frequency_hz,
            )
        else:
            reference = comb_reference_table.copy(copy_data=True)
            wave = np.asarray(reference["wavelength_nm"], dtype=float)
            reference = reference[
                (wave >= np.nanmin(edge_wave)) & (wave <= np.nanmax(edge_wave))
            ]
        predicted_by_order[int(order)] = _predict_reference_lines_for_order(
            reference,
            int(order),
            reference_wavelength_function=reference_wavelength_function,
            detector_shift_y=detector_shift_y,
            y_bounds=y_bounds,
        )
    shift = _match_peak_table_to_predicted_lines(
        peak_table,
        predicted_by_order,
        initial_matching_radius_pixel=initial_matching_radius_pixel,
        final_matching_radius_pixel=final_matching_radius_pixel,
    )
    _refresh_used_for_wavelength_fit(peak_table)
    return peak_table, shift


# Backwards-compatible spelling used by older notebooks.
identify_simlc_peaks = identify_simlc_lines


# -----------------------------------------------------------------------------
# Input/reference helpers
# -----------------------------------------------------------------------------


def _prepare_exposure(source, *, orders=None, variance=None, ccd=None, exposure_index=None, mjd_mid=np.nan):
    if isinstance(source, ExtractedExposure):
        exposure = source
        flux = np.asarray(exposure.summed.flux, dtype=float)
        var = None if exposure.summed.variance is None else np.asarray(exposure.summed.variance, dtype=float)
        physical_orders = np.asarray(exposure.orders, dtype=int)
        # ExtractedExposure is dispersion-first: (4112, n_orders).
        if flux.shape[0] == 4112 and flux.shape[1] == len(physical_orders):
            flux = flux.T
            if var is not None:
                var = var.T
        elif flux.shape[0] != len(physical_orders):
            raise ValueError("ExtractedExposure.summed.flux is incompatible with exposure.orders")
        return dict(
            counts=flux,
            variance=var,
            orders=physical_orders,
            ccd=str(exposure.ccd),
            exposure_index=int(exposure.run),
            mjd_mid=float(exposure.mjd_mid),
        )

    counts = np.asarray(source, dtype=float)
    if orders is None:
        raise ValueError("orders is required when source is not an ExtractedExposure")
    return dict(
        counts=counts,
        variance=None if variance is None else np.asarray(variance, dtype=float),
        orders=np.asarray(orders, dtype=int),
        ccd="" if ccd is None else str(ccd),
        exposure_index=-1 if exposure_index is None else int(exposure_index),
        mjd_mid=float(mjd_mid),
    )


def _coerce_wavelength_reference(reference):
    if callable(reference):
        return reference, None
    if hasattr(reference, "wavelength"):
        return reference.wavelength, None
    filename = Path(reference)
    solution, _ = read_wavelength_solution_fits(filename)
    return solution.wavelength, filename


def _candidate_lsf_reference_paths(reference_wavelength_path, ccd):
    if reference_wavelength_path is None:
        return []
    directory = Path(reference_wavelength_path).parent
    stem = Path(reference_wavelength_path).stem
    night = None
    parts = stem.split("_")
    for part in parts:
        if len(part) == 6 and part.isdigit():
            night = part
            break
    candidates = []
    if night is not None:
        candidates.append(directory / f"simlc_lsf_reference_{night}_ccd{ccd}.fits")
    candidates.append(directory / f"simlc_lsf_reference_ccd{ccd}.fits")
    return candidates


def _coerce_lsf_reference(reference_lsf, reference_wavelength_path, ccd):
    if isinstance(reference_lsf, LineSpreadFunctionModel):
        return reference_lsf
    if reference_lsf is not None:
        filename = Path(reference_lsf)
        return read_simlc_lsf_model_fits(filename)[0]
    for filename in _candidate_lsf_reference_paths(reference_wavelength_path, ccd):
        if filename.exists():
            return read_simlc_lsf_model_fits(filename)[0]
    return None


# -----------------------------------------------------------------------------
# Local first-pass LSF fits
# -----------------------------------------------------------------------------


def _order_lookup(orders):
    orders = np.asarray(orders, dtype=int)
    return {int(order): i for i, order in enumerate(orders)}


def _line_data(counts, variance, lookup, order, y0, half_width):
    j = lookup[int(order)]
    center_pixel = int(round(float(y0)))
    pixels = np.arange(center_pixel - half_width, center_pixel + half_width + 1)
    if pixels[0] < 0 or pixels[-1] >= counts.shape[1]:
        return None
    signal = np.asarray(counts[j, pixels], dtype=float)
    if variance is None:
        edge = np.r_[signal[:2], signal[-2:]]
        s = robust_sigma(edge)
        if not np.isfinite(s) or s <= 0:
            s = max(np.sqrt(max(np.nanmedian(np.abs(signal)), 1.0)), 1.0)
        var = np.full(len(signal), s * s)
    else:
        var = np.asarray(variance[j, pixels], dtype=float)
    if not np.all(np.isfinite(signal) & np.isfinite(var) & (var > 0)):
        return None
    return pixels.astype(float), signal, var


def _reference_shape_values(reference_model, y, order, shape):
    values = dict(fwhm=1.30, one_over_beta=0.26, wing_fraction=0.30, wing_sigma_ratio=1.90)
    if reference_model is None:
        return values
    try:
        fwhm = float(np.asarray(reference_model.fwhm(y, order)))
        if np.isfinite(fwhm):
            values["fwhm"] = fwhm
    except Exception:
        pass
    for name in ("one_over_beta", "wing_fraction", "wing_sigma_ratio"):
        try:
            value = float(np.asarray(reference_model.parameter(name, order)))
            if np.isfinite(value):
                values[name] = value
        except Exception:
            pass
    return values


def _fit_free_lsf_line(pixels, signal, variance, y0, shape, start, *, config):
    """Fit one line with free local LSF shape and return covariance errors."""
    y_reference = float(round(y0))
    background = float(np.nanmedian(np.r_[signal[:2], signal[-2:]]))
    amplitude = max(float(np.sum(np.clip(signal - background, 0.0, None))), 1.0)
    sigma = np.sqrt(variance)

    if shape == "gaussian":
        initial = np.array([amplitude, y0, start["fwhm"], background, 0.0])
        lower = np.array([0.0, y0-config.maximum_centroid_shift, config.minimum_fwhm, -np.inf, -np.inf])
        upper = np.array([np.inf, y0+config.maximum_centroid_shift, config.maximum_fwhm, np.inf, np.inf])
        shape_slice = slice(2, 3)
    elif shape == "moffat":
        initial = np.array([amplitude, y0, start["fwhm"], start["one_over_beta"], background, 0.0])
        lower = np.array([0.0, y0-config.maximum_centroid_shift, config.minimum_fwhm, 0.0, -np.inf, -np.inf])
        upper = np.array([np.inf, y0+config.maximum_centroid_shift, config.maximum_fwhm, config.maximum_one_over_beta, np.inf, np.inf])
        shape_slice = slice(2, 4)
    elif shape == "core_wing_gaussians":
        initial = np.array([
            amplitude, y0, start["fwhm"], start["wing_fraction"],
            start["wing_sigma_ratio"], background, 0.0,
        ])
        lower = np.array([
            0.0, y0-config.maximum_centroid_shift, config.minimum_fwhm, 0.0,
            config.minimum_wing_sigma_ratio, -np.inf, -np.inf,
        ])
        upper = np.array([
            np.inf, y0+config.maximum_centroid_shift, config.maximum_fwhm,
            config.maximum_wing_fraction, config.maximum_wing_sigma_ratio, np.inf, np.inf,
        ])
        shape_slice = slice(2, 5)
    else:
        raise ValueError(f"Unsupported LSF shape: {shape}")

    initial = np.minimum(np.maximum(initial, lower + 1e-10), upper - 1e-10)

    def unpack(parameters):
        if shape == "gaussian":
            amp, center, fwhm, bg, slope = parameters
            kwargs = dict(fwhm=fwhm)
        elif shape == "moffat":
            amp, center, fwhm, one_over_beta, bg, slope = parameters
            kwargs = dict(fwhm=fwhm, one_over_beta=one_over_beta)
        else:
            amp, center, fwhm, q, ratio, bg, slope = parameters
            kwargs = dict(fwhm=fwhm, wing_fraction=q, wing_sigma_ratio=ratio)
        return amp, center, bg, slope, kwargs

    def residuals(parameters):
        amp, center, bg, slope, kwargs = unpack(parameters)
        profile = pixel_integrated_lsf(pixels-center, shape, **kwargs)
        if not np.all(np.isfinite(profile)):
            return np.full(len(signal), 1e8)
        model = bg + slope * (pixels-y_reference) + amp * profile
        return (model-signal) / sigma

    robust = least_squares(
        residuals, initial, bounds=(lower, upper), loss="soft_l1", f_scale=1.0,
        max_nfev=300,
    )
    fit = least_squares(
        residuals, robust.x, bounds=(lower, upper), loss="linear", max_nfev=300,
    )
    amp, center, bg, slope, kwargs = unpack(fit.x)
    model = bg + slope * (pixels-y_reference) + amp * pixel_integrated_lsf(
        pixels-center, shape, **kwargs
    )
    residual = signal-model
    dof = max(1, len(signal)-len(fit.x))
    chi2 = float(np.sum((residual/sigma)**2))
    reduced_chi2 = chi2/dof
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * max(1.0, reduced_chi2)
    uncertainty = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    result = dict(
        fit_success=bool(fit.success and np.all(np.isfinite(fit.x))),
        integrated_counts=float(amp),
        integrated_counts_uncertainty=float(uncertainty[0]),
        y=float(center),
        y_uncertainty=float(uncertainty[1]),
        fwhm=float(kwargs["fwhm"]),
        fwhm_uncertainty=float(uncertainty[2]),
        background=float(bg),
        background_slope=float(slope),
        reduced_chi2=float(reduced_chi2),
        fit_rms=float(np.sqrt(np.mean(residual**2))),
    )
    if shape == "moffat":
        result["one_over_beta"] = float(kwargs["one_over_beta"])
        result["one_over_beta_uncertainty"] = float(uncertainty[3])
    elif shape == "core_wing_gaussians":
        result["wing_fraction"] = float(kwargs["wing_fraction"])
        result["wing_fraction_uncertainty"] = float(uncertainty[3])
        result["wing_sigma_ratio"] = float(kwargs["wing_sigma_ratio"])
        result["wing_sigma_ratio_uncertainty"] = float(uncertainty[4])
    return result


def _ensure_column(table, name, value, dtype=float):
    if name not in table.colnames:
        if np.isscalar(value):
            table[name] = np.full(len(table), value, dtype=dtype)
        else:
            table[name] = np.asarray(value, dtype=dtype)


def fit_initial_simlc_shapes(
    counts, orders, peak_table, *, variance=None, lsf_shape="moffat",
    reference_model=None, config=None,
):
    """Fit a free local LSF to every detected line with seed S/N >= 3."""
    if config is None:
        config = SimLCLSFConfig()
    shape = str(lsf_shape).lower()
    if shape not in SUPPORTED_LSF_SHAPES:
        raise ValueError(f"lsf_shape must be one of {SUPPORTED_LSF_SHAPES}")

    output = peak_table.copy(copy_data=True)
    n = len(output)
    _ensure_column(output, "y_initial", np.asarray(output["y"], float))
    _ensure_column(output, "y_uncertainty_initial", np.asarray(output["y_uncertainty"], float))
    _ensure_column(output, "fwhm_gaussian_initial", np.asarray(output["fwhm"], float))
    _ensure_column(output, "fwhm_gaussian_initial_uncertainty", np.asarray(output["fwhm_uncertainty"], float))
    _ensure_column(output, "signal_to_noise_initial", np.asarray(output["signal_to_noise"], float))

    for name in (
        "y_lsf_initial", "y_lsf_initial_uncertainty", "fwhm_initial",
        "fwhm_initial_uncertainty", "lsf_integrated_counts_initial",
        "lsf_integrated_counts_initial_uncertainty", "lsf_background_initial",
        "lsf_background_slope_initial", "lsf_reduced_chi2_initial",
        "lsf_fit_rms_initial",
    ):
        _ensure_column(output, name, np.nan)
    _ensure_column(output, "lsf_initial_fit_success", False, bool)
    _ensure_column(output, "lsf_shape", shape, "U32")
    if shape == "moffat":
        _ensure_column(output, "one_over_beta_initial", np.nan)
        _ensure_column(output, "one_over_beta_initial_uncertainty", np.nan)
    elif shape == "core_wing_gaussians":
        for name in (
            "wing_fraction_initial", "wing_fraction_initial_uncertainty",
            "wing_sigma_ratio_initial", "wing_sigma_ratio_initial_uncertainty",
        ):
            _ensure_column(output, name, np.nan)

    lookup = _order_lookup(orders)
    for i in range(n):
        snr = float(output["signal_to_noise_initial"][i])
        if not np.isfinite(snr) or snr < config.initial_detection_snr:
            continue
        order = int(output["order"][i])
        y0 = float(output["y_initial"][i])
        arr = _line_data(counts, variance, lookup, order, y0, config.fit_half_width)
        if arr is None:
            continue
        pixels, signal, var = arr
        start = _reference_shape_values(reference_model, y0, order, shape)
        try:
            result = _fit_free_lsf_line(
                pixels, signal, var, y0, shape, start, config=config
            )
        except Exception:
            continue
        output["y_lsf_initial"][i] = result["y"]
        output["y_lsf_initial_uncertainty"][i] = result["y_uncertainty"]
        output["fwhm_initial"][i] = result["fwhm"]
        output["fwhm_initial_uncertainty"][i] = result["fwhm_uncertainty"]
        output["lsf_integrated_counts_initial"][i] = result["integrated_counts"]
        output["lsf_integrated_counts_initial_uncertainty"][i] = result["integrated_counts_uncertainty"]
        output["lsf_background_initial"][i] = result["background"]
        output["lsf_background_slope_initial"][i] = result["background_slope"]
        output["lsf_reduced_chi2_initial"][i] = result["reduced_chi2"]
        output["lsf_fit_rms_initial"][i] = result["fit_rms"]
        output["lsf_initial_fit_success"][i] = result["fit_success"]
        if shape == "moffat":
            output["one_over_beta_initial"][i] = result["one_over_beta"]
            output["one_over_beta_initial_uncertainty"][i] = result["one_over_beta_uncertainty"]
        elif shape == "core_wing_gaussians":
            output["wing_fraction_initial"][i] = result["wing_fraction"]
            output["wing_fraction_initial_uncertainty"][i] = result["wing_fraction_uncertainty"]
            output["wing_sigma_ratio_initial"][i] = result["wing_sigma_ratio"]
            output["wing_sigma_ratio_initial_uncertainty"][i] = result["wing_sigma_ratio_uncertainty"]
    return output


# -----------------------------------------------------------------------------
# Uncertainty-aware smooth LSF field
# -----------------------------------------------------------------------------


def _normalise(values, bounds):
    lo, hi = map(float, bounds)
    center = 0.5*(lo+hi)
    scale = 0.5*(hi-lo)
    return (np.asarray(values, float)-center)/scale, center, scale


def _design_2d(y, order, y_degree, order_degree, y_bounds, order_bounds):
    yn, yc, ys = _normalise(y, y_bounds)
    mn, mc, ms = _normalise(order, order_bounds)
    yv = legvander(yn, y_degree)
    mv = legvander(mn, order_degree)
    design = np.einsum("ni,nj->nij", yv, mv).reshape(len(yn), -1)
    return design, (yc, ys, mc, ms)


def _fit_robust_linear(design, values, uncertainties, *, clip_sigma=4.5, max_iterations=6):
    values = np.asarray(values, float)
    uncertainty = np.asarray(uncertainties, float)
    finite = np.isfinite(values) & np.all(np.isfinite(design), axis=1)
    good_u = np.isfinite(uncertainty) & (uncertainty > 0)
    replacement = np.nanmedian(uncertainty[good_u]) if np.any(good_u) else 1.0
    uncertainty = np.where(good_u, uncertainty, replacement)
    used = finite.copy()
    coefficients = np.linalg.lstsq(design[used]/uncertainty[used,None], values[used]/uncertainty[used], rcond=None)[0]
    intrinsic = 0.0
    for _ in range(max_iterations):
        residual = values-design@coefficients
        scatter = robust_sigma(residual[used])
        intrinsic = 0.0 if not np.isfinite(scatter) else float(scatter)
        sigma_eff = np.sqrt(uncertainty**2 + intrinsic**2)
        new_used = finite & (np.abs(residual) <= clip_sigma*sigma_eff)
        if np.count_nonzero(new_used) < design.shape[1]+2:
            break
        coefficients_new = np.linalg.lstsq(
            design[new_used]/sigma_eff[new_used,None],
            values[new_used]/sigma_eff[new_used], rcond=None,
        )[0]
        converged = np.array_equal(new_used, used) and np.allclose(coefficients_new, coefficients, rtol=1e-8, atol=1e-10)
        used, coefficients = new_used, coefficients_new
        if converged:
            break
    residual = values-design@coefficients
    sigma_eff = np.sqrt(uncertainty**2 + intrinsic**2)
    weighted = design[used]/sigma_eff[used,None]
    normal = weighted.T@weighted
    covariance = np.linalg.pinv(normal)
    dof = max(1, np.count_nonzero(used)-design.shape[1])
    chi2 = np.sum((residual[used]/sigma_eff[used])**2)
    covariance *= max(1.0, chi2/dof)
    return coefficients, covariance, used, float(intrinsic)


def _coverage_metrics(y, y_bounds, bins):
    y = np.asarray(y, float)
    if len(y) == 0:
        return 0.0, 0
    span = (np.nanmax(y)-np.nanmin(y))/(float(y_bounds[1])-float(y_bounds[0]))
    edges = np.linspace(float(y_bounds[0]), float(y_bounds[1]), int(bins)+1)
    populated = np.count_nonzero(np.histogram(y, bins=edges)[0] > 0)
    return float(span), int(populated)


def _shape_training_indices(table, order, *, config, y_bounds):
    order_mask = np.asarray(table["order"], int) == int(order)
    flags = np.asarray(table["quality_flag"], np.int64)
    forbidden = int(
        CalibrationPeakFlag.EDGE | CalibrationPeakFlag.SATURATED
        | CalibrationPeakFlag.BLEND_CANDIDATE
    )
    eligible = order_mask.copy()
    eligible &= np.asarray(table["lsf_initial_fit_success"], bool)
    eligible &= np.isfinite(np.asarray(table["fwhm_initial"], float))
    eligible &= np.isfinite(np.asarray(table["fwhm_initial_uncertainty"], float))
    eligible &= (flags & forbidden) == 0
    eligible &= np.asarray(table["nearest_peak_distance_pixel"], float) >= config.minimum_neighbour_distance
    snr = np.asarray(table["signal_to_noise_initial"], float)
    eligible &= np.isfinite(snr) & (snr >= config.initial_detection_snr)
    base = np.flatnonzero(eligible)
    if len(base) == 0:
        return base, np.nan

    # Highest S/N threshold satisfying both target sample size and coverage.
    for minimum_lines, minimum_bins in (
        (config.target_shape_lines_per_order, config.minimum_populated_y_bins),
        (config.minimum_shape_lines_per_order, max(2, config.minimum_populated_y_bins-1)),
    ):
        for threshold in config.shape_snr_grid:
            idx = base[snr[base] >= float(threshold)]
            if len(idx) < minimum_lines:
                continue
            span, populated = _coverage_metrics(table["y_initial"][idx], y_bounds, config.y_coverage_bins)
            if span >= config.minimum_y_coverage_fraction and populated >= minimum_bins:
                return idx, float(threshold)
    if len(base) >= config.minimum_shape_lines_per_order:
        return base, float(config.initial_detection_snr)
    return np.array([], dtype=int), np.nan


def _weighted_order_measurements(table, indices, value_name, uncertainty_name):
    rows = []
    order_all = np.asarray(table["order"], int)
    for order in np.unique(order_all[indices]):
        idx = indices[order_all[indices] == int(order)]
        value = np.asarray(table[value_name][idx], float)
        error = np.asarray(table[uncertainty_name][idx], float)
        good = np.isfinite(value) & np.isfinite(error) & (error > 0)
        if np.count_nonzero(good) < 3:
            continue
        value, error = value[good], error[good]
        center = np.nanmedian(value)
        scatter = robust_sigma(value)
        if np.isfinite(scatter) and scatter > 0:
            keep = np.abs(value-center) <= 4.5*scatter
            value, error = value[keep], error[keep]
        weight = 1.0/error**2
        mean = np.sum(weight*value)/np.sum(weight)
        formal = np.sqrt(1.0/np.sum(weight))
        scatter = robust_sigma(value)
        empirical = 0.0 if not np.isfinite(scatter) else scatter/np.sqrt(max(1,len(value)))
        rows.append((int(order), float(mean), float(max(formal, empirical, 1e-6)), len(value)))
    return rows


def _fit_order_parameter(table, indices, value_name, uncertainty_name, order_bounds, *, degree, config):
    rows = _weighted_order_measurements(table, indices, value_name, uncertainty_name)
    if len(rows) < 2:
        raise RuntimeError(f"Too few orders constrain {value_name}")
    order = np.asarray([r[0] for r in rows], float)
    value = np.asarray([r[1] for r in rows], float)
    error = np.asarray([r[2] for r in rows], float)
    degree = min(int(degree), len(rows)-1)
    mn, _, _ = _normalise(order, order_bounds)
    design = legvander(mn, degree)
    coeff, cov, used, _ = _fit_robust_linear(
        design, value, error,
        clip_sigma=config.smooth_clip_sigma,
        max_iterations=config.smooth_max_iterations,
    )
    return coeff, cov, rows, used


def fit_simlc_lsf_model(
    peak_table, orders, *, lsf_shape="moffat", y_bounds=WAVELENGTH_Y_BOUNDS,
    reference_source="", config=None,
):
    """Fit FWHM(y,m) and the shape's order-only dimensionless parameters."""
    if config is None:
        config = SimLCLSFConfig()
    shape = str(lsf_shape).lower()
    orders = np.asarray(orders, int)
    output = peak_table.copy(copy_data=True)
    _ensure_column(output, "used_for_lsf_fit", False, bool)
    _ensure_column(output, "lsf_shape_snr_threshold", np.nan)

    selected = []
    thresholds = {}
    n_by_order = {}
    for order in orders:
        idx, threshold = _shape_training_indices(
            output, int(order), config=config, y_bounds=y_bounds
        )
        thresholds[int(order)] = float(threshold)
        n_by_order[int(order)] = int(len(idx))
        q = np.asarray(output["order"], int) == int(order)
        output["lsf_shape_snr_threshold"][q] = threshold
        if len(idx):
            output["used_for_lsf_fit"][idx] = True
            selected.extend(idx.tolist())
    selected = np.asarray(selected, int)
    if len(selected) < (config.fwhm_y_degree+1)*(config.fwhm_order_degree+1)+2:
        raise RuntimeError("Too few SimLC lines constrain the smooth FWHM(y,m) field")

    order_bounds = (float(np.min(orders)), float(np.max(orders)))
    design, coords = _design_2d(
        np.asarray(output["y_lsf_initial"][selected], float),
        np.asarray(output["order"][selected], float),
        config.fwhm_y_degree, config.fwhm_order_degree,
        y_bounds, order_bounds,
    )
    coeff_flat, fwhm_cov, used_local, _ = _fit_robust_linear(
        design,
        np.asarray(output["fwhm_initial"][selected], float),
        np.asarray(output["fwhm_initial_uncertainty"][selected], float),
        clip_sigma=config.smooth_clip_sigma,
        max_iterations=config.smooth_max_iterations,
    )
    fwhm_coeff = coeff_flat.reshape(config.fwhm_y_degree+1, config.fwhm_order_degree+1)
    # Mark robust outliers from the global FWHM surface as not used for LSF fit.
    rejected = selected[~used_local]
    output["used_for_lsf_fit"][rejected] = False
    selected = selected[used_local]

    yc, ys, mc, ms = coords
    parameter_coefficients = {}
    parameter_covariances = {}
    if shape == "moffat":
        coeff, cov, _, _ = _fit_order_parameter(
            output, selected, "one_over_beta_initial", "one_over_beta_initial_uncertainty",
            order_bounds, degree=config.order_parameter_degree, config=config,
        )
        parameter_coefficients["one_over_beta"] = coeff
        parameter_covariances["one_over_beta"] = cov
    elif shape == "core_wing_gaussians":
        for name in ("wing_fraction", "wing_sigma_ratio"):
            coeff, cov, _, _ = _fit_order_parameter(
                output, selected, f"{name}_initial", f"{name}_initial_uncertainty",
                order_bounds, degree=config.order_parameter_degree, config=config,
            )
            parameter_coefficients[name] = coeff
            parameter_covariances[name] = cov

    model = LineSpreadFunctionModel(
        shape=shape,
        fwhm_coefficients=fwhm_coeff,
        fwhm_covariance=fwhm_cov,
        y_center=yc, y_scale=ys, order_center=mc, order_scale=ms,
        parameter_coefficients=parameter_coefficients,
        parameter_covariances=parameter_covariances,
        order_snr_thresholds=thresholds,
        order_n_shape_lines=n_by_order,
        reference_source=str(reference_source),
    )

    # Save the smooth prediction and its uncertainty at every measured line.
    y_eval = np.asarray(output["y_lsf_initial"], float)
    fallback = ~np.isfinite(y_eval)
    y_eval[fallback] = np.asarray(output["y_initial"], float)[fallback]
    order_eval = np.asarray(output["order"], float)
    output["fwhm_model"] = np.asarray(model.fwhm(y_eval, order_eval), float)
    output["fwhm_model_uncertainty"] = np.asarray(model.fwhm_uncertainty(y_eval, order_eval), float)
    if shape == "moffat":
        output["one_over_beta_model"] = np.clip(
            np.asarray(model.parameter("one_over_beta", order_eval), float),
            0.0, config.maximum_one_over_beta,
        )
        output["one_over_beta_model_uncertainty"] = np.asarray(
            model.parameter_uncertainty("one_over_beta", order_eval), float
        )
    elif shape == "core_wing_gaussians":
        output["wing_fraction_model"] = np.clip(
            np.asarray(model.parameter("wing_fraction", order_eval), float),
            0.0, config.maximum_wing_fraction,
        )
        output["wing_fraction_model_uncertainty"] = np.asarray(
            model.parameter_uncertainty("wing_fraction", order_eval), float
        )
        output["wing_sigma_ratio_model"] = np.clip(
            np.asarray(model.parameter("wing_sigma_ratio", order_eval), float),
            config.minimum_wing_sigma_ratio, config.maximum_wing_sigma_ratio,
        )
        output["wing_sigma_ratio_model_uncertainty"] = np.asarray(
            model.parameter_uncertainty("wing_sigma_ratio", order_eval), float
        )
    return model, output


# -----------------------------------------------------------------------------
# Second-pass centroid fit with smooth shape fixed
# -----------------------------------------------------------------------------


def _model_shape_values(model, y, order, config):
    values = dict(fwhm=float(np.asarray(model.fwhm(y, order))))
    values["fwhm"] = float(np.clip(values["fwhm"], config.minimum_fwhm, config.maximum_fwhm))
    if model.shape == "moffat":
        values["one_over_beta"] = float(np.clip(
            np.asarray(model.parameter("one_over_beta", order)), 0.0, config.maximum_one_over_beta
        ))
    elif model.shape == "core_wing_gaussians":
        values["wing_fraction"] = float(np.clip(
            np.asarray(model.parameter("wing_fraction", order)), 0.0, config.maximum_wing_fraction
        ))
        values["wing_sigma_ratio"] = float(np.clip(
            np.asarray(model.parameter("wing_sigma_ratio", order)),
            config.minimum_wing_sigma_ratio, config.maximum_wing_sigma_ratio,
        ))
    return values


def _fit_fixed_shape_line(pixels, signal, variance, y0, order, model, *, config):
    y_reference = float(round(y0))
    background = float(np.nanmedian(np.r_[signal[:2], signal[-2:]]))
    amplitude = max(float(np.sum(np.clip(signal-background, 0.0, None))), 1.0)
    initial = np.array([amplitude, y0, background, 0.0])
    lower = np.array([0.0, y0-config.maximum_centroid_shift, -np.inf, -np.inf])
    upper = np.array([np.inf, y0+config.maximum_centroid_shift, np.inf, np.inf])
    sigma = np.sqrt(variance)

    def residuals(parameters):
        amp, center, bg, slope = parameters
        values = _model_shape_values(model, center, order, config)
        profile = pixel_integrated_lsf(pixels-center, model.shape, **values)
        model_flux = bg + slope*(pixels-y_reference) + amp*profile
        return (model_flux-signal)/sigma

    robust = least_squares(residuals, initial, bounds=(lower, upper), loss="soft_l1", f_scale=1.0)
    fit = least_squares(residuals, robust.x, bounds=(lower, upper), loss="linear")
    amp, center, bg, slope = fit.x
    values = _model_shape_values(model, center, order, config)
    profile = pixel_integrated_lsf(pixels-center, model.shape, **values)
    model_flux = bg + slope*(pixels-y_reference) + amp*profile
    residual = signal-model_flux
    dof = max(1, len(signal)-len(fit.x))
    chi2 = float(np.sum((residual/sigma)**2))
    reduced_chi2 = chi2/dof
    covariance = np.linalg.pinv(fit.jac.T@fit.jac) * max(1.0, reduced_chi2)
    uncertainty = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return dict(
        integrated_counts=float(amp), integrated_counts_uncertainty=float(uncertainty[0]),
        y=float(center), y_uncertainty=float(uncertainty[1]),
        background=float(bg), background_slope=float(slope),
        reduced_chi2=float(reduced_chi2), fit_rms=float(np.sqrt(np.mean(residual**2))),
        fit_success=bool(fit.success and np.all(np.isfinite(fit.x))), shape_values=values,
    )


def refit_simlc_peaks(
    counts, orders, peak_table, model, *, variance=None, minimum_snr=15.0, config=None,
):
    """Remeasure all identified SimLC centroids with the smooth LSF fixed."""
    if config is None:
        config = SimLCLSFConfig()
    output = peak_table.copy(copy_data=True)
    n = len(output)
    _ensure_column(output, "lsf_y_shift", np.nan)
    lookup = _order_lookup(orders)

    # These flags were based on the seed/free-shape fit and are reevaluated now.
    refit_bits = int(
        CalibrationPeakFlag.BAD_PROFILE_FIT | CalibrationPeakFlag.LARGE_CENTROID_ERROR
        | CalibrationPeakFlag.WIDTH_OUTLIER | CalibrationPeakFlag.LOW_SNR
    )

    for i in range(n):
        order = int(output["order"][i])
        if order not in lookup:
            continue
        y0 = float(output["y_lsf_initial"][i])
        if not np.isfinite(y0):
            y0 = float(output["y_initial"][i])
        arr = _line_data(counts, variance, lookup, order, y0, config.fit_half_width)
        if arr is None:
            continue
        pixels, signal, var = arr
        try:
            result = _fit_fixed_shape_line(
                pixels, signal, var, y0, order, model, config=config
            )
        except Exception:
            continue
        output["y"][i] = result["y"]
        output["y_uncertainty"][i] = result["y_uncertainty"]
        output["pixel_phase"][i] = result["y"]-np.round(result["y"])
        output["integrated_counts"][i] = result["integrated_counts"]
        output["integrated_counts_uncertainty"][i] = result["integrated_counts_uncertainty"]
        output["background"][i] = result["background"]
        output["background_slope"][i] = result["background_slope"]
        output["reduced_chi2"][i] = result["reduced_chi2"]
        output["fit_rms"][i] = result["fit_rms"]
        output["fit_success"][i] = result["fit_success"]
        output["lsf_y_shift"][i] = result["y"]-float(output["y_initial"][i])
        output["fwhm"][i] = result["shape_values"]["fwhm"]
        output["fwhm_uncertainty"][i] = float(output["fwhm_model_uncertainty"][i])
        output["fwhm_pixel"][i] = result["shape_values"]["fwhm"]
        output["fwhm_uncertainty_pixel"][i] = float(output["fwhm_model_uncertainty"][i])
        amp_err = result["integrated_counts_uncertainty"]
        if np.isfinite(amp_err) and amp_err > 0:
            output["signal_to_noise"][i] = result["integrated_counts"]/amp_err

        flag = int(output["quality_flag"][i]) & ~refit_bits
        if not result["fit_success"] or not np.isfinite(result["reduced_chi2"]):
            flag |= int(CalibrationPeakFlag.BAD_PROFILE_FIT)
        if not np.isfinite(result["y_uncertainty"]) or result["y_uncertainty"] > config.maximum_y_uncertainty:
            flag |= int(CalibrationPeakFlag.LARGE_CENTROID_ERROR)
        if not np.isfinite(float(output["signal_to_noise"][i])) or float(output["signal_to_noise"][i]) < float(minimum_snr):
            flag |= int(CalibrationPeakFlag.LOW_SNR)
        output["quality_flag"][i] = flag

    _refresh_used_for_wavelength_fit(output)
    return output


# -----------------------------------------------------------------------------
# FITS I/O for measurements + smooth LSF model
# -----------------------------------------------------------------------------


def _lsf_model_hdus(model):
    rows = []
    coeff = np.asarray(model.fwhm_coefficients, float)
    for i in range(coeff.shape[0]):
        for j in range(coeff.shape[1]):
            rows.append((i, j, coeff[i, j]))
    fwhm_hdu = fits.table_to_hdu(Table(rows=rows, names=("y_degree", "order_degree", "coefficient")))
    fwhm_hdu.name = "FWHM_COEFF"
    fwhm_cov = np.empty((0,0), float) if model.fwhm_covariance is None else np.asarray(model.fwhm_covariance, float)
    fwhm_cov_hdu = fits.ImageHDU(fwhm_cov, name="FWHM_COV")

    parameter_names = sorted(model.parameter_coefficients)
    pindex_values, pname_values, degree_values, coefficient_values = [], [], [], []
    for pindex, name in enumerate(parameter_names):
        for degree, value in enumerate(np.asarray(model.parameter_coefficients[name], float)):
            pindex_values.append(pindex)
            pname_values.append(name)
            degree_values.append(degree)
            coefficient_values.append(value)
    param_table = Table(dict(
        parameter_index=np.asarray(pindex_values, dtype=int),
        parameter=np.asarray(pname_values, dtype="U32"),
        degree=np.asarray(degree_values, dtype=int),
        coefficient=np.asarray(coefficient_values, dtype=float),
    ))
    param_hdu = fits.table_to_hdu(param_table)
    param_hdu.name = "PARAM_COEFF"
    cov_hdus = []
    for pindex, name in enumerate(parameter_names):
        cov = model.parameter_covariances.get(name)
        data = np.empty((0,0), float) if cov is None else np.asarray(cov, float)
        hdu = fits.ImageHDU(data, name=f"P{pindex}COV")
        hdu.header["PNAME"] = name
        cov_hdus.append(hdu)

    orders = sorted(set(model.order_snr_thresholds) | set(model.order_n_shape_lines))
    qa = Table(dict(
        order=np.asarray(orders, int),
        minimum_shape_snr=np.asarray([model.order_snr_thresholds.get(o, np.nan) for o in orders], float),
        n_shape_lines=np.asarray([model.order_n_shape_lines.get(o, 0) for o in orders], int),
    ))
    qa_hdu = fits.table_to_hdu(qa); qa_hdu.name = "LSF_ORDER_QA"
    return [fwhm_hdu, fwhm_cov_hdu, param_hdu, *cov_hdus, qa_hdu]


def _lsf_primary_header(model, *, ccd=None, mjd_mid=np.nan):
    header = fits.Header()
    header["ORIGIN"] = "velocereduction"
    header["PRODUCT"] = "SIMLC_LSF"
    header["LSFSHAPE"] = model.shape
    header["YCENTER"] = float(model.y_center)
    header["YSCALE"] = float(model.y_scale)
    header["MCENTER"] = float(model.order_center)
    header["MSCALE"] = float(model.order_scale)
    header["FYDEG"] = int(model.fwhm_y_degree)
    header["FMDEG"] = int(model.fwhm_order_degree)
    if ccd is not None:
        header["CCD"] = str(ccd)
    if np.isfinite(mjd_mid):
        header["MJD-MID"] = float(mjd_mid)
    if model.reference_source:
        header["REFLSF"] = str(model.reference_source)[:68]
    return header


def write_simlc_measurements_fits(line_set, filename, *, overwrite=False):
    """Write final line measurements and the LSF model into one FITS product."""
    if not isinstance(line_set, CalibrationLineSet):
        raise TypeError("line_set must be a CalibrationLineSet")
    model = line_set.lsf
    if not isinstance(model, LineSpreadFunctionModel):
        raise TypeError("line_set.lsf must be a LineSpreadFunctionModel")
    ccd = str(line_set.lines["ccd"][0]) if len(line_set.lines) else None
    mjd = float(line_set.lines["mjd_mid"][0]) if len(line_set.lines) else np.nan
    primary = fits.PrimaryHDU(header=_lsf_primary_header(model, ccd=ccd, mjd_mid=mjd))
    primary.header["CALTYPE"] = "SimLC"
    if np.isfinite(line_set.calibration_shift_y):
        primary.header["CALSHFTY"] = float(line_set.calibration_shift_y)
    lines_hdu = fits.table_to_hdu(line_set.lines); lines_hdu.name = "LINES"
    fits.HDUList([primary, lines_hdu, *_lsf_model_hdus(model)]).writeto(filename, overwrite=overwrite)


def write_simlc_lsf_reference_fits(model, filename, *, ccd=None, mjd_mid=np.nan, overwrite=False):
    """Write a compact smooth LSF model suitable as a future-night seed."""
    primary = fits.PrimaryHDU(header=_lsf_primary_header(model, ccd=ccd, mjd_mid=mjd_mid))
    primary.header["CONTENT"] = "SimLC LSF reference"
    fits.HDUList([primary, *_lsf_model_hdus(model)]).writeto(filename, overwrite=overwrite)


def read_simlc_lsf_model_fits(filename):
    """Read a smooth SimLC LSF model written by this module."""
    with fits.open(filename) as hdul:
        header = hdul[0].header.copy()
        ftable = Table(hdul["FWHM_COEFF"].data)
        y_degree = int(np.max(ftable["y_degree"]))
        m_degree = int(np.max(ftable["order_degree"]))
        fcoeff = np.zeros((y_degree+1, m_degree+1), float)
        for row in ftable:
            fcoeff[int(row["y_degree"]), int(row["order_degree"])] = float(row["coefficient"])
        fcov = np.asarray(hdul["FWHM_COV"].data, float)
        if fcov.size == 0: fcov = None
        pcoeff, pcov = {}, {}
        if "PARAM_COEFF" in hdul:
            ptable = Table(hdul["PARAM_COEFF"].data)
            if len(ptable):
                for name in np.unique(np.asarray(ptable["parameter"], str)):
                    q = ptable[np.asarray(ptable["parameter"], str) == name]
                    degree = int(np.max(q["degree"]))
                    arr = np.zeros(degree+1, float)
                    for row in q: arr[int(row["degree"])] = float(row["coefficient"])
                    pcoeff[str(name)] = arr
                for hdu in hdul:
                    if hdu.name.startswith("P") and hdu.name.endswith("COV") and "PNAME" in hdu.header:
                        data = np.asarray(hdu.data, float)
                        if data.size: pcov[str(hdu.header["PNAME"])] = data
        thresholds, n_lines = {}, {}
        if "LSF_ORDER_QA" in hdul:
            qtable = Table(hdul["LSF_ORDER_QA"].data)
            for row in qtable:
                thresholds[int(row["order"])] = float(row["minimum_shape_snr"])
                n_lines[int(row["order"])] = int(row["n_shape_lines"])
    model = LineSpreadFunctionModel(
        shape=str(header["LSFSHAPE"]).lower(), fwhm_coefficients=fcoeff,
        fwhm_covariance=fcov, y_center=float(header["YCENTER"]), y_scale=float(header["YSCALE"]),
        order_center=float(header["MCENTER"]), order_scale=float(header["MSCALE"]),
        parameter_coefficients=pcoeff, parameter_covariances=pcov,
        order_snr_thresholds=thresholds, order_n_shape_lines=n_lines,
        reference_source=str(header.get("REFLSF", "")),
    )
    return model, header


# Backwards-compatible I/O names.
write_simlc_lsf_fits = write_simlc_lsf_reference_fits
read_simlc_lsf_fits = read_simlc_lsf_model_fits


# -----------------------------------------------------------------------------
# High-level measurement
# -----------------------------------------------------------------------------


def measure_simlc_lines(
    exposure,
    reference_wavelength,
    *,
    lsf_shape="moffat",
    minimum_snr=SIMLC_DEFAULT_MINIMUM_SNR,
    detector_shift_y=0.0,
    reference_lsf=None,
    peak_config=None,
    lsf_config=None,
    comb_reference_table=None,
    diagnostics="basic",
    diagnostic_dir=None,
    output_filename=None,
    overwrite=False,
    log_level=None,
    # Expert/backwards-compatible array inputs:
    orders=None,
    variance=None,
    ccd=None,
    exposure_index=None,
    mjd_mid=np.nan,
    fibre=-1,
    trace_x_function=None,
):
    """Measure SimLC peaks and remeasure centres with a smooth optimized LSF.

    Parameters
    ----------
    exposure
        Prefer an :class:`~velocereduction.models.ExtractedExposure`.  Raw
        ``(n_orders, n_pixels)`` counts remain supported for expert use when
        ``orders=`` is supplied.
    reference_wavelength
        Static/reference-night wavelength solution, either a WavelengthSolution,
        a callable ``wavelength(y, order)``, or a FITS filename.
    lsf_shape
        ``'moffat'`` (default), ``'gaussian'``, or ``'core_wing_gaussians'``.
    minimum_snr
        Final S/N threshold for optimized centroids to be marked usable for the
        later wavelength fit.
    """
    if lsf_config is None:
        lsf_config = SimLCLSFConfig()
    shape = str(lsf_shape).lower()
    if shape not in SUPPORTED_LSF_SHAPES:
        raise ValueError(f"lsf_shape must be one of {SUPPORTED_LSF_SHAPES}")
    if float(minimum_snr) < lsf_config.initial_detection_snr:
        raise ValueError("minimum_snr cannot be below the initial detection S/N")

    data = _prepare_exposure(
        exposure, orders=orders, variance=variance, ccd=ccd,
        exposure_index=exposure_index, mjd_mid=mjd_mid,
    )
    counts, var, physical_orders = data["counts"], data["variance"], data["orders"]
    ccd, exposure_index, mjd_mid = data["ccd"], data["exposure_index"], data["mjd_mid"]
    wavelength_function, wavelength_path = _coerce_wavelength_reference(reference_wavelength)
    reference_model = _coerce_lsf_reference(reference_lsf, wavelength_path, ccd)

    # Preserve user tuning unrelated to the SimLC S/N floor, but always make
    # the seed pass deep enough to characterize the LSF.
    base_peak = CalibrationPeakConfig() if peak_config is None else peak_config
    seed_peak = replace(
        base_peak,
        detection_snr=float(lsf_config.initial_detection_snr),
        prominence_snr=min(float(base_peak.prominence_snr), float(lsf_config.initial_detection_snr)),
        minimum_fit_snr=float(lsf_config.initial_detection_snr),
        fit_half_width=int(lsf_config.fit_half_width),
    )
    lines = measure_calibration_peaks(
        counts, physical_orders, variance=var, calibration_type="SimLC",
        ccd=ccd, exposure_index=exposure_index, mjd_mid=mjd_mid, fibre=fibre,
        trace_x_function=trace_x_function, config=seed_peak,
        diagnostics="none", diagnostic_dir=diagnostic_dir, log_level=log_level,
    )
    if len(lines) == 0:
        return CalibrationLineSet(lines, "SimLC", np.nan, None)

    lines = fit_initial_simlc_shapes(
        counts, physical_orders, lines, variance=var, lsf_shape=shape,
        reference_model=reference_model, config=lsf_config,
    )
    reference_source = "" if reference_model is None else (
        str(reference_lsf) if reference_lsf is not None else "auto reference LSF"
    )
    model, lines = fit_simlc_lsf_model(
        lines, physical_orders, lsf_shape=shape, y_bounds=WAVELENGTH_Y_BOUNDS,
        reference_source=reference_source, config=lsf_config,
    )
    lines = refit_simlc_peaks(
        counts, physical_orders, lines, model, variance=var,
        minimum_snr=float(minimum_snr), config=lsf_config,
    )
    lines, shift = identify_simlc_lines(
        lines,
        reference_wavelength_function=wavelength_function,
        detector_shift_y=detector_shift_y,
        y_bounds=WAVELENGTH_Y_BOUNDS,
        comb_reference_table=comb_reference_table,
    )
    result = CalibrationLineSet(lines, "SimLC", shift, model)

    diagnostics = str(diagnostics).lower()
    if diagnostics not in {"none", "basic", "full"}:
        raise ValueError("diagnostics must be 'none', 'basic', or 'full'")
    if diagnostics != "none" and diagnostic_dir is not None:
        from . import diagnostics as diagnostic_plots
        diagnostic_dir = Path(diagnostic_dir)
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        summary_name = diagnostic_dir / f"simlc_ccd{ccd}_exposure{exposure_index:03d}_lsf.png"
        diagnostic_plots.plot_simlc_lsf_diagnostics(
            lines, model, filename=summary_name, show=False
        )
        if diagnostics == "full":
            example_name = diagnostic_dir / f"simlc_ccd{ccd}_exposure{exposure_index:03d}_fits.pdf"
            diagnostic_plots.save_simlc_fit_examples(
                counts, physical_orders, lines, model, filename=example_name,
                variance=var, fit_half_width=lsf_config.fit_half_width,
            )

    if output_filename is not None:
        write_simlc_measurements_fits(result, output_filename, overwrite=overwrite)

    debug = (isinstance(log_level, str) and log_level.upper() == "DEBUG") or (
        not isinstance(log_level, str) and log_level is not None and int(log_level) <= 10
    )
    if debug:
        summary = calibration_quality_summary(lines)
        print(
            f"SimLC CCD{ccd} exposure {exposure_index}: shape={shape}; "
            f"matched={summary['identified']}/{summary['total']}; "
            f"retained(S/N>={minimum_snr:g})={summary['accepted']}"
        )
        for order in physical_orders:
            q = np.asarray(lines["order"], int) == int(order)
            if not np.any(q):
                continue
            threshold = model.order_snr_thresholds.get(int(order), np.nan)
            nshape = model.order_n_shape_lines.get(int(order), 0)
            print(f"    order {int(order)}: shape S/N>={threshold:.1f}, n_shape={nshape}")
    return result
