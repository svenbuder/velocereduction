"""Source-independent calibration-line measurement helpers.

Public wavelength workflows should normally enter through ``thorium.py`` or
``simlc.py``.  This module contains only the common detector-level operations:
initial peak detection, a pixel-integrated Gaussian seed fit, quality flags,
and reference-line matching utilities.

The generic Gaussian is deliberately only a seed/measurement model.  SimLC
centroids are subsequently remeasured with the order-dependent LSF in
``simlc.py`` before they are passed to ``wavelength.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path
from typing import Any, Callable

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import median_filter
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.signal import find_peaks
from scipy.special import ndtr


@dataclass
class CalibrationLineSet:
    """Identified line measurements returned by a source-specific module.

    ``lines`` follows one common table schema irrespective of whether the
    source is FibTh, SimTh, or SimLC.  ``calibration_shift_y`` is the small
    residual shift used while matching against the bootstrap wavelength
    solution.  ``lsf`` is populated for sources with a separately inferred
    shared LSF (currently SimLC).
    """
    lines: Table
    source: str
    calibration_shift_y: float = np.nan
    lsf: Any | None = None



# Physical Veloce echelle orders in the row order used by extracted spectra.
VELOCE_CCD_ORDERS = {
    "1": np.arange(167, 138 - 1, -1),
    "2": np.arange(140, 103 - 1, -1),
    "3": np.arange(104, 65 - 1, -1),
}

class CalibrationPeakFlag(IntFlag):
    """Bit mask describing why a calibration peak should not be trusted."""

    GOOD = 0

    EDGE = 1 << 0
    SATURATED = 1 << 1
    LOW_SNR = 1 << 2
    WIDTH_OUTLIER = 1 << 3
    BLEND_CANDIDATE = 1 << 4
    BAD_PROFILE_FIT = 1 << 5
    LARGE_CENTROID_ERROR = 1 << 6
    AMBIGUOUS_MATCH = 1 << 7
    UNMATCHED = 1 << 8
    ATLAS_BLEND = 1 << 9
    WAVELENGTH_OUTLIER = 1 << 10

@dataclass
class CalibrationPeakConfig:
    """Tunable settings for detecting and fitting unresolved calibration lines."""

    # Candidate detection.
    background_window: int = 31
    noise_window: int = 101
    detection_snr: float = 5.0
    prominence_snr: float = 4.0
    minimum_peak_distance: int = 3

    # Pixel-integrated Gaussian profile fit.
    fit_half_width: int = 4
    maximum_centroid_shift: float = 1.5
    minimum_sigma: float = 0.20
    maximum_sigma: float = 3.0

    # Quality cuts on individual fits.
    minimum_fit_snr: float = 10.0
    maximum_y_uncertainty: float = 0.10
    maximum_reduced_chi2: float | None = None

    # Maximum value in the extracted 1D spectrum allowed anywhere in the
    # local fitting window.  Set separately for SimLC/SimTh/FibTh if needed.
    # ``None`` disables this cut.
    maximum_signal: float | None = None

    # Ensemble FWHM rejection.  These are relative to the typical line width
    # in each order so the cut can follow the instrumental profile.
    minimum_fwhm_ratio: float = 0.55
    maximum_fwhm_ratio: float = 1.80
    fwhm_mad_sigma: float = 4.0

    # A measured neighbour this close is considered a likely blend.
    measured_blend_fwhm_factor: float = 2.0

    # A reference-atlas neighbour this close is considered unresolved.
    atlas_blend_fwhm_factor: float = 1.5

def robust_sigma(values: np.ndarray) -> float:
    """Gaussian-equivalent scatter estimated from the MAD."""

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan

    median = np.nanmedian(values[finite])
    return 1.4826 * np.nanmedian(np.abs(values[finite] - median))

def calibration_quality_summary(peak_table: Table) -> dict[str, int]:
    """Count accepted/identified lines and every active quality flag.

    Flag counts are intentionally non-exclusive: a rejected line can contribute
    to more than one reason.  This makes attrition between detection, profile
    quality, reference matching, and wavelength fitting explicit in DEBUG QA.
    """
    summary = {"total": int(len(peak_table)), "accepted": 0, "identified": 0}
    if len(peak_table) == 0:
        return summary

    if "used_for_wavelength_fit" in peak_table.colnames:
        summary["accepted"] = int(np.count_nonzero(peak_table["used_for_wavelength_fit"]))
    if "wavelength_nm" in peak_table.colnames:
        summary["identified"] = int(
            np.count_nonzero(np.isfinite(np.asarray(peak_table["wavelength_nm"], dtype=float)))
        )

    flags = np.asarray(peak_table["quality_flag"], dtype=np.int64)
    for flag in CalibrationPeakFlag:
        if flag == CalibrationPeakFlag.GOOD:
            continue
        summary[flag.name.lower()] = int(np.count_nonzero((flags & int(flag)) != 0))
    return summary


def _debug_enabled(log_level) -> bool:
    if isinstance(log_level, str):
        return log_level.upper() == "DEBUG"
    return bool(log_level is not None and int(log_level) <= 10)


def sum_extracted_calibration_order(
    extracted_counts: np.ndarray,
    *,
    extracted_variance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sum one extracted 2D order over the cross-dispersion direction.

    Parameters
    ----------
    extracted_counts
        Array with shape ``(n_dispersion_pixels, n_cross_dispersion_pixels)``.
    extracted_variance
        Optional variance array with the same shape.  Variances add when the
        extracted detector pixels are summed.

    Returns
    -------
    counts, variance
        One-dimensional arrays along the dispersion coordinate ``y``.
    """

    extracted_counts = np.asarray(extracted_counts, dtype=float)
    if extracted_counts.ndim != 2:
        raise ValueError("extracted_counts must be a 2D order matrix")

    counts = np.nansum(extracted_counts, axis=1)

    variance = None
    if extracted_variance is not None:
        extracted_variance = np.asarray(extracted_variance, dtype=float)
        if extracted_variance.shape != extracted_counts.shape:
            raise ValueError("extracted_variance must match extracted_counts")
        variance = np.nansum(extracted_variance, axis=1)

    return counts, variance

def pixel_integrated_gaussian(
    y: np.ndarray,
    integrated_counts: float,
    y_center: float,
    sigma: float,
) -> np.ndarray:
    """Gaussian line-spread function integrated over finite detector pixels.

    This is preferable to evaluating a Gaussian only at pixel centres when
    the calibration lines are only sampled by a few detector pixels.
    """

    y = np.asarray(y, dtype=float)

    lower_edge = (y - 0.5 - y_center) / sigma
    upper_edge = (y + 0.5 - y_center) / sigma

    return integrated_counts * (ndtr(upper_edge) - ndtr(lower_edge))

def calibration_line_model(
    y: np.ndarray,
    integrated_counts: float,
    y_center: float,
    sigma: float,
    background: float,
    background_slope: float,
    *,
    y_reference: float,
) -> np.ndarray:
    """Pixel-integrated Gaussian plus a local linear background."""

    return (
        background
        + background_slope * (y - y_reference)
        + pixel_integrated_gaussian(y, integrated_counts, y_center, sigma)
    )

def detect_calibration_peaks(
    counts: np.ndarray,
    *,
    config: CalibrationPeakConfig | None = None,
):
    """Detect candidate emission lines in one extracted echelle order.

    Detection is performed on a locally background-subtracted, approximately
    S/N-normalised spectrum.  The returned integer positions are only starting
    guesses; sub-pixel positions are measured by ``fit_calibration_peak``.
    """

    if config is None:
        config = CalibrationPeakConfig()

    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError("counts must be one-dimensional")

    finite_counts = counts[np.isfinite(counts)]
    if len(finite_counts) == 0:
        return (
            np.array([], dtype=int),
            np.full_like(counts, np.nan),
            np.full_like(counts, np.nan),
            np.full_like(counts, np.nan),
            {},
        )

    fill_value = float(np.nanmedian(finite_counts))
    working_counts = np.where(np.isfinite(counts), counts, fill_value)

    background = median_filter(
        working_counts,
        size=config.background_window,
        mode="nearest",
    )
    line_signal = working_counts - background

    local_center = median_filter(
        line_signal,
        size=config.noise_window,
        mode="nearest",
    )
    absolute_deviation = np.abs(line_signal - local_center)
    local_noise = 1.4826 * median_filter(
        absolute_deviation,
        size=config.noise_window,
        mode="nearest",
    )

    global_noise = robust_sigma(line_signal)
    if not np.isfinite(global_noise) or global_noise <= 0:
        global_noise = 1.0

    local_noise = np.maximum(local_noise, global_noise)
    detection_snr = np.divide(
        line_signal,
        local_noise,
        out=np.zeros_like(line_signal),
        where=local_noise > 0,
    )

    candidate_pixels, properties = find_peaks(
        detection_snr,
        height=config.detection_snr,
        prominence=config.prominence_snr,
        distance=config.minimum_peak_distance,
    )

    return candidate_pixels, background, local_noise, detection_snr, properties

def fit_calibration_peak(
    counts: np.ndarray,
    candidate_pixel: int,
    *,
    variance: np.ndarray | None = None,
    background: np.ndarray | None = None,
    noise: np.ndarray | None = None,
    config: CalibrationPeakConfig | None = None,
) -> dict:
    """Fit one calibration peak with a pixel-integrated Gaussian profile."""

    if config is None:
        config = CalibrationPeakConfig()

    counts = np.asarray(counts, dtype=float)
    n_pixels = len(counts)

    left = max(0, int(candidate_pixel) - config.fit_half_width)
    right = min(n_pixels, int(candidate_pixel) + config.fit_half_width + 1)

    y = np.arange(left, right, dtype=float)
    signal = counts[left:right]
    y_reference = float(candidate_pixel)

    edge = left == 0 or right == n_pixels

    maximum_observed_signal = float(np.nanmax(signal))
    saturated = (
        config.maximum_signal is not None
        and maximum_observed_signal >= config.maximum_signal
    )

    if background is None:
        n_edge = min(2, max(1, len(signal) // 3))
        background_guess = float(
            np.nanmedian(np.concatenate([signal[:n_edge], signal[-n_edge:]]))
        )
    else:
        background_guess = float(background[candidate_pixel])

    peak_height_guess = max(
        float(counts[candidate_pixel]) - background_guess,
        1.0,
    )
    sigma_guess = 0.6
    integrated_counts_guess = (
        peak_height_guess * np.sqrt(2.0 * np.pi) * sigma_guess
    )

    initial_parameters = np.array(
        [
            integrated_counts_guess,
            float(candidate_pixel),
            sigma_guess,
            background_guess,
            0.0,
        ],
        dtype=float,
    )

    lower_bounds = np.array(
        [
            0.0,
            candidate_pixel - config.maximum_centroid_shift,
            config.minimum_sigma,
            -np.inf,
            -np.inf,
        ],
        dtype=float,
    )
    upper_bounds = np.array(
        [
            np.inf,
            candidate_pixel + config.maximum_centroid_shift,
            config.maximum_sigma,
            np.inf,
            np.inf,
        ],
        dtype=float,
    )

    if variance is not None:
        variance = np.asarray(variance, dtype=float)
        if variance.shape != counts.shape:
            raise ValueError("variance must have the same shape as counts")
        sigma_counts = np.sqrt(np.clip(variance[left:right], 0.0, None))
    elif noise is not None:
        sigma_counts = np.asarray(noise[left:right], dtype=float).copy()
    else:
        local_sigma = robust_sigma(signal)
        if not np.isfinite(local_sigma) or local_sigma <= 0:
            local_sigma = 1.0
        sigma_counts = np.full(len(y), local_sigma, dtype=float)

    valid_sigma = np.isfinite(sigma_counts) & (sigma_counts > 0)
    if np.any(valid_sigma):
        replacement_sigma = float(np.nanmedian(sigma_counts[valid_sigma]))
    else:
        replacement_sigma = 1.0
    sigma_counts[~valid_sigma] = replacement_sigma

    def residuals(parameters):
        model = calibration_line_model(
            y,
            *parameters,
            y_reference=y_reference,
        )
        return (model - signal) / sigma_counts

    # The first robust pass limits the leverage of a deviant pixel.  The
    # second ordinary weighted pass gives a more interpretable Jacobian for
    # the local covariance estimate.
    robust_fit = least_squares(
        residuals,
        initial_parameters,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=1.0,
    )

    final_fit = least_squares(
        residuals,
        robust_fit.x,
        bounds=(lower_bounds, upper_bounds),
        loss="linear",
    )

    (
        integrated_counts,
        y_center,
        sigma,
        fitted_background,
        background_slope,
    ) = final_fit.x

    model = calibration_line_model(
        y,
        *final_fit.x,
        y_reference=y_reference,
    )
    fit_residual = signal - model

    degrees_of_freedom = max(1, len(y) - len(final_fit.x))
    chi2 = float(np.sum((fit_residual / sigma_counts) ** 2))
    reduced_chi2 = chi2 / degrees_of_freedom

    covariance = np.linalg.pinv(final_fit.jac.T @ final_fit.jac)
    covariance *= max(1.0, reduced_chi2)
    parameter_uncertainty = np.sqrt(
        np.clip(np.diag(covariance), 0.0, None)
    )

    integrated_counts_uncertainty = float(parameter_uncertainty[0])
    y_uncertainty = float(parameter_uncertainty[1])
    sigma_uncertainty = float(parameter_uncertainty[2])

    gaussian_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhm = float(gaussian_to_fwhm * sigma)
    fwhm_uncertainty = float(gaussian_to_fwhm * sigma_uncertainty)

    if integrated_counts_uncertainty > 0:
        signal_to_noise = float(
            integrated_counts / integrated_counts_uncertainty
        )
    else:
        signal_to_noise = np.nan

    intrinsic_peak_amplitude = float(
        integrated_counts / (np.sqrt(2.0 * np.pi) * sigma)
    )
    fit_rms = float(np.sqrt(np.nanmean(fit_residual**2)))

    flag = CalibrationPeakFlag.GOOD
    if edge:
        flag |= CalibrationPeakFlag.EDGE
    if saturated:
        flag |= CalibrationPeakFlag.SATURATED
    if not final_fit.success or not np.all(np.isfinite(final_fit.x)):
        flag |= CalibrationPeakFlag.BAD_PROFILE_FIT
    if not np.isfinite(signal_to_noise) or signal_to_noise < config.minimum_fit_snr:
        flag |= CalibrationPeakFlag.LOW_SNR
    if not np.isfinite(y_uncertainty) or y_uncertainty > config.maximum_y_uncertainty:
        flag |= CalibrationPeakFlag.LARGE_CENTROID_ERROR
    if not np.isfinite(reduced_chi2):
        flag |= CalibrationPeakFlag.BAD_PROFILE_FIT
    elif (
        config.maximum_reduced_chi2 is not None
        and reduced_chi2 > config.maximum_reduced_chi2
    ):
        flag |= CalibrationPeakFlag.BAD_PROFILE_FIT

    return dict(
        y=float(y_center),
        y_uncertainty=y_uncertainty,
        pixel_phase=float(y_center - np.round(y_center)),
        integrated_counts=float(integrated_counts),
        integrated_counts_uncertainty=integrated_counts_uncertainty,
        intrinsic_peak_amplitude=intrinsic_peak_amplitude,
        maximum_signal=maximum_observed_signal,
        fwhm=fwhm,
        fwhm_uncertainty=fwhm_uncertainty,
        background=float(fitted_background),
        background_slope=float(background_slope),
        signal_to_noise=signal_to_noise,
        reduced_chi2=float(reduced_chi2),
        fit_rms=fit_rms,
        fit_success=bool(final_fit.success),
        quality_flag=int(flag),
    )

def _blank_identification_columns() -> dict:
    """Columns filled only after an LC/Th reference line is assigned."""

    return dict(
        wavelength_nm=np.nan,
        wavelength_uncertainty_nm=np.nan,
        m_times_lambda=np.nan,
        reference_id="",
        species="",
        reference_intensity=np.nan,
        comb_mode=-1,
        comb_frequency_hz=np.nan,
        y_reference=np.nan,
        y_expected=np.nan,
        match_residual_y=np.nan,
        atlas_neighbour_distance_pixel=np.nan,
        used_for_wavelength_fit=False,
        wavelength_residual_nm=np.nan,
        pixel_residual=np.nan,
        velocity_residual_mps=np.nan,
    )

def _apply_ensemble_peak_flags(
    peak_table: Table,
    *,
    config: CalibrationPeakConfig,
) -> None:
    """Flag unusual line widths and measured close neighbours per order."""

    orders = np.asarray(peak_table["order"], dtype=int)

    for order in np.unique(orders):
        indices = np.where(orders == order)[0]

        # Use lines that are not already obviously unusable to estimate the
        # local instrumental FWHM distribution.
        basic_good = np.array(
            [
                int(peak_table["quality_flag"][i])
                & int(
                    CalibrationPeakFlag.SATURATED
                    | CalibrationPeakFlag.BAD_PROFILE_FIT
                    | CalibrationPeakFlag.EDGE
                )
                == 0
                for i in indices
            ],
            dtype=bool,
        )

        if np.count_nonzero(basic_good) >= 3:
            good_indices = indices[basic_good]
            widths = np.asarray(
                peak_table["fwhm"][good_indices],
                dtype=float,
            )

            median_fwhm = float(np.nanmedian(widths))
            fwhm_scatter = robust_sigma(widths)

            lower = config.minimum_fwhm_ratio * median_fwhm
            upper = config.maximum_fwhm_ratio * median_fwhm

            if np.isfinite(fwhm_scatter) and fwhm_scatter > 0:
                lower = max(
                    lower,
                    median_fwhm - config.fwhm_mad_sigma * fwhm_scatter,
                )
                upper = min(
                    upper,
                    median_fwhm + config.fwhm_mad_sigma * fwhm_scatter,
                )

            for i in indices:
                width = float(peak_table["fwhm"][i])
                if not np.isfinite(width) or width < lower or width > upper:
                    peak_table["quality_flag"][i] = int(
                        int(peak_table["quality_flag"][i])
                        | int(CalibrationPeakFlag.WIDTH_OUTLIER)
                    )

        # Independent measured-neighbour blend check.
        y_values = np.asarray(peak_table["y"][indices], dtype=float)
        sort_index = np.argsort(y_values)
        sorted_indices = indices[sort_index]
        sorted_y = y_values[sort_index]

        nearest = np.full(len(sorted_indices), np.inf)
        if len(sorted_indices) > 1:
            separation = np.diff(sorted_y)
            nearest[:-1] = np.minimum(nearest[:-1], separation)
            nearest[1:] = np.minimum(nearest[1:], separation)

        for local_i, table_i in enumerate(sorted_indices):
            peak_table["nearest_peak_distance_pixel"][table_i] = nearest[local_i]
            fwhm = float(peak_table["fwhm"][table_i])
            if (
                np.isfinite(fwhm)
                and nearest[local_i] < config.measured_blend_fwhm_factor * fwhm
            ):
                peak_table["quality_flag"][table_i] = int(
                    int(peak_table["quality_flag"][table_i])
                    | int(CalibrationPeakFlag.BLEND_CANDIDATE)
                )

def measure_calibration_peaks(
    counts: np.ndarray,
    orders: np.ndarray,
    *,
    variance: np.ndarray | None = None,
    calibration_type: str = "",
    ccd: int | str | None = None,
    exposure_index: int | None = None,
    mjd_mid: float = np.nan,
    fibre: int = -1,
    trace_x_function: Callable[[int, float], float] | None = None,
    config: CalibrationPeakConfig | None = None,
    diagnostics: str = "none",
    diagnostic_dir: str | Path | None = None,
    log_level: str | int | None = None,
) -> Table:
    """Measure every candidate calibration peak in one extracted exposure.

    ``log_level='DEBUG'`` prints progress and a quality summary for every order.
    ``diagnostics='full'`` additionally writes the v0.7-style per-order peak QA
    pages (plus rejected/worst local fits) to ``diagnostic_dir``.
    """
    if config is None:
        config = CalibrationPeakConfig()

    diagnostics = str(diagnostics).lower()
    if diagnostics not in {"none", "basic", "full"}:
        raise ValueError("diagnostics must be 'none', 'basic', or 'full'")
    debug = _debug_enabled(log_level)

    counts = np.asarray(counts, dtype=float)
    orders = np.asarray(orders, dtype=int)
    if counts.ndim != 2:
        raise ValueError("counts must have shape (n_orders, n_dispersion_pixels)")
    if len(orders) != counts.shape[0]:
        raise ValueError("orders must have one entry for every row in counts")

    if variance is not None:
        variance = np.asarray(variance, dtype=float)
        if variance.shape != counts.shape:
            raise ValueError("variance must have the same shape as counts")

    rows = []
    peak_id = 0
    order_diagnostics = {}
    fibre_text = "" if int(fibre) == -1 else f" fibre {int(fibre):+d}"

    if debug:
        print("\n" + "=" * 78)
        print(f"Measuring {calibration_type} CCD{ccd} exposure {exposure_index}{fibre_text}")
        print(
            f"  detection S/N >= {config.detection_snr:.1f}; "
            f"prominence >= {config.prominence_snr:.1f}; "
            f"fit S/N >= {config.minimum_fit_snr:.1f}; "
            f"sigma_y <= {config.maximum_y_uncertainty:.3f} pix; "
            f"FWHM ratio={config.minimum_fwhm_ratio:.2f}--{config.maximum_fwhm_ratio:.2f}; "
            f"measured blend < {config.measured_blend_fwhm_factor:.1f} FWHM"
        )

    for order_index, order in enumerate(orders):
        order_counts = counts[order_index]
        order_variance = None if variance is None else variance[order_index]
        candidates, background, local_noise, detection_snr, _ = detect_calibration_peaks(
            order_counts, config=config
        )

        if debug:
            print(
                f"{calibration_type} CCD{ccd} exposure {exposure_index}"
                f"{fibre_text} order {int(order)}: fitting {len(candidates)} candidates"
            )

        for candidate in candidates:
            try:
                result = fit_calibration_peak(
                    order_counts,
                    int(candidate),
                    variance=order_variance,
                    background=background,
                    noise=local_noise,
                    config=config,
                )
            except Exception as error:
                if debug:
                    print(f"    y={int(candidate)} fit failed: {error}")
                continue

            trace_x = np.nan
            if trace_x_function is not None:
                trace_x = float(trace_x_function(int(order), result["y"]))

            rows.append(dict(
                peak_id=int(peak_id), candidate_pixel=int(candidate),
                calibration_type=str(calibration_type),
                ccd=str(ccd) if ccd is not None else "",
                exposure_index=int(exposure_index) if exposure_index is not None else -1,
                mjd_mid=float(mjd_mid), fibre=int(fibre), order=int(order),
                y=result["y"], y_uncertainty=result["y_uncertainty"],
                x=trace_x, x_uncertainty=np.nan,
                x_source=("trace_model" if np.isfinite(trace_x) else ""),
                pixel_phase=result["pixel_phase"], fwhm=result["fwhm"],
                fwhm_uncertainty=result["fwhm_uncertainty"],
                fwhm_pixel=result["fwhm"],
                fwhm_uncertainty_pixel=result["fwhm_uncertainty"],
                integrated_counts=result["integrated_counts"],
                integrated_counts_uncertainty=result["integrated_counts_uncertainty"],
                intrinsic_peak_amplitude=result["intrinsic_peak_amplitude"],
                maximum_signal=result["maximum_signal"],
                signal_to_noise=result["signal_to_noise"],
                background=result["background"], background_slope=result["background_slope"],
                reduced_chi2=result["reduced_chi2"], fit_rms=result["fit_rms"],
                fit_success=result["fit_success"], nearest_peak_distance_pixel=np.inf,
                quality_flag=int(result["quality_flag"]), **_blank_identification_columns(),
            ))
            peak_id += 1

        order_diagnostics[int(order)] = dict(
            counts=order_counts, background=background, detection_snr=detection_snr,
            candidate_pixels=candidates,
        )

    peak_table = Table(rows=rows)
    if len(peak_table) > 0:
        peak_table["reference_id"] = np.full(len(peak_table), "", dtype="U64")
        peak_table["species"] = np.full(len(peak_table), "", dtype="U32")
        _apply_ensemble_peak_flags(peak_table, config=config)
        peak_table["used_for_wavelength_fit"] = (
            np.asarray(peak_table["quality_flag"], dtype=np.int64) == 0
        )

    if debug:
        for order in orders:
            subset = peak_table[np.asarray(peak_table["order"], int) == int(order)] if len(peak_table) else peak_table
            summary = calibration_quality_summary(subset)
            if len(subset):
                print(
                    f"    order {int(order)}: {summary['accepted']}/{summary['total']} profile-good; "
                    f"median FWHM={np.nanmedian(subset['fwhm']):.3f} px; "
                    f"median sigma_y={np.nanmedian(subset['y_uncertainty']):.4f} px; "
                    f"median S/N={np.nanmedian(subset['signal_to_noise']):.1f}; "
                    f"width={summary.get('width_outlier', 0)}, "
                    f"blend={summary.get('blend_candidate', 0)}, "
                    f"lowS/N={summary.get('low_snr', 0)}"
                )

    if diagnostics == "full" and diagnostic_dir is not None and len(peak_table):
        from . import diagnostics as diagnostic_plots
        diagnostic_pdf = diagnostic_plots.save_calibration_order_diagnostics(
            order_diagnostics, peak_table, config=config,
            calibration_type=calibration_type, ccd=str(ccd),
            exposure_index=-1 if exposure_index is None else int(exposure_index),
            fibre=int(fibre), diagnostic_dir=diagnostic_dir,
        )
        if debug:
            print(f"  full peak diagnostics -> {diagnostic_pdf}")

    return peak_table

def _invert_reference_wavelength_solution(
    reference_wavelength_function,
    order: int,
    wavelengths_nm: np.ndarray,
    *,
    y_bounds: tuple[float, float],
    interpolation_grid_size: int = 16385,
) -> np.ndarray:
    """Convert reference wavelengths into reference-night dispersion pixels.

    A dense monotonic interpolation is considerably faster than solving a
    scalar root independently for every comb line.
    """

    y_grid = np.linspace(
        float(y_bounds[0]),
        float(y_bounds[1]),
        interpolation_grid_size,
    )
    wavelength_grid = np.asarray(
        reference_wavelength_function(y_grid, int(order)),
        dtype=float,
    )

    finite = np.isfinite(wavelength_grid)
    if np.count_nonzero(finite) < 2:
        return np.full(len(wavelengths_nm), np.nan)

    y_grid = y_grid[finite]
    wavelength_grid = wavelength_grid[finite]

    # np.interp expects increasing x.  Echelle orders may have wavelength
    # increasing or decreasing with detector y, so sort by wavelength.
    sort_index = np.argsort(wavelength_grid)
    wavelength_sorted = wavelength_grid[sort_index]
    y_sorted = y_grid[sort_index]

    # Remove duplicate wavelength values if numerical noise produces any.
    wavelength_unique, unique_index = np.unique(
        wavelength_sorted,
        return_index=True,
    )
    y_unique = y_sorted[unique_index]

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    y_reference = np.full(len(wavelengths_nm), np.nan)

    inside = (
        wavelengths_nm >= wavelength_unique[0]
    ) & (
        wavelengths_nm <= wavelength_unique[-1]
    )

    y_reference[inside] = np.interp(
        wavelengths_nm[inside],
        wavelength_unique,
        y_unique,
    )

    return y_reference

def _predict_reference_lines_for_order(
    reference_table: Table,
    order: int,
    *,
    reference_wavelength_function,
    detector_shift_y: float,
    y_bounds: tuple[float, float],
) -> Table:
    """Predict current detector positions of reference lines in one order."""

    reference = reference_table.copy(copy_data=True)
    wavelengths = np.asarray(reference["wavelength_nm"], dtype=float)

    y_reference = _invert_reference_wavelength_solution(
        reference_wavelength_function,
        int(order),
        wavelengths,
        y_bounds=y_bounds,
    )

    valid = np.isfinite(y_reference)
    reference = reference[valid]
    y_reference = y_reference[valid]

    reference["y_reference"] = y_reference
    reference["y_expected"] = y_reference + float(detector_shift_y)

    return reference

def _one_to_one_match(
    measured_y: np.ndarray,
    expected_y: np.ndarray,
    *,
    matching_radius_pixel: float,
):
    """Minimum-cost one-to-one assignment followed by a distance cut."""

    measured_y = np.asarray(measured_y, dtype=float)
    expected_y = np.asarray(expected_y, dtype=float)

    if len(measured_y) == 0 or len(expected_y) == 0:
        return []

    cost = np.abs(measured_y[:, None] - expected_y[None, :])
    measured_index, reference_index = linear_sum_assignment(cost)

    matches = []
    for measured_i, reference_i in zip(measured_index, reference_index):
        residual = measured_y[measured_i] - expected_y[reference_i]
        if abs(residual) <= matching_radius_pixel:
            matches.append(
                (int(measured_i), int(reference_i), float(residual))
            )

    return matches

def _clear_identification(peak_table: Table) -> None:
    """Reset wavelength-identification columns while preserving measurements."""

    float_columns = [
        "wavelength_nm",
        "wavelength_uncertainty_nm",
        "m_times_lambda",
        "reference_intensity",
        "comb_frequency_hz",
        "y_reference",
        "y_expected",
        "match_residual_y",
        "atlas_neighbour_distance_pixel",
        "wavelength_residual_nm",
        "pixel_residual",
        "velocity_residual_mps",
    ]

    for name in float_columns:
        peak_table[name] = np.full(len(peak_table), np.nan, dtype=float)

    peak_table["reference_id"] = np.full(len(peak_table), "", dtype="U64")
    peak_table["species"] = np.full(len(peak_table), "", dtype="U32")
    peak_table["comb_mode"] = np.full(len(peak_table), -1, dtype=np.int64)
    peak_table["used_for_wavelength_fit"] = np.zeros(len(peak_table), dtype=bool)

    clear_flags = int(
        CalibrationPeakFlag.AMBIGUOUS_MATCH
        | CalibrationPeakFlag.UNMATCHED
        | CalibrationPeakFlag.ATLAS_BLEND
        | CalibrationPeakFlag.WAVELENGTH_OUTLIER
    )

    for i in range(len(peak_table)):
        peak_table["quality_flag"][i] = int(
            int(peak_table["quality_flag"][i]) & ~clear_flags
        )

def _copy_reference_match(
    peak_table: Table,
    peak_index: int,
    reference_row,
    *,
    y_expected: float,
    match_residual_y: float,
) -> None:
    """Copy one identified LC/Th reference line into a measured peak row."""

    order = int(peak_table["order"][peak_index])
    wavelength_nm = float(reference_row["wavelength_nm"])

    peak_table["wavelength_nm"][peak_index] = wavelength_nm

    wavelength_uncertainty = float(reference_row["wavelength_uncertainty_nm"])
    peak_table["wavelength_uncertainty_nm"][peak_index] = wavelength_uncertainty

    peak_table["m_times_lambda"][peak_index] = order * wavelength_nm
    peak_table["reference_id"][peak_index] = str(reference_row["reference_id"])
    peak_table["species"][peak_index] = str(reference_row["species"])
    peak_table["reference_intensity"][peak_index] = float(
        reference_row["reference_intensity"]
    )

    if "comb_mode" in reference_row.colnames:
        peak_table["comb_mode"][peak_index] = int(reference_row["comb_mode"])
    if "comb_frequency_hz" in reference_row.colnames:
        peak_table["comb_frequency_hz"][peak_index] = float(
            reference_row["comb_frequency_hz"]
        )

    peak_table["y_reference"][peak_index] = float(reference_row["y_reference"])
    peak_table["y_expected"][peak_index] = float(y_expected)
    peak_table["match_residual_y"][peak_index] = float(match_residual_y)

def _match_peak_table_to_predicted_lines(
    peak_table: Table,
    predicted_by_order: dict[int, Table],
    *,
    initial_matching_radius_pixel: float,
    final_matching_radius_pixel: float,
) -> float:
    """Two-pass unique matching after the independently measured detector shift.

    The first pass measures only a small residual calibration shift.  It must
    not be allowed to absorb an entire comb spacing; the external detector
    registration is what prevents a one-mode SimLC ambiguity.
    """

    orders = np.asarray(peak_table["order"], dtype=int)
    first_pass_residuals = []

    for order, predicted in predicted_by_order.items():
        measured_indices = np.where(orders == int(order))[0]
        if len(measured_indices) == 0 or len(predicted) == 0:
            continue

        matches = _one_to_one_match(
            np.asarray(peak_table["y"][measured_indices], dtype=float),
            np.asarray(predicted["y_expected"], dtype=float),
            matching_radius_pixel=initial_matching_radius_pixel,
        )

        for _, _, residual in matches:
            first_pass_residuals.append(residual)

    if len(first_pass_residuals) == 0:
        raise RuntimeError(
            "No secure calibration-line matches were found after applying "
            "detector_shift_y. Check its sign/value and the reference solution."
        )

    calibration_shift_y = float(np.nanmedian(first_pass_residuals))

    if abs(calibration_shift_y) > initial_matching_radius_pixel:
        raise RuntimeError(
            "The residual calibration shift is unexpectedly large.  Check "
            "detector_shift_y before attempting wavelength identification."
        )

    matched_peak_indices = set()

    for order, predicted in predicted_by_order.items():
        measured_indices = np.where(orders == int(order))[0]
        if len(measured_indices) == 0 or len(predicted) == 0:
            continue

        expected_refined = (
            np.asarray(predicted["y_expected"], dtype=float)
            + calibration_shift_y
        )

        matches = _one_to_one_match(
            np.asarray(peak_table["y"][measured_indices], dtype=float),
            expected_refined,
            matching_radius_pixel=final_matching_radius_pixel,
        )

        for local_measured, reference_i, residual in matches:
            peak_i = int(measured_indices[local_measured])
            _copy_reference_match(
                peak_table,
                peak_i,
                predicted[reference_i],
                y_expected=float(expected_refined[reference_i]),
                match_residual_y=float(residual),
            )
            matched_peak_indices.add(peak_i)

    for i in range(len(peak_table)):
        if i not in matched_peak_indices:
            peak_table["quality_flag"][i] = int(
                int(peak_table["quality_flag"][i])
                | int(CalibrationPeakFlag.UNMATCHED)
            )

    return calibration_shift_y

def _refresh_used_for_wavelength_fit(peak_table: Table) -> None:
    """Set the final pre-surface-fit line-selection mask."""

    identified = np.isfinite(np.asarray(peak_table["wavelength_nm"], dtype=float))
    good_flag = np.asarray(peak_table["quality_flag"], dtype=np.int64) == 0
    finite_measurement = (
        np.isfinite(np.asarray(peak_table["y"], dtype=float))
        & np.isfinite(np.asarray(peak_table["y_uncertainty"], dtype=float))
    )

    peak_table["used_for_wavelength_fit"] = (
        identified & good_flag & finite_measurement
    )



# -----------------------------------------------------------------------------
# Compact line-table FITS I/O
# -----------------------------------------------------------------------------

def write_calibration_line_fits(
    line_set: CalibrationLineSet | Table,
    filename,
    *,
    source: str | None = None,
    calibration_shift_y: float | None = None,
    overwrite: bool = False,
) -> None:
    """Write a common identified calibration-line table to FITS."""
    if isinstance(line_set, CalibrationLineSet):
        table = line_set.lines
        source = line_set.source if source is None else source
        if calibration_shift_y is None:
            calibration_shift_y = line_set.calibration_shift_y
    else:
        table = line_set

    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "velocereduction"
    primary.header["CONTENT"] = "Identified calibration lines"
    if source is not None:
        primary.header["CALTYPE"] = str(source)
    if calibration_shift_y is not None and np.isfinite(calibration_shift_y):
        primary.header["CALSHFTY"] = (float(calibration_shift_y), "residual line-match shift [pix]")

    hdu = fits.table_to_hdu(table)
    hdu.name = "LINES"
    fits.HDUList([primary, hdu]).writeto(filename, overwrite=overwrite)


def read_calibration_line_fits(filename) -> CalibrationLineSet:
    """Read a common identified calibration-line FITS product."""
    with fits.open(filename) as hdul:
        header = hdul[0].header
        table = Table(hdul["LINES"].data)
    return CalibrationLineSet(
        table,
        str(header.get("CALTYPE", "")),
        float(header.get("CALSHFTY", np.nan)),
    )
