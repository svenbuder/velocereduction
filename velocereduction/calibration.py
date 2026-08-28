"""Calibration-line measurement for Veloce spectra.

This module currently covers only the part of the wavelength-calibration
pipeline that has been implemented and tested so far:

1. detect emission-line candidates in extracted SimLC/SimTh/FibTh spectra;
2. fit each candidate with a pixel-integrated Gaussian profile;
3. flag saturated, weak, poorly measured, unusually broad, or blended peaks;
4. save one peak-measurement FITS table per calibration type;
5. create optional QA diagnostics.

Line identification (comb-mode / Th-line assignment) and the global wavelength
surface are intentionally kept out of this file until those pipeline stages
are implemented.

Coordinate convention
---------------------
y : dispersion direction
x : cross-dispersion direction
m : physical echelle order
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from astropy.table import Table, vstack
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import ndtr


logger = logging.getLogger(__name__)


# Physical Veloce echelle orders in the same descending order used by the
# extracted calibration arrays.
VELOCE_CCD_ORDERS = {
    "1": np.arange(167, 138 - 1, -1),
    "2": np.arange(140, 103 - 1, -1),
    "3": np.arange(104, 65 - 1, -1),
}


class CalibrationPeakFlag(IntFlag):
    """Bit mask describing why a measured calibration peak is not trusted."""

    GOOD = 0
    EDGE = 1 << 0
    SATURATED = 1 << 1
    LOW_SNR = 1 << 2
    WIDTH_OUTLIER = 1 << 3
    BLEND_CANDIDATE = 1 << 4
    BAD_PROFILE_FIT = 1 << 5
    LARGE_CENTROID_ERROR = 1 << 6


@dataclass
class CalibrationPeakConfig:
    """Settings for detecting, fitting, and quality-checking calibration peaks."""

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
    maximum_sigma: float = 3.00

    # Individual-peak quality cuts.
    minimum_fit_snr: float = 10.0
    maximum_y_uncertainty: float = 0.10

    # Maximum value in the extracted 1D spectrum within the local fit window.
    # Set this separately for SimTh/FibTh if needed. None disables the cut.
    maximum_signal: float | None = None

    # Ensemble rejection within each order.
    fwhm_mad_sigma: float = 4.0
    measured_blend_fwhm_factor: float = 2.0


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def robust_sigma(values: np.ndarray) -> float:
    """Return a Gaussian-equivalent scatter estimated from the MAD."""

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    if not np.any(finite):
        return np.nan

    median = np.nanmedian(values[finite])
    return 1.4826 * np.nanmedian(np.abs(values[finite] - median))


def _normalise_diagnostics(diagnostics: str | None) -> str:
    """Return one of ``none``, ``basic``, or ``full``."""

    if diagnostics is None:
        return "none"

    diagnostics = str(diagnostics).lower()

    if diagnostics not in {"none", "basic", "full"}:
        raise ValueError(
            "diagnostics must be one of 'none', 'basic', or 'full'"
        )

    return diagnostics


def _debug_enabled(log_level: str | int | None) -> bool:
    """Whether verbose per-order console output was requested."""

    if isinstance(log_level, str):
        return log_level.upper() == "DEBUG"

    if isinstance(log_level, int):
        return log_level <= logging.DEBUG

    return logger.isEnabledFor(logging.DEBUG)



def _count_flag(peak_table: Table, flag: CalibrationPeakFlag) -> int:
    """Count rows containing one quality-flag bit."""

    if len(peak_table) == 0:
        return 0

    flags = np.asarray(peak_table["quality_flag"], dtype=np.int64)
    return int(np.sum((flags & int(flag)) != 0))


def _quality_summary(peak_table: Table) -> dict[str, int]:
    """Return counts for the currently implemented quality flags."""

    summary = {
        "total": len(peak_table),
        "accepted": 0,
    }

    if len(peak_table) > 0:
        summary["accepted"] = int(
            np.sum(np.asarray(peak_table["used_for_wavelength_fit"], dtype=bool))
        )

    for flag in CalibrationPeakFlag:
        if flag == CalibrationPeakFlag.GOOD:
            continue
        summary[flag.name.lower()] = _count_flag(peak_table, flag)

    return summary


# -----------------------------------------------------------------------------
# Peak detection
# -----------------------------------------------------------------------------


def detect_calibration_peaks(
    counts: np.ndarray,
    *,
    config: CalibrationPeakConfig | None = None,
):
    """Detect candidate emission peaks in one extracted echelle order.

    Detection is performed on a locally background-subtracted spectrum and
    uses an empirical local-noise estimate. This noise estimate is used only
    for *detection*. It is deliberately not passed to the profile fit because
    a dense comb spectrum can inflate it with real neighbouring peaks.

    Parameters
    ----------
    counts
        One-dimensional extracted counts along the dispersion coordinate y.
    config
        Peak-detection/fitting configuration.

    Returns
    -------
    candidate_pixels, background, detection_noise, detection_snr, properties
        Integer peak locations and diagnostic arrays from the detection step.
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

    # Fill only for the detection filters; the original counts are still used
    # for the profile fits.
    fill_value = float(np.nanmedian(finite_counts))
    working_counts = np.where(np.isfinite(counts), counts, fill_value)

    # Median-filtered background: broad enough not to follow individual narrow
    # calibration peaks.
    background = median_filter(
        working_counts,
        size=config.background_window,
        mode="nearest",
    )

    line_signal = working_counts - background

    # Robust local scatter for peak detection. This does not need to be a
    # perfect statistical variance; it only normalises the find_peaks criteria.
    local_center = median_filter(
        line_signal,
        size=config.noise_window,
        mode="nearest",
    )

    absolute_deviation = np.abs(line_signal - local_center)

    detection_noise = 1.4826 * median_filter(
        absolute_deviation,
        size=config.noise_window,
        mode="nearest",
    )

    global_noise = robust_sigma(line_signal)
    if not np.isfinite(global_noise) or global_noise <= 0:
        global_noise = 1.0

    detection_noise = np.maximum(detection_noise, global_noise)

    detection_snr = np.divide(
        line_signal,
        detection_noise,
        out=np.zeros_like(line_signal),
        where=detection_noise > 0,
    )

    candidate_pixels, properties = find_peaks(
        detection_snr,
        height=config.detection_snr,
        prominence=config.prominence_snr,
        distance=config.minimum_peak_distance,
    )

    return (
        candidate_pixels,
        background,
        detection_noise,
        detection_snr,
        properties,
    )


# -----------------------------------------------------------------------------
# Pixel-integrated line-profile fit
# -----------------------------------------------------------------------------


def pixel_integrated_gaussian(
    y: np.ndarray,
    integrated_counts: float,
    y_center: float,
    sigma: float,
) -> np.ndarray:
    """Evaluate a Gaussian line profile integrated over finite detector pixels."""

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
        + pixel_integrated_gaussian(
            y,
            integrated_counts,
            y_center,
            sigma,
        )
    )


def fit_calibration_peak(
    counts: np.ndarray,
    candidate_pixel: int,
    *,
    background: np.ndarray | None = None,
    config: CalibrationPeakConfig | None = None,
    return_diagnostics: bool = False,
) -> dict:
    """Fit one calibration peak with a pixel-integrated Gaussian.

    The first fit uses a robust loss to reduce the leverage of a deviant pixel.
    A second ordinary least-squares fit starts from that solution so that the
    Jacobian can be used for a simple local covariance estimate.

    We do not yet have a propagated variance spectrum. The fit is therefore
    unweighted, and its covariance is scaled by the measured residual variance
    of the local profile fit. The empirical noise used for peak detection is
    intentionally not reused here.
    """

    if config is None:
        config = CalibrationPeakConfig()

    counts = np.asarray(counts, dtype=float)
    n_pixels = len(counts)

    left = max(0, int(candidate_pixel) - config.fit_half_width)
    right = min(
        n_pixels,
        int(candidate_pixel) + config.fit_half_width + 1,
    )

    y = np.arange(left, right, dtype=float)
    observed_counts = counts[left:right]
    y_reference = float(candidate_pixel)

    finite = np.isfinite(observed_counts)
    if np.count_nonzero(finite) < 6:
        raise ValueError("too few finite pixels in the line-fit window")

    # least_squares cannot handle NaNs. For the very unusual case of an
    # isolated non-finite value, fill it from the local median and flag the fit
    # as bad below if the optimiser does not behave well.
    if not np.all(finite):
        observed_counts = observed_counts.copy()
        observed_counts[~finite] = np.nanmedian(observed_counts[finite])

    edge = left == 0 or right == n_pixels

    maximum_observed_signal = float(np.nanmax(observed_counts))
    saturated = (
        config.maximum_signal is not None
        and maximum_observed_signal >= config.maximum_signal
    )

    # Use the background found during the detection step when available.
    if background is None:
        n_edge = min(2, max(1, len(observed_counts) // 3))
        background_guess = float(
            np.nanmedian(
                np.concatenate(
                    [
                        observed_counts[:n_edge],
                        observed_counts[-n_edge:],
                    ]
                )
            )
        )
    else:
        background_guess = float(background[candidate_pixel])

    peak_height_guess = max(
        float(counts[candidate_pixel]) - background_guess,
        1.0,
    )

    sigma_guess = 0.7
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

    def residuals(parameters):
        model = calibration_line_model(
            y,
            *parameters,
            y_reference=y_reference,
        )
        return model - observed_counts

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

    fit_residual = observed_counts - model
    fit_rms = float(np.sqrt(np.nanmean(fit_residual**2)))

    # Local covariance estimate. Because the fit is currently unweighted,
    # scale (J^T J)^-1 by the measured residual variance.
    n_parameters = len(final_fit.x)
    degrees_of_freedom = max(1, len(y) - n_parameters)
    residual_variance = float(
        np.nansum(fit_residual**2) / degrees_of_freedom
    )

    covariance = np.linalg.pinv(final_fit.jac.T @ final_fit.jac)
    covariance *= residual_variance

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

    flag = CalibrationPeakFlag.GOOD

    if edge:
        flag |= CalibrationPeakFlag.EDGE

    if saturated:
        flag |= CalibrationPeakFlag.SATURATED

    if not final_fit.success or not np.all(np.isfinite(final_fit.x)):
        flag |= CalibrationPeakFlag.BAD_PROFILE_FIT

    if (
        not np.isfinite(signal_to_noise)
        or signal_to_noise < config.minimum_fit_snr
    ):
        flag |= CalibrationPeakFlag.LOW_SNR

    if (
        not np.isfinite(y_uncertainty)
        or y_uncertainty > config.maximum_y_uncertainty
    ):
        flag |= CalibrationPeakFlag.LARGE_CENTROID_ERROR

    result = dict(
        y=float(y_center),
        y_uncertainty=y_uncertainty,
        pixel_phase=float(y_center - np.round(y_center)),
        integrated_counts=float(integrated_counts),
        integrated_counts_uncertainty=integrated_counts_uncertainty,
        maximum_signal=maximum_observed_signal,
        signal_to_noise=signal_to_noise,
        fwhm=fwhm,
        fwhm_uncertainty=fwhm_uncertainty,
        background=float(fitted_background),
        background_slope=float(background_slope),
        fit_rms=fit_rms,
        fit_success=bool(final_fit.success),
        quality_flag=int(flag),
    )

    if return_diagnostics:
        result.update(
            fit_y=y,
            fit_observed=observed_counts,
            fit_model=model,
            fit_residual=fit_residual,
        )

    return result


# -----------------------------------------------------------------------------
# Ensemble quality checks
# -----------------------------------------------------------------------------


def _apply_ensemble_peak_flags(
    peak_table: Table,
    *,
    config: CalibrationPeakConfig,
) -> None:
    """Flag unusual widths and very close measured neighbours per order."""

    if len(peak_table) == 0:
        return

    orders = np.asarray(peak_table["order"], dtype=int)

    for order in np.unique(orders):
        indices = np.where(orders == order)[0]

        # Estimate the normal FWHM using peaks that are not already clearly
        # unusable for reasons independent of their width.
        excluded_for_width_reference = int(
            CalibrationPeakFlag.SATURATED
            | CalibrationPeakFlag.BAD_PROFILE_FIT
            | CalibrationPeakFlag.EDGE
        )

        basic_good = np.array(
            [
                (
                    int(peak_table["quality_flag"][i])
                    & excluded_for_width_reference
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

            if np.isfinite(fwhm_scatter) and fwhm_scatter > 0:
                lower = median_fwhm - config.fwhm_mad_sigma * fwhm_scatter
                upper = median_fwhm + config.fwhm_mad_sigma * fwhm_scatter

                for i in indices:
                    width = float(peak_table["fwhm"][i])
                    if (
                        not np.isfinite(width)
                        or width < lower
                        or width > upper
                    ):
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
            peak_table["nearest_peak_distance_pixel"][table_i] = nearest[
                local_i
            ]

            fwhm = float(peak_table["fwhm"][table_i])

            if (
                np.isfinite(fwhm)
                and nearest[local_i]
                < config.measured_blend_fwhm_factor * fwhm
            ):
                peak_table["quality_flag"][table_i] = int(
                    int(peak_table["quality_flag"][table_i])
                    | int(CalibrationPeakFlag.BLEND_CANDIDATE)
                )

    peak_table["used_for_wavelength_fit"] = (
        np.asarray(peak_table["quality_flag"], dtype=np.int64) == 0
    )


# -----------------------------------------------------------------------------
# Diagnostic plots
# -----------------------------------------------------------------------------


def _plot_calibration_order_diagnostic(
    counts: np.ndarray,
    background: np.ndarray,
    detection_snr: np.ndarray,
    candidate_pixels: np.ndarray,
    order_table: Table,
    *,
    config: CalibrationPeakConfig,
    calibration_type: str,
    ccd: str,
    exposure_index: int,
    order: int,
):
    """Create one full-diagnostic page for a single echelle order."""

    y = np.arange(len(counts), dtype=float)
    good = np.asarray(order_table["used_for_wavelength_fit"], dtype=bool)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2, 1.2, 1.2]},
    )

    # Extracted spectrum and fitted centroids.
    ax = axes[0]
    ax.plot(y, counts, lw=0.7, label="Extracted spectrum")
    ax.plot(y, background, lw=0.8, label="Detection background")
    if calibration_type.lower() in {"simth", "fibth"}:
        ax.set_yscale("log")

    if np.any(good):
        ax.scatter(
            order_table["y"][good],
            np.interp(order_table["y"][good], y, counts),
            s=12,
            label="Accepted",
            zorder=5,
        )

    if np.any(~good):
        ax.scatter(
            order_table["y"][~good],
            np.interp(order_table["y"][~good], y, counts),
            marker="x",
            s=28,
            label="Rejected",
            zorder=6,
        )

    if config.maximum_signal is not None:
        ax.axhline(
            config.maximum_signal,
            ls=":",
            lw=1,
            label="Maximum signal",
        )

    ax.set_ylabel("Counts")
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    ax.set_title(
        f"{calibration_type} | CCD {ccd} | exposure {exposure_index} | "
        f"order {order}"
    )

    # Detection statistic.
    ax = axes[1]
    ax.plot(y, detection_snr, lw=0.7)
    if len(candidate_pixels) > 0:
        ax.scatter(
            candidate_pixels,
            detection_snr[candidate_pixels],
            marker="x",
            s=15,
        )
    if calibration_type.lower() in {"simth", "fibth"}:
        ax.set_yscale("log")
    ax.axhline(config.detection_snr, ls="--", lw=1)
    ax.set_ylabel("Detection S/N")

    # FWHM versus detector position.
    ax = axes[2]
    if np.any(good):
        ax.scatter(
            order_table["y"][good],
            order_table["fwhm"][good].clip(0,4),
            s=10,
            label="Accepted",
        )
    if np.any(~good):
        ax.scatter(
            order_table["y"][~good],
            order_table["fwhm"][~good].clip(0,4),
            marker="x",
            s=20,
            label="Rejected",
        )

    width_reference = np.asarray(order_table["fwhm"][good], dtype=float)
    if np.any(np.isfinite(width_reference)):
        median_fwhm = np.nanmedian(width_reference)
        ax.axhline(median_fwhm, ls="--", lw=1)

    ax.set_ylabel("FWHM [pixel]")

    # Centroid precision and fit RMS.
    ax = axes[3]
    if np.any(good):
        ax.scatter(
            order_table["y"][good],
            order_table["y_uncertainty"][good].clip(0, 0.2),
            s=10,
            label=r"Accepted",
        )
    if np.any(~good):
        ax.scatter(
            order_table["y"][~good],
            order_table["y_uncertainty"][~good].clip(0, 0.2),
            marker="x",
            s=20,
            label=r"Rejected",
        ) 
    ax.axhline(config.maximum_y_uncertainty, ls="--", lw=1)
    ax.set_ylabel(r"$\sigma_y$ [pixel]")
    ax.set_xlabel(r"Dispersion pixel $y$")

    summary = _quality_summary(order_table)
    fig.text(
        0.99,
        0.01,
        (
            f"candidates={len(candidate_pixels)} | fitted={len(order_table)} | "
            f"accepted={summary['accepted']} | "
            f"saturated={summary.get('saturated', 0)} | "
            f"width={summary.get('width_outlier', 0)} | "
            f"blend={summary.get('blend_candidate', 0)}"
        ),
        ha="right",
        va="bottom",
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.025, 1, 1))
    return fig


def _plot_calibration_summary(
    peak_table: Table,
    *,
    calibration_type: str,
    ccd: str,
    exposure_index: int,
    filename: str | Path,
) -> None:
    """Save a compact CCD/exposure-level peak-measurement QA figure."""

    if len(peak_table) == 0:
        return

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    good = np.asarray(peak_table["used_for_wavelength_fit"], dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # FWHM over detector coordinates (y, m).
    ax = axes[0, 0]
    if np.any(good):
        scatter = ax.scatter(
            peak_table["y"][good],
            peak_table["order"][good],
            c=peak_table["fwhm"][good],
            s=6,
        )
        fig.colorbar(scatter, ax=ax, label="FWHM [pixel]")
    ax.set_xlabel(r"Dispersion pixel $y$")
    ax.set_ylabel(r"Echelle order $m$")
    ax.set_title("Line width")

    # Centroid precision.
    ax = axes[0, 1]
    if np.any(good):
        ax.scatter(
            peak_table["y"][good],
            peak_table["y_uncertainty"][good],
            s=6,
        )
    ax.set_xlabel(r"Dispersion pixel $y$")
    ax.set_ylabel(r"$\sigma_y$ [pixel]")
    ax.set_title("Centroid precision")

    # Fitted signal-to-noise.
    ax = axes[1, 0]
    if np.any(good):
        ax.scatter(
            peak_table["y"][good],
            peak_table["signal_to_noise"][good],
            s=6,
        )
    ax.set_xlabel(r"Dispersion pixel $y$")
    ax.set_ylabel("Fitted line S/N")
    ax.set_title("Line signal-to-noise")

    # Pixel-phase distribution.
    ax = axes[1, 1]
    if np.any(good):
        ax.hist(peak_table["pixel_phase"][good], bins=30)
    ax.set_xlabel("Pixel phase")
    ax.set_ylabel("Number of accepted lines")
    ax.set_title("Sub-pixel centroid sampling")

    summary = _quality_summary(peak_table)
    fig.suptitle(
        f"{calibration_type} | CCD {ccd} | exposure {exposure_index} | "
        f"{summary['accepted']}/{summary['total']} accepted"
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_worst_peak_fits(
    counts: np.ndarray,
    order_table: Table,
    *,
    config: CalibrationPeakConfig,
    calibration_type: str,
    ccd: str,
    exposure_index: int,
    order: int,
    max_peaks: int = 8,
):
    """Plot local profiles for rejected or otherwise worst-fitting peaks.

    This is used only in ``diagnostics='full'``. Rejected lines are preferred;
    if an order has fewer rejected lines than ``max_peaks``, the remaining
    panels are filled with the largest fit-RMS measurements.
    """

    if len(order_table) == 0:
        return None

    rejected = np.where(
        ~np.asarray(order_table["used_for_wavelength_fit"], dtype=bool)
    )[0]

    fit_rms = np.asarray(order_table["fit_rms"], dtype=float)
    ranking = np.argsort(np.nan_to_num(fit_rms, nan=-np.inf))[::-1]

    selected = list(rejected[:max_peaks])
    for index in ranking:
        if index not in selected:
            selected.append(int(index))
        if len(selected) >= max_peaks:
            break

    selected = selected[:max_peaks]
    if len(selected) == 0:
        return None

    n_columns = 2
    n_rows = int(np.ceil(len(selected) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, 3.0 * n_rows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, table_index in zip(axes.ravel(), selected):
        ax.set_visible(True)

        candidate_pixel = int(order_table["candidate_pixel"][table_index])
        try:
            fit = fit_calibration_peak(
                counts,
                candidate_pixel,
                config=config,
                return_diagnostics=True,
            )
        except Exception as error:
            ax.text(0.5, 0.5, str(error), ha="center", va="center")
            continue

        ax.scatter(fit["fit_y"], fit["fit_observed"], s=20, label="data")
        ax.plot(fit["fit_y"], fit["fit_model"], lw=1.2, label="fit")
        ax.axvline(fit["y"], ls="--", lw=0.8)

        flag_value = int(order_table["quality_flag"][table_index])
        flag_names = [
            flag.name
            for flag in CalibrationPeakFlag
            if flag != CalibrationPeakFlag.GOOD
            and (flag_value & int(flag)) != 0
        ]
        flag_text = ", ".join(flag_names) if flag_names else "GOOD"

        ax.set_title(
            f"y={fit['y']:.3f}, FWHM={fit['fwhm']:.2f}, "
            f"S/N={fit['signal_to_noise']:.1f}\n{flag_text}",
            fontsize=9,
        )
        ax.set_xlabel(r"Dispersion pixel $y$")
        ax.set_ylabel("Counts")

    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(
        f"{calibration_type} | CCD {ccd} | exposure {exposure_index} | "
        f"order {order}: rejected / worst local fits"
    )
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Measure all peaks in one extracted calibration exposure
# -----------------------------------------------------------------------------


def measure_calibration_peaks(
    counts: np.ndarray,
    orders: np.ndarray,
    *,
    config: CalibrationPeakConfig | None = None,
    calibration_type: str = "",
    ccd: str | int = "",
    exposure_index: int = 0,
    diagnostics: str = "none",
    diagnostic_pdf: str | Path | None = None,
    log_level: str | int | None = None,
) -> Table:
    """Detect and fit all calibration peaks in one extracted exposure.

    Parameters
    ----------
    counts
        Array with shape ``(n_orders, n_dispersion_pixels)``.
    orders
        Physical echelle order corresponding to each row of ``counts``.
    config
        Peak measurement and quality-cut settings.
    calibration_type, ccd, exposure_index
        Labels used only for diagnostics and log output.
    diagnostics
        ``'none'``, ``'basic'`` or ``'full'``. Only ``'full'`` creates the
        per-order PDF here; the exposure summary PNG is created by the nightly
        wrapper after exposure metadata have been added.
    diagnostic_pdf
        Filename for the multi-page PDF used with ``diagnostics='full'``.
    log_level
        When ``'DEBUG'``, print detailed per-order fitting summaries.

    Returns
    -------
    astropy.table.Table
        One row per detected calibration peak. Rejected peaks are retained;
        ``quality_flag`` describes the reason and ``used_for_wavelength_fit``
        is True only for currently accepted peaks.
    """

    if config is None:
        config = CalibrationPeakConfig()

    diagnostics = _normalise_diagnostics(diagnostics)
    debug = _debug_enabled(log_level)

    counts = np.asarray(counts, dtype=float)
    orders = np.asarray(orders, dtype=int)

    if counts.ndim != 2:
        raise ValueError(
            "counts must have shape (n_orders, n_dispersion_pixels)"
        )

    if len(orders) != counts.shape[0]:
        raise ValueError(
            "orders must have one entry for every row in counts"
        )

    rows = []
    order_diagnostics = {}

    for order_index, order in enumerate(orders):
        order_counts = counts[order_index]

        (
            candidate_pixels,
            background,
            detection_noise,
            detection_snr,
            _,
        ) = detect_calibration_peaks(
            order_counts,
            config=config,
        )

        if debug:
            print(
                f"{calibration_type} CCD{ccd} exposure {exposure_index} "
                f"order {order}: {len(candidate_pixels)} candidates"
            )

        n_fit_failed = 0

        for candidate_pixel in candidate_pixels:
            try:
                result = fit_calibration_peak(
                    order_counts,
                    int(candidate_pixel),
                    background=background,
                    config=config,
                )
            except Exception as error:
                n_fit_failed += 1
                logger.warning(
                    "%s CCD%s exposure %s order %s: could not fit peak "
                    "at y=%s: %s",
                    calibration_type,
                    ccd,
                    exposure_index,
                    order,
                    candidate_pixel,
                    error,
                )
                if debug:
                    print(
                        f"    fit failed at y={candidate_pixel}: {error}"
                    )
                continue

            rows.append(
                dict(
                    order=int(order),
                    candidate_pixel=int(candidate_pixel),
                    y=result["y"],
                    y_uncertainty=result["y_uncertainty"],
                    pixel_phase=result["pixel_phase"],
                    fwhm=result["fwhm"],
                    fwhm_uncertainty=result["fwhm_uncertainty"],
                    integrated_counts=result["integrated_counts"],
                    integrated_counts_uncertainty=result[
                        "integrated_counts_uncertainty"
                    ],
                    maximum_signal=result["maximum_signal"],
                    signal_to_noise=result["signal_to_noise"],
                    background=result["background"],
                    background_slope=result["background_slope"],
                    fit_rms=result["fit_rms"],
                    nearest_peak_distance_pixel=np.inf,
                    quality_flag=int(result["quality_flag"]),
                    used_for_wavelength_fit=False,
                )
            )

        order_diagnostics[int(order)] = dict(
            counts=order_counts,
            background=background,
            detection_noise=detection_noise,
            detection_snr=detection_snr,
            candidate_pixels=candidate_pixels,
            n_fit_failed=n_fit_failed,
        )

    peak_table = Table(rows=rows)

    if len(peak_table) > 0:
        _apply_ensemble_peak_flags(
            peak_table,
            config=config,
        )

    # Per-order summaries are most useful after all ensemble flags have been
    # applied, because accepted/rejected then represents the final state of the
    # current peak-measurement stage.
    if debug:
        for order in orders:
            order_mask = (
                np.asarray(peak_table["order"], dtype=int) == int(order)
                if len(peak_table) > 0
                else np.array([], dtype=bool)
            )
            order_table = peak_table[order_mask]
            diagnostic = order_diagnostics[int(order)]

            if len(order_table) == 0:
                print(
                    f"    order {order}: 0 fitted peaks "
                    f"({diagnostic['n_fit_failed']} fit failures)"
                )
                continue

            summary = _quality_summary(order_table)
            print(
                f"    order {order}: "
                f"{summary['accepted']}/{summary['total']} accepted; "
                f"median FWHM={np.nanmedian(order_table['fwhm']):.3f} pix; "
                f"median sigma_y={np.nanmedian(order_table['y_uncertainty']):.4f} pix; "
                f"median S/N={np.nanmedian(order_table['signal_to_noise']):.1f}; "
                f"sat={summary.get('saturated', 0)}, "
                f"width={summary.get('width_outlier', 0)}, "
                f"blend={summary.get('blend_candidate', 0)}, "
                f"lowS/N={summary.get('low_snr', 0)}"
            )

    if diagnostics == "full" and diagnostic_pdf is not None:
        diagnostic_pdf = Path(diagnostic_pdf)
        diagnostic_pdf.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(diagnostic_pdf) as pdf:
            for order in orders:
                order_mask = (
                    np.asarray(peak_table["order"], dtype=int) == int(order)
                    if len(peak_table) > 0
                    else np.array([], dtype=bool)
                )
                order_table = peak_table[order_mask]
                diagnostic = order_diagnostics[int(order)]

                if len(order_table) == 0:
                    continue

                fig = _plot_calibration_order_diagnostic(
                    diagnostic["counts"],
                    diagnostic["background"],
                    diagnostic["detection_snr"],
                    diagnostic["candidate_pixels"],
                    order_table,
                    config=config,
                    calibration_type=calibration_type,
                    ccd=str(ccd),
                    exposure_index=exposure_index,
                    order=int(order),
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # A second page shows a small set of rejected/worst local fits.
                fit_fig = _plot_worst_peak_fits(
                    diagnostic["counts"],
                    order_table,
                    config=config,
                    calibration_type=calibration_type,
                    ccd=str(ccd),
                    exposure_index=exposure_index,
                    order=int(order),
                )
                if fit_fig is not None:
                    pdf.savefig(fit_fig, bbox_inches="tight")
                    plt.close(fit_fig)

        if debug:
            print(f"    full diagnostics -> {diagnostic_pdf}")

    return peak_table


# -----------------------------------------------------------------------------
# High-level nightly interface used by reduce_night.py / reduce_night.ipynb
# -----------------------------------------------------------------------------


def measure_calibration_peaks_for_night(
    calibration_spectra: dict,
    *,
    output_dir: str | Path | None = None,
    diagnostic_dir: str | Path | None = None,
    calibration_types: tuple[str, ...] = ("SimLC", "SimTh", "FibTh"),
    maximum_signal: dict[str, float | None] | float | None = None,
    minimum_signal_to_noise: float = 10.0,
    maximum_y_uncertainty: float = 0.10,
    detection_snr: float = 5.0,
    prominence_snr: float = 4.0,
    minimum_peak_distance: int = 3,
    fit_half_width: int = 4,
    diagnostics: str = "basic",
    log_level: str | int | None = None,
    overwrite: bool = False,
) -> dict[str, Table]:
    """Measure calibration peaks for all requested exposures in one night.

    This is the only peak-measurement function that should normally be called
    from ``reduce_night.py`` or ``reduce_night.ipynb``.

    It loops over calibration type, CCD, exposure, and echelle order; combines
    the resulting measurements into one table per calibration type; optionally
    saves those tables as FITS files; and returns them so later wavelength-
    identification code can continue in memory.

    Diagnostics
    -----------
    diagnostics='none'
        No figures are created.
    diagnostics='basic'
        Save one summary PNG for every calibration type / CCD / exposure.
    diagnostics='full'
        Save the basic summary PNG plus a multi-page PDF containing an order
        overview and rejected/worst local line fits for every order.

    When ``log_level='DEBUG'`` a detailed per-order summary is printed while
    the peak measurements are being made.
    """

    diagnostics = _normalise_diagnostics(diagnostics)
    debug = _debug_enabled(log_level)

    if overwrite == False:
        try:
            calibration_peak_tables = {}
            for calibration_type in ['SimTh','SimLC','FibTh']:
                calibration_peak_tables[calibration_type] = Table.read(
                    output_dir / f'{calibration_type.lower()}_peaks.fits',
                    1,
                )
            print(f'Read in existing peak tables from {output_dir}, skipping peak measurement')
            return(calibration_peak_tables)
        except:
            print(f'Could not find SimTh_peaks.fits, SimLC_peaks.fits and/or FibTh_peaks.fits in {output_dir}, measuring peaks from scratch')

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if diagnostic_dir is not None:
        diagnostic_dir = Path(diagnostic_dir)
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    peak_tables = {}

    for calibration_type in calibration_types:
        if calibration_type not in calibration_spectra:
            continue

        if isinstance(maximum_signal, dict):
            calibration_maximum_signal = maximum_signal.get(
                calibration_type,
                None,
            )
        else:
            calibration_maximum_signal = maximum_signal

        config = CalibrationPeakConfig(
            detection_snr=detection_snr,
            prominence_snr=prominence_snr,
            minimum_peak_distance=minimum_peak_distance,
            fit_half_width=fit_half_width,
            minimum_fit_snr=minimum_signal_to_noise,
            maximum_y_uncertainty=maximum_y_uncertainty,
            maximum_signal=calibration_maximum_signal,
        )

        calibration_tables = []

        if debug:
            print("\n" + "=" * 78)
            print(f"Measuring {calibration_type} calibration peaks")
            print(
                f"  detection S/N >= {config.detection_snr:.1f}; "
                f"prominence >= {config.prominence_snr:.1f}; "
                f"fit S/N >= {config.minimum_fit_snr:.1f}; "
                f"sigma_y <= {config.maximum_y_uncertainty:.3f} pix; "
                f"maximum signal = {config.maximum_signal}"
            )

        for ccd, exposures in calibration_spectra[calibration_type].items():
            ccd = str(ccd)

            if ccd not in VELOCE_CCD_ORDERS:
                raise ValueError(f'Unknown CCD "{ccd}"')

            for exposure_index, exposure in enumerate(exposures):
                counts = np.asarray(exposure["counts"], dtype=float)

                orders = np.asarray(
                    exposure.get("orders", VELOCE_CCD_ORDERS[ccd]),
                    dtype=int,
                )

                if counts.ndim != 2:
                    raise ValueError(
                        f"{calibration_type}, CCD {ccd}, exposure "
                        f"{exposure_index}: counts must be a 2D "
                        "(n_orders, n_dispersion_pixels) array"
                    )

                if counts.shape[0] != len(orders):
                    raise ValueError(
                        f"{calibration_type}, CCD {ccd}, exposure "
                        f"{exposure_index}: counts contains "
                        f"{counts.shape[0]} orders but {len(orders)} order "
                        "numbers were supplied"
                    )

                if debug:
                    print(
                        f"\n{calibration_type} CCD{ccd} exposure "
                        f"{exposure_index}: {counts.shape[0]} orders x "
                        f"{counts.shape[1]} dispersion pixels"
                    )

                diagnostic_pdf = None
                if diagnostics == "full" and diagnostic_dir is not None:
                    diagnostic_pdf = diagnostic_dir / (
                        f"{calibration_type.lower()}_ccd{ccd}_"
                        f"exposure{exposure_index:03d}_peaks.pdf"
                    )

                peak_table = measure_calibration_peaks(
                    counts,
                    orders,
                    config=config,
                    calibration_type=calibration_type,
                    ccd=ccd,
                    exposure_index=exposure_index,
                    diagnostics=diagnostics,
                    diagnostic_pdf=diagnostic_pdf,
                    log_level=log_level,
                )

                if len(peak_table) == 0:
                    logger.warning(
                        "%s CCD%s exposure %s: no calibration peaks measured",
                        calibration_type,
                        ccd,
                        exposure_index,
                    )
                    continue

                n_peaks = len(peak_table)

                # Exposure metadata needed to distinguish rows once all CCDs
                # and exposures are stacked into one table.
                peak_table["calibration_type"] = np.full(
                    n_peaks,
                    calibration_type,
                    dtype="U8",
                )
                peak_table["ccd"] = np.full(
                    n_peaks,
                    int(ccd),
                    dtype=int,
                )
                peak_table["exposure_index"] = np.full(
                    n_peaks,
                    exposure_index,
                    dtype=int,
                )

                mjd_mid = exposure.get(
                    "mjd_mid",
                    exposure.get("mjd", np.nan),
                )
                peak_table["mjd_mid"] = np.full(
                    n_peaks,
                    float(mjd_mid),
                    dtype=float,
                )

                calibration_tables.append(peak_table)

                summary = _quality_summary(peak_table)

                if debug:
                    print(
                        f"  CCD{ccd} exposure {exposure_index} total: "
                        f"{summary['accepted']}/{summary['total']} accepted; "
                        f"saturated={summary.get('saturated', 0)}, "
                        f"width={summary.get('width_outlier', 0)}, "
                        f"blend={summary.get('blend_candidate', 0)}, "
                        f"lowS/N={summary.get('low_snr', 0)}, "
                        f"large sigma_y={summary.get('large_centroid_error', 0)}"
                    )

                if diagnostics in {"basic", "full"} and diagnostic_dir is not None:
                    summary_filename = diagnostic_dir / (
                        f"{calibration_type.lower()}_ccd{ccd}_"
                        f"exposure{exposure_index:03d}_summary.png"
                    )
                    _plot_calibration_summary(
                        peak_table,
                        calibration_type=calibration_type,
                        ccd=ccd,
                        exposure_index=exposure_index,
                        filename=summary_filename,
                    )
                    if debug:
                        print(f"    summary diagnostic -> {summary_filename}")

        if len(calibration_tables) == 0:
            peak_tables[calibration_type] = Table()
            continue

        combined_table = vstack(
            calibration_tables,
            metadata_conflicts="silent",
        )

        peak_tables[calibration_type] = combined_table

        combined_summary = _quality_summary(combined_table)

        if output_dir is not None:
            filename = output_dir / f"{calibration_type.lower()}_peaks.fits"
            combined_table.write(
                filename,
                format="fits",
                overwrite=overwrite,
            )

            print(
                f"{calibration_type}: {combined_summary['total']} peaks measured, "
                f"{combined_summary['accepted']} pass quality cuts -> {filename}"
            )
        else:
            print(
                f"{calibration_type}: {combined_summary['total']} peaks measured, "
                f"{combined_summary['accepted']} pass quality cuts"
            )

    return peak_tables
