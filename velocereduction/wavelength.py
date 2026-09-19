"""Global wavelength-surface fitting for Veloce echelle spectra.

Recommended destination:
    velocereduction/wavelength.py

Coordinate convention:
    y = dispersion-direction detector coordinate
    x = cross-dispersion detector coordinate
    m = physical echelle order

The fitted quantity is explicitly called ``m_times_lambda`` throughout:

    m_times_lambda(y, m)
        = sum_ij c_ij L_i(y_normalised) L_j(m_normalised)

and

    wavelength(y, m) = m_times_lambda(y, m) / m.

Peak measurement and identification are source-specific: ``thorium.py``
handles FibTh/SimTh and ``simlc.py`` handles the laser comb and its LSF.  This
module only consumes their common identified-line table and fits/evaluates
wavelength models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from numpy.polynomial.legendre import legder, legval2d, legvander

from .calibration import CalibrationPeakFlag
from .constants import CCD_ORDERS, N_DISPERSION, SCIENCE_FIBRES


SPEED_OF_LIGHT_MPS = 299_792_458.0

# Production Veloce wavelength-surface defaults.
WAVELENGTH_Y_DEGREES = 7
WAVELENGTH_ORDER_DEGREES = 5
WAVELENGTH_Y_BOUNDS = (0.0, float(N_DISPERSION - 1))


def _full_order_bounds(order):
    """Return the full physical order range of the CCD containing ``order``."""
    values = np.asarray(order, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("Cannot infer CCD order bounds from an empty order array")
    lower, upper = float(np.nanmin(finite)), float(np.nanmax(finite))
    matches = []
    for ccd, ccd_orders in CCD_ORDERS.items():
        lo, hi = float(np.min(ccd_orders)), float(np.max(ccd_orders))
        if lower >= lo and upper <= hi:
            matches.append((ccd, (lo, hi)))
    if len(matches) != 1:
        raise ValueError(
            "Could not uniquely infer a Veloce CCD from order range "
            f"[{lower:g}, {upper:g}]"
        )
    return matches[0][1]


def _normalise_coordinate(values, bounds):
    """Map a detector coordinate onto the Legendre interval [-1, +1]."""

    values = np.asarray(values, dtype=float)
    lower, upper = map(float, bounds)

    if not upper > lower:
        raise ValueError("Coordinate bounds must satisfy upper > lower")

    center = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)

    return (values - center) / scale, center, scale


def _build_design_matrix(
    y_normalised,
    order_normalised,
    y_degree,
    order_degree,
):
    """Construct all L_i(y) * L_j(m) terms of the 2D Legendre surface."""

    y_basis = legvander(y_normalised, y_degree)
    order_basis = legvander(order_normalised, order_degree)

    return np.einsum(
        "ni,nj->nij",
        y_basis,
        order_basis,
    ).reshape(len(y_normalised), -1)


def _mad_std(values):
    """Gaussian-equivalent robust scatter from the MAD."""

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan

    median = np.nanmedian(values[finite])
    return 1.4826 * np.nanmedian(np.abs(values[finite] - median))


def _safe_residual_scale(residual, reference_scale):
    """Return a robust residual scale with a floating-point noise floor."""

    residual = np.asarray(residual, dtype=float)
    scale = _mad_std(residual)

    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(residual)

    numerical_floor = (
        100.0
        * np.finfo(float).eps
        * max(1.0, float(reference_scale))
    )

    if not np.isfinite(scale):
        return numerical_floor

    return max(float(scale), numerical_floor)


@dataclass
class WavelengthSolution:
    """A fitted global Veloce wavelength surface for one CCD/fibre state."""

    coefficients: np.ndarray
    y_center: float
    y_scale: float
    order_center: float
    order_scale: float
    covariance: np.ndarray | None = None

    def _normalised_coordinates(self, y, order):
        y, order = np.broadcast_arrays(
            np.asarray(y, dtype=float),
            np.asarray(order, dtype=float),
        )

        return (
            (y - self.y_center) / self.y_scale,
            (order - self.order_center) / self.order_scale,
        )

    def m_times_lambda(self, y, order):
        """Evaluate m * lambda at dispersion pixel y and echelle order m."""

        y_normalised, order_normalised = self._normalised_coordinates(
            y,
            order,
        )

        return legval2d(
            y_normalised,
            order_normalised,
            self.coefficients,
        )

    def wavelength(self, y, order):
        """Evaluate wavelength at dispersion pixel y and echelle order m."""

        order = np.asarray(order, dtype=float)
        if np.any(order == 0):
            raise ValueError("Echelle order must be non-zero")

        return self.m_times_lambda(y, order) / order

    def dispersion(self, y, order):
        """Evaluate d(lambda)/dy in wavelength units per detector pixel."""

        y_normalised, order_normalised = self._normalised_coordinates(
            y,
            order,
        )

        derivative_coefficients = legder(
            self.coefficients,
            axis=0,
        )

        derivative_m_times_lambda = (
            legval2d(
                y_normalised,
                order_normalised,
                derivative_coefficients,
            )
            / self.y_scale
        )

        return derivative_m_times_lambda / np.asarray(order, dtype=float)


@dataclass
class WavelengthFitResult:
    """Global wavelength solution plus line-by-line fit diagnostics."""

    solution: WavelengthSolution
    used: np.ndarray
    robust_weight: np.ndarray
    residual_wavelength: np.ndarray
    residual_pixel: np.ndarray
    residual_velocity: np.ndarray
    normalised_residual: np.ndarray
    n_iterations: int


def fit_wavelength_surface(
    y,
    order,
    wavelength,
    *,
    y_uncertainty=None,
    wavelength_uncertainty=None,
    y_degree=WAVELENGTH_Y_DEGREES,
    order_degree=WAVELENGTH_ORDER_DEGREES,
    y_bounds=None,
    order_bounds=None,
    max_iterations=12,
    huber_k=1.5,
    clip_sigma=6.0,
):
    """Fit a robust global ``m_times_lambda(y, order)`` surface.

    Parameters
    ----------
    y
        Measured dispersion-direction peak positions in current detector
        coordinates.
    order
        Physical echelle order m.
    wavelength
        Assigned reference wavelength.  All wavelength inputs/outputs must use
        the same unit; the pipeline uses nm.
    y_uncertainty
        1-sigma centroid uncertainty in detector pixels.
    wavelength_uncertainty
        1-sigma laboratory/reference wavelength uncertainty.
    y_degree, order_degree
        Legendre degrees in detector y and order.
    y_bounds
        Fixed full dispersion-coordinate range for the CCD, e.g. (0, 4111).
        Keep this fixed across exposures so coefficients are comparable in time.
    order_bounds
        Fixed physical order range for the CCD.

    Notes
    -----
    The detector shift is *not* subtracted here.  It was used earlier to
    identify which reference line corresponds to each measured peak.  The
    final wavelength solution is always fitted at the actual current detector
    coordinate y.
    """

    y = np.asarray(y, dtype=float)
    order = np.asarray(order, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)

    if not (y.shape == order.shape == wavelength.shape):
        raise ValueError("y, order, and wavelength must have identical shapes")
    if y.ndim != 1:
        raise ValueError("Calibration-line inputs must be one-dimensional")

    if y_uncertainty is None:
        y_uncertainty = np.zeros_like(y)
    else:
        y_uncertainty = np.broadcast_to(
            np.asarray(y_uncertainty, dtype=float),
            y.shape,
        ).copy()

    if wavelength_uncertainty is None:
        wavelength_uncertainty = np.zeros_like(wavelength)
    else:
        wavelength_uncertainty = np.broadcast_to(
            np.asarray(wavelength_uncertainty, dtype=float),
            wavelength.shape,
        ).copy()

    if np.any(y_uncertainty < 0):
        raise ValueError("y_uncertainty must be non-negative")
    if np.any(wavelength_uncertainty < 0):
        raise ValueError("wavelength_uncertainty must be non-negative")

    finite = (
        np.isfinite(y)
        & np.isfinite(order)
        & np.isfinite(wavelength)
        & np.isfinite(y_uncertainty)
        & np.isfinite(wavelength_uncertainty)
        & (order != 0)
    )

    if not np.any(finite):
        raise ValueError("No finite calibration lines were supplied")

    if y_bounds is None:
        y_bounds = WAVELENGTH_Y_BOUNDS

    if order_bounds is None:
        order_bounds = _full_order_bounds(order[finite])

    y_normalised, y_center, y_scale = _normalise_coordinate(
        y,
        y_bounds,
    )
    order_normalised, order_center, order_scale = _normalise_coordinate(
        order,
        order_bounds,
    )

    design_matrix = _build_design_matrix(
        y_normalised,
        order_normalised,
        y_degree=y_degree,
        order_degree=order_degree,
    )

    m_times_lambda = order * wavelength
    n_parameters = design_matrix.shape[1]

    if np.count_nonzero(finite) <= n_parameters:
        raise ValueError(
            "Not enough finite calibration lines for the requested surface: "
            f"{np.count_nonzero(finite)} lines for {n_parameters} coefficients"
        )

    used = finite.copy()
    robust_weight = np.ones_like(y, dtype=float)

    solution = None
    sigma_m_times_lambda = None
    effective_sigma = None

    for iteration in range(1, max_iterations + 1):

        # --------------------------------------------------------------
        # 1. Translate centroid/reference errors into m * lambda.
        # --------------------------------------------------------------
        sigma_lambda_squared = wavelength_uncertainty**2

        if solution is not None:
            sigma_lambda_squared += (
                solution.dispersion(y, order) * y_uncertainty
            ) ** 2

        sigma_lambda = np.sqrt(sigma_lambda_squared)

        positive_uncertainty = (
            finite
            & np.isfinite(sigma_lambda)
            & (sigma_lambda > 0)
        )

        if np.any(positive_uncertainty):
            typical_sigma_lambda = float(
                np.nanmedian(sigma_lambda[positive_uncertainty])
            )
            sigma_lambda = np.where(
                positive_uncertainty,
                sigma_lambda,
                typical_sigma_lambda,
            )
            sigma_m_times_lambda = np.abs(order) * sigma_lambda
        else:
            sigma_m_times_lambda = None

        # --------------------------------------------------------------
        # 2. Weighted linear least squares.
        # --------------------------------------------------------------
        weights = robust_weight.copy()

        if sigma_m_times_lambda is not None:
            weights /= sigma_m_times_lambda**2

        weights[~used] = 0.0

        sqrt_weights = np.sqrt(weights[used])
        weighted_design_matrix = (
            design_matrix[used] * sqrt_weights[:, None]
        )
        weighted_m_times_lambda = (
            m_times_lambda[used] * sqrt_weights
        )

        coefficient_vector, *_ = np.linalg.lstsq(
            weighted_design_matrix,
            weighted_m_times_lambda,
            rcond=None,
        )

        coefficients = coefficient_vector.reshape(
            y_degree + 1,
            order_degree + 1,
        )

        solution = WavelengthSolution(
            coefficients=coefficients,
            y_center=y_center,
            y_scale=y_scale,
            order_center=order_center,
            order_scale=order_scale,
        )

        residual_m_times_lambda = (
            m_times_lambda - solution.m_times_lambda(y, order)
        )

        # --------------------------------------------------------------
        # 3. Robust residual scale.
        #
        # Formal centroid errors can be much smaller than an imperfect
        # low-order surface.  Adding the robust ensemble scatter to the
        # clipping scale prevents an early iteration from rejecting most
        # perfectly sensible lines simply because the model is not yet final.
        # --------------------------------------------------------------
        robust_scatter = _safe_residual_scale(
            residual_m_times_lambda[used],
            reference_scale=np.nanmedian(np.abs(m_times_lambda[finite])),
        )

        if sigma_m_times_lambda is None:
            effective_sigma = np.full_like(
                residual_m_times_lambda,
                robust_scatter,
                dtype=float,
            )
        else:
            effective_sigma = np.sqrt(
                sigma_m_times_lambda**2 + robust_scatter**2
            )

        normalised_residual = (
            residual_m_times_lambda / effective_sigma
        )

        # --------------------------------------------------------------
        # 4. Huber weighting + wider hard clipping.
        # --------------------------------------------------------------
        new_used = (
            finite
            & np.isfinite(normalised_residual)
            & (np.abs(normalised_residual) <= clip_sigma)
        )

        if np.count_nonzero(new_used) <= n_parameters:
            raise RuntimeError(
                "Robust clipping left too few calibration lines to fit "
                "the requested wavelength surface"
            )

        absolute_residual = np.abs(normalised_residual)
        new_robust_weight = np.ones_like(y, dtype=float)

        downweight = (
            np.isfinite(absolute_residual)
            & (absolute_residual > huber_k)
        )
        new_robust_weight[downweight] = (
            huber_k / absolute_residual[downweight]
        )
        new_robust_weight[~finite] = 0.0

        converged = (
            np.array_equal(new_used, used)
            and np.nanmax(np.abs(new_robust_weight - robust_weight)) < 1e-3
        )

        used = new_used
        robust_weight = new_robust_weight

        if converged:
            break

    # ------------------------------------------------------------------
    # Approximate formal coefficient covariance.
    # ------------------------------------------------------------------
    final_weights = robust_weight.copy()
    if sigma_m_times_lambda is not None:
        final_weights /= sigma_m_times_lambda**2
    final_weights[~used] = 0.0

    normal_matrix = (
        design_matrix[used].T
        @ (
            design_matrix[used]
            * final_weights[used, None]
        )
    )
    covariance = np.linalg.pinv(normal_matrix)

    if sigma_m_times_lambda is None:
        final_residual = (
            m_times_lambda[used]
            - design_matrix[used] @ solution.coefficients.ravel()
        )
        degrees_of_freedom = max(
            1,
            np.count_nonzero(used) - n_parameters,
        )
        residual_variance = (
            np.sum(final_weights[used] * final_residual**2)
            / degrees_of_freedom
        )
        covariance *= residual_variance

    solution.covariance = covariance

    # ------------------------------------------------------------------
    # Diagnostics in wavelength, detector pixels, and velocity.
    # Sign convention is measured/reference wavelength minus model.
    # ------------------------------------------------------------------
    model_wavelength = solution.wavelength(y, order)
    residual_wavelength = wavelength - model_wavelength

    local_dispersion = solution.dispersion(y, order)
    residual_pixel = np.divide(
        residual_wavelength,
        local_dispersion,
        out=np.full_like(residual_wavelength, np.nan),
        where=np.abs(local_dispersion) > 0,
    )

    residual_velocity = (
        SPEED_OF_LIGHT_MPS
        * residual_wavelength
        / wavelength
    )

    if effective_sigma is None:
        final_normalised_residual = np.full_like(y, np.nan)
    else:
        final_normalised_residual = (
            order * residual_wavelength / effective_sigma
        )

    return WavelengthFitResult(
        solution=solution,
        used=used,
        robust_weight=robust_weight,
        residual_wavelength=residual_wavelength,
        residual_pixel=residual_pixel,
        residual_velocity=residual_velocity,
        normalised_residual=final_normalised_residual,
        n_iterations=iteration,
    )


# -----------------------------------------------------------------------------
# Peak-table wrapper
# -----------------------------------------------------------------------------


def fit_wavelength_from_peak_table(
    peak_table: Table,
    *,
    y_bounds: tuple[float, float] | None = None,
    order_bounds: tuple[float, float] | None = None,
    y_degree: int = WAVELENGTH_Y_DEGREES,
    order_degree: int = WAVELENGTH_ORDER_DEGREES,
    max_iterations: int = 12,
) -> tuple[WavelengthFitResult, Table]:
    """Fit a global wavelength surface from an identified calibration table."""

    peak_table = peak_table.copy(copy_data=True)

    good = np.asarray(peak_table["used_for_wavelength_fit"], dtype=bool)
    good &= np.isfinite(np.asarray(peak_table["y"], dtype=float))
    good &= np.isfinite(np.asarray(peak_table["wavelength_nm"], dtype=float))

    fit_indices = np.where(good)[0]
    if len(fit_indices) == 0:
        raise RuntimeError("No calibration peaks are available for wavelength fitting")

    wavelength_uncertainty = np.asarray(
        peak_table["wavelength_uncertainty_nm"][fit_indices],
        dtype=float,
    )

    # The Murphy table does not provide a usable uncertainty for every line.
    # Keep NaN in the persistent peak table but treat unknown reference errors
    # as zero here rather than dropping otherwise useful lines.
    wavelength_uncertainty_for_fit = np.where(
        np.isfinite(wavelength_uncertainty),
        wavelength_uncertainty,
        0.0,
    )

    fit = fit_wavelength_surface(
        y=np.asarray(peak_table["y"][fit_indices], dtype=float),
        order=np.asarray(peak_table["order"][fit_indices], dtype=float),
        wavelength=np.asarray(
            peak_table["wavelength_nm"][fit_indices],
            dtype=float,
        ),
        y_uncertainty=np.asarray(
            peak_table["y_uncertainty"][fit_indices],
            dtype=float,
        ),
        wavelength_uncertainty=wavelength_uncertainty_for_fit,
        y_degree=y_degree,
        order_degree=order_degree,
        y_bounds=y_bounds,
        order_bounds=order_bounds,
        max_iterations=max_iterations,
    )

    for name in [
        "wavelength_residual_nm",
        "pixel_residual",
        "velocity_residual_mps",
    ]:
        peak_table[name] = np.full(len(peak_table), np.nan, dtype=float)

    for local_i, table_i in enumerate(fit_indices):
        peak_table["wavelength_residual_nm"][table_i] = (
            fit.residual_wavelength[local_i]
        )
        peak_table["pixel_residual"][table_i] = (
            fit.residual_pixel[local_i]
        )
        peak_table["velocity_residual_mps"][table_i] = (
            fit.residual_velocity[local_i]
        )

        if not fit.used[local_i]:
            peak_table["used_for_wavelength_fit"][table_i] = False
            peak_table["quality_flag"][table_i] = int(
                int(peak_table["quality_flag"][table_i])
                | int(CalibrationPeakFlag.WAVELENGTH_OUTLIER)
            )

    return fit, peak_table


# -----------------------------------------------------------------------------
# FITS serialization
# -----------------------------------------------------------------------------


def make_wavelength_coefficient_table(fit: WavelengthFitResult) -> Table:
    """Convert the 2D Legendre coefficient array to a FITS-friendly table."""

    coefficients = np.asarray(fit.solution.coefficients, dtype=float)
    covariance = fit.solution.covariance

    if covariance is None:
        coefficient_uncertainty = np.full(coefficients.shape, np.nan)
    else:
        coefficient_uncertainty = np.sqrt(
            np.clip(np.diag(covariance), 0.0, None)
        ).reshape(coefficients.shape)

    rows = []
    for y_degree in range(coefficients.shape[0]):
        for order_degree in range(coefficients.shape[1]):
            rows.append(
                dict(
                    y_degree=int(y_degree),
                    order_degree=int(order_degree),
                    coefficient=float(coefficients[y_degree, order_degree]),
                    coefficient_uncertainty=float(
                        coefficient_uncertainty[y_degree, order_degree]
                    ),
                )
            )

    return Table(rows=rows)


def make_order_wavelength_qa_table(peak_table: Table) -> Table:
    """Create one-row-per-order wavelength-calibration QA statistics."""

    rows = []
    orders = np.asarray(peak_table["order"], dtype=int)
    flags = np.asarray(peak_table["quality_flag"], dtype=np.int64)

    for order in np.unique(orders):
        in_order = orders == order
        identified = in_order & np.isfinite(
            np.asarray(peak_table["wavelength_nm"], dtype=float)
        )
        used = in_order & np.asarray(
            peak_table["used_for_wavelength_fit"],
            dtype=bool,
        )

        velocity = np.asarray(
            peak_table["velocity_residual_mps"],
            dtype=float,
        )[used]
        pixel = np.asarray(
            peak_table["pixel_residual"],
            dtype=float,
        )[used]
        fwhm = np.asarray(peak_table["fwhm"], dtype=float)[in_order]
        snr = np.asarray(
            peak_table["signal_to_noise"],
            dtype=float,
        )[in_order]

        rows.append(
            dict(
                order=int(order),
                n_peaks=int(np.count_nonzero(in_order)),
                n_identified=int(np.count_nonzero(identified)),
                n_used=int(np.count_nonzero(used)),
                n_saturated=int(
                    np.count_nonzero(
                        in_order
                        & ((flags & int(CalibrationPeakFlag.SATURATED)) != 0)
                    )
                ),
                n_width_outlier=int(
                    np.count_nonzero(
                        in_order
                        & ((flags & int(CalibrationPeakFlag.WIDTH_OUTLIER)) != 0)
                    )
                ),
                n_blend=int(
                    np.count_nonzero(
                        in_order
                        & (
                            (
                                flags
                                & int(
                                    CalibrationPeakFlag.BLEND_CANDIDATE
                                    | CalibrationPeakFlag.ATLAS_BLEND
                                )
                            )
                            != 0
                        )
                    )
                ),
                rms_pixel=(
                    float(np.sqrt(np.nanmean(pixel**2)))
                    if len(pixel)
                    else np.nan
                ),
                rms_velocity_mps=(
                    float(np.sqrt(np.nanmean(velocity**2)))
                    if len(velocity)
                    else np.nan
                ),
                median_abs_velocity_mps=(
                    float(np.nanmedian(np.abs(velocity)))
                    if len(velocity)
                    else np.nan
                ),
                median_fwhm=(
                    float(np.nanmedian(fwhm)) if len(fwhm) else np.nan
                ),
                median_signal_to_noise=(
                    float(np.nanmedian(snr)) if len(snr) else np.nan
                ),
            )
        )

    return Table(rows=rows)


def write_wavelength_fit_fits(
    fit: WavelengthFitResult,
    peak_table: Table,
    filename: str | Path,
    *,
    detector_shift_y: float = np.nan,
    detector_shift_y_uncertainty: float = np.nan,
    calibration_shift_y: float = np.nan,
    calibration_type: str = "",
    ccd: int | str | None = None,
    mjd_mid: float = np.nan,
    source_peak_file: str | None = None,
    overwrite: bool = False,
) -> None:
    """Save wavelength coefficients, covariance, fitted lines, and QA to FITS."""

    primary = fits.PrimaryHDU()
    header = primary.header

    header["ORIGIN"] = "velocereduction"
    header["CONTENT"] = "Global echelle wavelength solution"
    header["CALTYPE"] = str(calibration_type)
    if ccd is not None:
        header["CCD"] = str(ccd)
    if np.isfinite(mjd_mid):
        header["MJD-MID"] = float(mjd_mid)

    if np.isfinite(detector_shift_y):
        header["DETSHFTY"] = (
            float(detector_shift_y),
            "y_current - y_reference [pix]",
        )
    if np.isfinite(detector_shift_y_uncertainty):
        header["DETSHYER"] = (
            float(detector_shift_y_uncertainty),
            "uncertainty of DETSHFTY [pix]",
        )
    if np.isfinite(calibration_shift_y):
        header["CALSHFTY"] = (
            float(calibration_shift_y),
            "residual measured-predicted y shift [pix]",
        )

    solution = fit.solution
    header["YCENTER"] = float(solution.y_center)
    header["YSCALE"] = float(solution.y_scale)
    header["MCENTER"] = float(solution.order_center)
    header["MSCALE"] = float(solution.order_scale)
    header["YDEG"] = int(solution.coefficients.shape[0] - 1)
    header["MDEG"] = int(solution.coefficients.shape[1] - 1)
    header["NLINE"] = int(len(peak_table))
    header["NUSED"] = int(
        np.count_nonzero(peak_table["used_for_wavelength_fit"])
    )
    header["NITER"] = int(fit.n_iterations)

    used = np.asarray(peak_table["used_for_wavelength_fit"], dtype=bool)
    velocity = np.asarray(
        peak_table["velocity_residual_mps"],
        dtype=float,
    )[used]
    pixel = np.asarray(
        peak_table["pixel_residual"],
        dtype=float,
    )[used]

    if len(pixel):
        header["RMSPIX"] = float(np.sqrt(np.nanmean(pixel**2)))
    if len(velocity):
        header["RMSMPS"] = float(np.sqrt(np.nanmean(velocity**2)))
        header["MADMPS"] = float(np.nanmedian(np.abs(velocity)))

    if source_peak_file is not None:
        header["PEAKFILE"] = Path(source_peak_file).name

    coefficient_hdu = fits.table_to_hdu(
        make_wavelength_coefficient_table(fit)
    )
    coefficient_hdu.name = "WAVE_COEFF"

    if solution.covariance is None:
        covariance_data = np.empty((0, 0), dtype=float)
    else:
        covariance_data = np.asarray(solution.covariance, dtype=float)
    covariance_hdu = fits.ImageHDU(
        covariance_data,
        name="WAVE_COVAR",
    )

    line_hdu = fits.table_to_hdu(peak_table)
    line_hdu.name = "FIT_LINES"

    qa_hdu = fits.table_to_hdu(
        make_order_wavelength_qa_table(peak_table)
    )
    qa_hdu.name = "ORDER_QA"

    fits.HDUList(
        [
            primary,
            coefficient_hdu,
            covariance_hdu,
            line_hdu,
            qa_hdu,
        ]
    ).writeto(filename, overwrite=overwrite)


def read_wavelength_solution_fits(
    filename: str | Path,
) -> tuple[WavelengthSolution, fits.Header]:
    """Reconstruct a WavelengthSolution from a saved wavelength FITS file."""

    with fits.open(filename) as hdul:
        header = hdul[0].header.copy()
        coefficient_table = Table(hdul["WAVE_COEFF"].data)
        covariance = np.asarray(
            hdul["WAVE_COVAR"].data,
            dtype=float,
        )

    y_degree = int(np.max(coefficient_table["y_degree"]))
    order_degree = int(np.max(coefficient_table["order_degree"]))

    coefficients = np.zeros(
        (y_degree + 1, order_degree + 1),
        dtype=float,
    )

    for row in coefficient_table:
        coefficients[
            int(row["y_degree"]),
            int(row["order_degree"]),
        ] = float(row["coefficient"])

    if covariance.size == 0:
        covariance = None

    solution = WavelengthSolution(
        coefficients=coefficients,
        y_center=float(header["YCENTER"]),
        y_scale=float(header["YSCALE"]),
        order_center=float(header["MCENTER"]),
        order_scale=float(header["MSCALE"]),
        covariance=covariance,
    )

    return solution, header


# -----------------------------------------------------------------------------
# Diagnostic figure
# -----------------------------------------------------------------------------


def plot_wavelength_fit_diagnostics(
    peak_table: Table,
    *,
    filename: str | Path | None = None,
):
    """Make a compact QA figure for one global wavelength solution."""

    import matplotlib.pyplot as plt

    identified = np.isfinite(
        np.asarray(peak_table["wavelength_nm"], dtype=float)
    )
    used = np.asarray(
        peak_table["used_for_wavelength_fit"],
        dtype=bool,
    )

    y = np.asarray(peak_table["y"], dtype=float)
    order = np.asarray(peak_table["order"], dtype=int)
    velocity = np.asarray(
        peak_table["velocity_residual_mps"],
        dtype=float,
    )
    pixel_residual = np.asarray(
        peak_table["pixel_residual"],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    if np.any(identified):
        scatter = ax.scatter(
            y[identified],
            order[identified],
            c=velocity[identified],
            s=8,
        )
        fig.colorbar(scatter, ax=ax, label="velocity residual [m/s]")
    ax.set_xlabel("dispersion pixel y")
    ax.set_ylabel("echelle order m")
    ax.set_title("Calibration-line residual map")

    ax = axes[0, 1]
    if np.any(used):
        ax.scatter(
            y[used],
            velocity[used],
            s=8,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("dispersion pixel y")
    ax.set_ylabel("velocity residual [m/s]")
    ax.set_title("Residual versus detector position")

    ax = axes[1, 0]
    if np.any(used):
        ax.scatter(
            order[used],
            pixel_residual[used],
            s=8,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("echelle order m")
    ax.set_ylabel("residual [pixel]")
    ax.set_title("Residual versus order")

    ax = axes[1, 1]
    finite_velocity = used & np.isfinite(velocity)
    if np.any(finite_velocity):
        ax.hist(velocity[finite_velocity], bins=40)
    ax.set_xlabel("velocity residual [m/s]")
    ax.set_ylabel("number of lines")
    ax.set_title("Residual distribution")

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=200, bbox_inches="tight")

    return fig


# -----------------------------------------------------------------------------
# Source dispatch and blocked surface-complexity validation
# -----------------------------------------------------------------------------


def measure_reference_lines(source, *args, **kwargs):
    """Call the source-specific line measurement without coupling it to the fit.

    Returns a :class:`~velocereduction.calibration.CalibrationLineSet` whose
    ``lines`` table is the only object required by the wavelength fitter.
    """
    source_key = str(source).lower()
    if source_key in {"fibth", "simth"}:
        from .thorium import measure_thorium_lines
        kwargs.setdefault("source", "FibTh" if source_key == "fibth" else "SimTh")
        return measure_thorium_lines(*args, **kwargs)
    if source_key == "simlc":
        from .simlc import measure_simlc_lines
        return measure_simlc_lines(*args, **kwargs)
    raise ValueError("source must be 'FibTh', 'SimTh', or 'SimLC'")


def _surface_validation_residuals(solution, y, order, wavelength):
    wavelength = np.asarray(wavelength, dtype=float)
    model = solution.wavelength(y, order)
    residual_wavelength = wavelength - model
    dispersion = solution.dispersion(y, order)
    residual_pixel = np.divide(
        residual_wavelength,
        dispersion,
        out=np.full_like(residual_wavelength, np.nan),
        where=np.abs(dispersion) > 0,
    )
    residual_velocity = SPEED_OF_LIGHT_MPS * residual_wavelength / wavelength
    return residual_pixel, residual_velocity


def make_surface_cv_folds(
    y,
    order,
    *,
    y_bounds,
    n_folds=5,
    y_blocks=10,
):
    """Assign deterministic spatially blocked CV folds in the (y,m) plane.

    Lines are not shuffled individually.  Each physical order is divided into
    contiguous dispersion blocks and neighbouring order/block cells are sent to
    different folds.  Validation therefore tests interpolation over genuinely
    unseen detector regions rather than near-duplicate neighbouring lines.
    """
    y = np.asarray(y, dtype=float)
    order = np.asarray(order, dtype=int)
    if y.shape != order.shape:
        raise ValueError("y and order must have identical shapes")
    lo, hi = map(float, y_bounds)
    if not hi > lo:
        raise ValueError("y_bounds must satisfy upper > lower")
    scaled = (y - lo) / (hi - lo)
    y_block = np.floor(np.clip(scaled, 0.0, np.nextafter(1.0, 0.0)) * y_blocks).astype(int)
    unique_orders = np.sort(np.unique(order))
    rank = {int(m): i for i, m in enumerate(unique_orders)}
    order_rank = np.asarray([rank[int(m)] for m in order], dtype=int)
    return (y_block + order_rank) % int(n_folds)


def cross_validate_wavelength_surface(
    peak_table,
    *,
    y_bounds=None,
    order_bounds=None,
    y_degrees=range(3, 13),
    order_degrees=range(1, 9),
    n_folds=5,
    y_blocks=10,
    max_iterations=12,
):
    """Blocked cross-validation for the 2-D Legendre wavelength surface.

    One row is returned per viable ``(y_degree, order_degree)`` pair.  In
    addition to training/held-out RMS, the table records robust held-out
    residual statistics and the largest design-matrix condition number across
    folds.  These diagnostics make it possible to distinguish genuine
    predictive improvement from over-fitting or an ill-conditioned basis.
    """
    use = np.asarray(peak_table["used_for_wavelength_fit"], dtype=bool)
    use &= np.isfinite(np.asarray(peak_table["y"], dtype=float))
    use &= np.isfinite(np.asarray(peak_table["wavelength_nm"], dtype=float))
    indices = np.flatnonzero(use)
    if len(indices) == 0:
        raise RuntimeError("No accepted calibration lines for cross-validation")

    y = np.asarray(peak_table["y"][indices], dtype=float)
    order = np.asarray(peak_table["order"][indices], dtype=float)
    wavelength = np.asarray(peak_table["wavelength_nm"][indices], dtype=float)
    if y_bounds is None:
        y_bounds = WAVELENGTH_Y_BOUNDS
    if order_bounds is None:
        order_bounds = _full_order_bounds(order)
    y_uncertainty = np.asarray(peak_table["y_uncertainty"][indices], dtype=float)
    wavelength_uncertainty = np.asarray(
        peak_table["wavelength_uncertainty_nm"][indices], dtype=float
    )
    wavelength_uncertainty = np.where(
        np.isfinite(wavelength_uncertainty), wavelength_uncertainty, 0.0
    )
    folds = make_surface_cv_folds(
        y, order.astype(int), y_bounds=y_bounds, n_folds=n_folds, y_blocks=y_blocks
    )

    y_normalised, _, _ = _normalise_coordinate(y, y_bounds)
    order_normalised, _, _ = _normalise_coordinate(order, order_bounds)

    rows = []
    for y_degree in y_degrees:
        for order_degree in order_degrees:
            y_degree = int(y_degree)
            order_degree = int(order_degree)
            n_parameters = (y_degree + 1) * (order_degree + 1)
            design_matrix = _build_design_matrix(
                y_normalised, order_normalised, y_degree, order_degree
            )

            train_pixel, valid_pixel = [], []
            train_velocity, valid_velocity = [], []
            all_valid_pixel, all_valid_velocity = [], []
            condition_numbers, n_train, n_valid = [], [], []

            viable = True
            for fold in range(n_folds):
                validation = folds == fold
                training = ~validation
                if np.count_nonzero(training) <= n_parameters or not np.any(validation):
                    viable = False
                    break

                try:
                    fit = fit_wavelength_surface(
                        y[training],
                        order[training],
                        wavelength[training],
                        y_uncertainty=y_uncertainty[training],
                        wavelength_uncertainty=wavelength_uncertainty[training],
                        y_degree=y_degree,
                        order_degree=order_degree,
                        y_bounds=y_bounds,
                        order_bounds=order_bounds,
                        max_iterations=max_iterations,
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    viable = False
                    break

                tr_pix, tr_vel = _surface_validation_residuals(
                    fit.solution, y[training], order[training], wavelength[training]
                )
                va_pix, va_vel = _surface_validation_residuals(
                    fit.solution, y[validation], order[validation], wavelength[validation]
                )

                train_pixel.append(float(np.sqrt(np.nanmean(tr_pix**2))))
                valid_pixel.append(float(np.sqrt(np.nanmean(va_pix**2))))
                train_velocity.append(float(np.sqrt(np.nanmean(tr_vel**2))))
                valid_velocity.append(float(np.sqrt(np.nanmean(va_vel**2))))
                all_valid_pixel.append(np.asarray(va_pix, dtype=float))
                all_valid_velocity.append(np.asarray(va_vel, dtype=float))
                condition_numbers.append(float(np.linalg.cond(design_matrix[training])))
                n_train.append(int(np.count_nonzero(training)))
                n_valid.append(int(np.count_nonzero(validation)))

            if not viable or len(valid_pixel) != n_folds:
                continue

            valid_pixel = np.asarray(valid_pixel, dtype=float)
            valid_velocity = np.asarray(valid_velocity, dtype=float)
            heldout_pixel = np.concatenate(all_valid_pixel)
            heldout_velocity = np.concatenate(all_valid_velocity)
            finite_pixel = heldout_pixel[np.isfinite(heldout_pixel)]
            finite_velocity = heldout_velocity[np.isfinite(heldout_velocity)]
            abs_pixel = np.abs(finite_pixel)
            abs_velocity = np.abs(finite_velocity)

            # Fold-level statistics are intentionally retained in several robust
            # summaries.  A single spatially blocked fold can occasionally become
            # poorly constrained for an otherwise ordinary degree pair; the mean
            # RMS alone then looks catastrophic even when four of five folds are
            # well behaved.  Keep the mean/SE for the one-standard-error selector,
            # but expose the median, robust scatter, upper tail, and max/median
            # ratio so those unstable models are immediately recognisable in QA.
            valid_pixel_median = float(np.nanmedian(valid_pixel))
            valid_pixel_mad = float(
                1.4826 * np.nanmedian(np.abs(valid_pixel - valid_pixel_median))
            )
            valid_pixel_max = float(np.nanmax(valid_pixel))
            valid_pixel_ratio = (
                valid_pixel_max / valid_pixel_median
                if np.isfinite(valid_pixel_median) and valid_pixel_median > 0
                else np.inf
            )

            rows.append(dict(
                y_degree=y_degree,
                order_degree=order_degree,
                n_parameters=int(n_parameters),
                n_folds=int(n_folds),
                mean_n_train=float(np.mean(n_train)),
                mean_n_validation=float(np.mean(n_valid)),
                train_rms_pixel=float(np.mean(train_pixel)),
                validation_rms_pixel=float(np.mean(valid_pixel)),
                validation_rms_pixel_std=float(np.std(valid_pixel, ddof=1)),
                validation_rms_pixel_se=float(np.std(valid_pixel, ddof=1) / np.sqrt(n_folds)),
                validation_median_fold_rms_pixel=valid_pixel_median,
                validation_mad_fold_rms_pixel=valid_pixel_mad,
                validation_p90_fold_rms_pixel=float(np.nanpercentile(valid_pixel, 90)),
                validation_max_fold_rms_pixel=valid_pixel_max,
                validation_fold_rms_ratio=float(valid_pixel_ratio),
                validation_median_pixel=float(np.nanmedian(finite_pixel)),
                validation_median_abs_pixel=float(np.nanmedian(abs_pixel)),
                validation_p95_abs_pixel=float(np.nanpercentile(abs_pixel, 95)),
                train_rms_velocity_mps=float(np.mean(train_velocity)),
                validation_rms_velocity_mps=float(np.mean(valid_velocity)),
                validation_rms_velocity_mps_std=float(np.std(valid_velocity, ddof=1)),
                validation_median_fold_rms_velocity_mps=float(np.nanmedian(valid_velocity)),
                validation_max_fold_rms_velocity_mps=float(np.nanmax(valid_velocity)),
                validation_median_abs_velocity_mps=float(np.nanmedian(abs_velocity)),
                validation_p95_abs_velocity_mps=float(np.nanpercentile(abs_velocity, 95)),
                generalisation_gap_pixel=float(np.mean(valid_pixel) - np.mean(train_pixel)),
                max_condition_number=float(np.nanmax(condition_numbers)),
            ))

    return Table(rows=rows)

def select_wavelength_surface_degree(validation_table):
    """Choose the simplest surface within one standard error of the best CV RMS."""
    if len(validation_table) == 0:
        raise ValueError("validation_table is empty")
    rms = np.asarray(validation_table["validation_rms_pixel"], dtype=float)
    best = int(np.nanargmin(rms))
    threshold = float(rms[best] + validation_table["validation_rms_pixel_se"][best])
    acceptable = np.flatnonzero(rms <= threshold)
    complexity = np.asarray(validation_table["n_parameters"], dtype=int)
    minimum_complexity = np.min(complexity[acceptable])
    candidates = acceptable[complexity[acceptable] == minimum_complexity]
    chosen = int(candidates[np.argmin(rms[candidates])])
    return validation_table[chosen]


def fit_validated_wavelength_from_peak_table(
    peak_table,
    *,
    y_bounds=None,
    order_bounds=None,
    y_degrees=range(3, 13),
    order_degrees=range(1, 9),
    n_folds=5,
    y_blocks=10,
    max_iterations=12,
):
    """Select Legendre complexity by blocked CV, then refit all accepted lines."""
    validation = cross_validate_wavelength_surface(
        peak_table,
        y_bounds=y_bounds,
        order_bounds=order_bounds,
        y_degrees=y_degrees,
        order_degrees=order_degrees,
        n_folds=n_folds,
        y_blocks=y_blocks,
        max_iterations=max_iterations,
    )
    chosen = select_wavelength_surface_degree(validation)
    fit, fitted_lines = fit_wavelength_from_peak_table(
        peak_table,
        y_bounds=y_bounds,
        order_bounds=order_bounds,
        y_degree=int(chosen["y_degree"]),
        order_degree=int(chosen["order_degree"]),
        max_iterations=max_iterations,
    )
    return fit, fitted_lines, validation, chosen

# -----------------------------------------------------------------------------
# Differential pixel-shift models: source transfer, fibres, and time
# -----------------------------------------------------------------------------


@dataclass
class PixelShiftSurface:
    """Smooth detector-pixel displacement surface ``delta_y(y, m)``.

    The sign convention is always ``y_current - y_reference``.  A positive
    shift therefore means that a fixed wavelength is measured at a larger
    detector y coordinate than in the reference solution.
    """

    coefficients: np.ndarray
    y_center: float
    y_scale: float
    order_center: float
    order_scale: float
    covariance: np.ndarray | None = None

    def _normalised_coordinates(self, y, order):
        y, order = np.broadcast_arrays(
            np.asarray(y, dtype=float), np.asarray(order, dtype=float)
        )
        return (
            (y - self.y_center) / self.y_scale,
            (order - self.order_center) / self.order_scale,
        )

    def shift(self, y, order):
        y_n, m_n = self._normalised_coordinates(y, order)
        return legval2d(y_n, m_n, self.coefficients)

    __call__ = shift


@dataclass
class ShiftFitResult:
    """Pixel-shift surface plus line-by-line fit diagnostics."""

    surface: PixelShiftSurface
    used: np.ndarray
    robust_weight: np.ndarray
    residual_pixel: np.ndarray
    n_iterations: int


def fit_shift_surface(
    y,
    order,
    shift_y,
    *,
    shift_uncertainty=None,
    y_degree=2,
    order_degree=1,
    y_bounds=WAVELENGTH_Y_BOUNDS,
    order_bounds=None,
    max_iterations=10,
    huber_k=1.5,
    clip_sigma=6.0,
):
    """Fit a robust low-order surface to differential detector shifts."""

    y = np.asarray(y, dtype=float)
    order = np.asarray(order, dtype=float)
    shift_y = np.asarray(shift_y, dtype=float)
    if not (y.shape == order.shape == shift_y.shape) or y.ndim != 1:
        raise ValueError("y, order, and shift_y must be matching 1-D arrays")

    if shift_uncertainty is None:
        shift_uncertainty = np.zeros_like(y)
    else:
        shift_uncertainty = np.broadcast_to(
            np.asarray(shift_uncertainty, dtype=float), y.shape
        ).copy()

    finite = (
        np.isfinite(y)
        & np.isfinite(order)
        & np.isfinite(shift_y)
        & np.isfinite(shift_uncertainty)
        & (shift_uncertainty >= 0)
    )
    if order_bounds is None:
        order_bounds = (float(np.nanmin(order[finite])), float(np.nanmax(order[finite])))

    y_n, y_center, y_scale = _normalise_coordinate(y, y_bounds)
    m_n, order_center, order_scale = _normalise_coordinate(order, order_bounds)
    design = _build_design_matrix(y_n, m_n, y_degree, order_degree)
    n_parameters = design.shape[1]
    if np.count_nonzero(finite) <= n_parameters:
        raise ValueError(
            f"Not enough shift measurements: {np.count_nonzero(finite)} lines "
            f"for {n_parameters} coefficients"
        )

    positive_sigma = finite & (shift_uncertainty > 0)
    if np.any(positive_sigma):
        typical_sigma = float(np.nanmedian(shift_uncertainty[positive_sigma]))
        sigma = np.where(positive_sigma, shift_uncertainty, typical_sigma)
    else:
        sigma = np.ones_like(y)

    used = finite.copy()
    robust_weight = np.ones_like(y)
    coefficients = None

    for iteration in range(1, max_iterations + 1):
        old_used = used.copy()
        weights = robust_weight / sigma**2
        sqrt_w = np.sqrt(weights[used])
        coefficients_flat, *_ = np.linalg.lstsq(
            design[used] * sqrt_w[:, None], shift_y[used] * sqrt_w, rcond=None
        )
        coefficients = coefficients_flat.reshape(y_degree + 1, order_degree + 1)
        model = legval2d(y_n, m_n, coefficients)
        residual = shift_y - model

        robust_scatter = _safe_residual_scale(
            residual[used], reference_scale=max(1.0, np.nanmedian(np.abs(shift_y[finite])))
        )
        effective_sigma = np.sqrt(sigma**2 + robust_scatter**2)
        normalised = residual / effective_sigma
        robust_weight = np.ones_like(y)
        large = np.abs(normalised) > huber_k
        robust_weight[large] = huber_k / np.abs(normalised[large])
        used = finite & (np.abs(normalised) <= clip_sigma)

        if np.count_nonzero(used) <= n_parameters:
            raise RuntimeError("Shift-surface clipping left too few calibration lines")
        if np.array_equal(used, old_used):
            break

    weights = robust_weight / sigma**2
    sqrt_w = np.sqrt(weights[used])
    coefficients_flat, *_ = np.linalg.lstsq(
        design[used] * sqrt_w[:, None], shift_y[used] * sqrt_w, rcond=None
    )
    coefficients = coefficients_flat.reshape(y_degree + 1, order_degree + 1)

    weighted_design = design[used] * sqrt_w[:, None]
    covariance = np.linalg.pinv(weighted_design.T @ weighted_design)
    model = legval2d(y_n, m_n, coefficients)
    residual = shift_y - model
    dof = max(1, np.count_nonzero(used) - n_parameters)
    scale2 = float(np.sum((residual[used] * sqrt_w) ** 2) / dof)
    covariance *= max(scale2, 1.0)

    return ShiftFitResult(
        surface=PixelShiftSurface(
            coefficients=coefficients,
            y_center=y_center,
            y_scale=y_scale,
            order_center=order_center,
            order_scale=order_scale,
            covariance=covariance,
        ),
        used=used,
        robust_weight=robust_weight,
        residual_pixel=residual,
        n_iterations=iteration,
    )


def reference_y_from_wavelength(
    solution: WavelengthSolution,
    wavelength_nm,
    order,
    *,
    y_bounds=(0.0, 4111.0),
    grid_size=16385,
):
    """Invert a monotonic reference wavelength solution order by order."""

    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    order = np.asarray(order, dtype=int)
    wavelength_nm, order = np.broadcast_arrays(wavelength_nm, order)
    output = np.full(wavelength_nm.shape, np.nan, dtype=float)
    y_grid = np.linspace(float(y_bounds[0]), float(y_bounds[1]), int(grid_size))

    for m in np.unique(order):
        q = order == int(m)
        wave_grid = np.asarray(solution.wavelength(y_grid, int(m)), dtype=float)
        finite = np.isfinite(wave_grid)
        if np.count_nonzero(finite) < 2:
            continue
        wave = wave_grid[finite]
        yy = y_grid[finite]
        idx = np.argsort(wave)
        wave, yy = wave[idx], yy[idx]
        wave, unique = np.unique(wave, return_index=True)
        yy = yy[unique]
        target = wavelength_nm[q]
        inside = np.isfinite(target) & (target >= wave[0]) & (target <= wave[-1])
        values = np.full(target.shape, np.nan)
        values[inside] = np.interp(target[inside], wave, yy)
        output[q] = values

    return output


def fit_shift_from_peak_table(
    peak_table: Table,
    reference_solution: WavelengthSolution,
    *,
    y_bounds=None,
    order_bounds=None,
    y_degree=2,
    order_degree=1,
    max_iterations=10,
):
    """Fit ``y_measured - y_reference(lambda,m)`` for an identified line table."""

    table = peak_table.copy(copy_data=True)
    use = np.asarray(table["used_for_wavelength_fit"], dtype=bool)
    use &= np.isfinite(np.asarray(table["y"], dtype=float))
    use &= np.isfinite(np.asarray(table["wavelength_nm"], dtype=float))
    indices = np.flatnonzero(use)
    if len(indices) == 0:
        raise RuntimeError("No accepted identified lines are available for shift fitting")

    y_measured = np.asarray(table["y"][indices], dtype=float)
    order = np.asarray(table["order"][indices], dtype=int)
    if y_bounds is None:
        y_bounds = WAVELENGTH_Y_BOUNDS
    if order_bounds is None:
        order_bounds = _full_order_bounds(order)
    wave = np.asarray(table["wavelength_nm"][indices], dtype=float)
    y_reference = reference_y_from_wavelength(
        reference_solution, wave, order, y_bounds=y_bounds
    )
    finite = np.isfinite(y_reference)
    if not np.any(finite):
        raise RuntimeError("Reference solution could not be inverted for the supplied lines")

    uncertainty = np.asarray(table["y_uncertainty"][indices], dtype=float)
    uncertainty = np.where(np.isfinite(uncertainty), uncertainty, 0.0)
    result = fit_shift_surface(
        y_reference[finite],
        order[finite],
        y_measured[finite] - y_reference[finite],
        shift_uncertainty=uncertainty[finite],
        y_degree=y_degree,
        order_degree=order_degree,
        y_bounds=y_bounds,
        order_bounds=order_bounds,
        max_iterations=max_iterations,
    )

    for name in ("reference_y", "shift_y", "shift_residual_y"):
        table[name] = np.full(len(table), np.nan, dtype=float)
    selected = indices[finite]
    table["reference_y"][selected] = y_reference[finite]
    table["shift_y"][selected] = y_measured[finite] - y_reference[finite]
    table["shift_residual_y"][selected] = result.residual_pixel
    for local, table_i in enumerate(selected):
        if not result.used[local]:
            table["used_for_wavelength_fit"][table_i] = False

    return result, table


def _line_table(value):
    return value.lines if hasattr(value, "lines") else value


def build_hybrid_static_peak_table(
    fibth_lines,
    simlc_lines,
    preliminary_solution: WavelengthSolution,
    *,
    y_bounds=None,
    order_bounds=None,
    transfer_y_degree=2,
    transfer_order_degree=1,
):
    """Transfer one SimLC exposure onto the summed-FibTh coordinate system.

    FibTh remains the absolute anchor.  A low-order SimLC source-offset surface
    is measured against the preliminary FibTh solution and removed from the
    eLSF-refined comb centroids before the two line tables are stacked.  The
    returned table can then be used for a denser final static fit.
    """

    from astropy.table import vstack

    fibth = _line_table(fibth_lines).copy(copy_data=True)
    simlc = _line_table(simlc_lines).copy(copy_data=True)
    transfer, simlc_shift_table = fit_shift_from_peak_table(
        simlc,
        preliminary_solution,
        y_bounds=y_bounds,
        order_bounds=order_bounds,
        y_degree=transfer_y_degree,
        order_degree=transfer_order_degree,
    )

    y_raw = np.asarray(simlc_shift_table["y"], dtype=float).copy()
    reference_y = np.asarray(simlc_shift_table["reference_y"], dtype=float)
    predicted_shift = transfer.surface.shift(
        reference_y, np.asarray(simlc_shift_table["order"], dtype=int)
    )
    good = np.isfinite(reference_y) & np.isfinite(predicted_shift)
    simlc_shift_table["y_source_raw"] = y_raw
    simlc_shift_table["coordinate_transfer_shift_y"] = np.full(len(simlc), np.nan)
    simlc_shift_table["coordinate_transfer_shift_y"][good] = predicted_shift[good]
    simlc_shift_table["y"][good] = y_raw[good] - predicted_shift[good]

    # Astropy vstack needs identical broad schemas.  Keep only common columns;
    # all source-specific diagnostic products are already persisted separately.
    common = [name for name in fibth.colnames if name in simlc_shift_table.colnames]
    hybrid = vstack([fibth[common], simlc_shift_table[common]], metadata_conflicts="silent")
    return hybrid, transfer, simlc_shift_table


@dataclass
class FibreDisplacementModel:
    """Joint fibre-to-summed-FibTh displacement model for all three CCDs.

    The sign convention is ``delta_y = y_fibre - y_reference``, where
    ``y_reference`` is the summed-FibTh detector coordinate.  Fibre-mode
    amplitudes are shared across CCDs while each mode has a separate smooth
    Legendre surface on each CCD.
    """

    fibres: np.ndarray
    fibre_modes: np.ndarray
    offsets: dict[str, np.ndarray]
    surface_coefficients: dict[str, np.ndarray]
    y_center: dict[str, float]
    y_scale: dict[str, float]
    order_center: dict[str, float]
    order_scale: dict[str, float]
    degree: int = 4
    reference_night: str = ""
    fit_rms_pixel: dict[str, float] | None = None
    fit_n_lines: dict[str, int] | None = None

    def __post_init__(self):
        self.fibres = np.asarray(self.fibres, dtype=int)
        self.fibre_modes = np.asarray(self.fibre_modes, dtype=float)
        if self.fibre_modes.ndim != 2 or self.fibre_modes.shape[0] != len(self.fibres):
            raise ValueError("fibre_modes must have shape (n_fibres, rank)")
        for ccd in self.surface_coefficients:
            ccd = str(ccd)
            coeff = np.asarray(self.surface_coefficients[ccd], dtype=float)
            expected = (self.rank, self.degree + 1, self.degree + 1)
            if coeff.shape != expected:
                raise ValueError(f"CCD{ccd} coefficient shape {coeff.shape}; expected {expected}")
            self.surface_coefficients[ccd] = coeff
            self.offsets[ccd] = np.asarray(self.offsets[ccd], dtype=float)

    @property
    def rank(self):
        return int(self.fibre_modes.shape[1])

    def _fibre_index(self, fibre):
        match = np.flatnonzero(self.fibres == int(fibre))
        if len(match) != 1:
            raise KeyError(f"Fibre {fibre} is not present in the displacement model")
        return int(match[0])

    def _normalised_coordinates(self, ccd, y_reference, order):
        ccd = str(ccd)
        y_reference, order = np.broadcast_arrays(
            np.asarray(y_reference, dtype=float), np.asarray(order, dtype=float)
        )
        return (
            (y_reference - self.y_center[ccd]) / self.y_scale[ccd],
            (order - self.order_center[ccd]) / self.order_scale[ccd],
        )

    def displacement(self, ccd, fibre, y_reference, order):
        """Evaluate ``y_fibre - y_reference`` in detector pixels."""
        ccd = str(ccd)
        fibre_index = self._fibre_index(fibre)
        y_normalised, order_normalised = self._normalised_coordinates(ccd, y_reference, order)
        result = np.full(
            np.broadcast(y_normalised, order_normalised).shape,
            self.offsets[ccd][fibre_index],
            dtype=float,
        )
        for mode in range(self.rank):
            result += self.fibre_modes[fibre_index, mode] * legval2d(
                y_normalised,
                order_normalised,
                self.surface_coefficients[ccd][mode],
            )
        return result

    def fibre_to_reference_y(
        self,
        ccd,
        fibre,
        y_fibre,
        order,
        *,
        fixed_point_iterations=3,
    ):
        """Map native fibre pixels onto the summed-FibTh reference coordinate."""
        y_fibre, order = np.broadcast_arrays(
            np.asarray(y_fibre, dtype=float), np.asarray(order, dtype=float)
        )
        y_reference = y_fibre.copy()
        for _ in range(max(1, int(fixed_point_iterations))):
            y_reference = y_fibre - self.displacement(ccd, fibre, y_reference, order)
        return y_reference


def _fibre_legendre_terms(degree):
    return [
        (iy, total_degree - iy)
        for total_degree in range(1, int(degree) + 1)
        for iy in range(total_degree, -1, -1)
    ]


def _fibre_legendre_basis(y_reference, order, ccd, degree):
    """Total-degree Legendre basis excluding the constant term."""
    ccd = str(ccd)
    y_reference = np.asarray(y_reference, dtype=float)
    order = np.asarray(order, dtype=float)
    y_center = 0.5 * (N_DISPERSION - 1)
    y_scale = 0.5 * (N_DISPERSION - 1)
    order_values = np.asarray(CCD_ORDERS[ccd], dtype=float)
    order_center = 0.5 * (np.min(order_values) + np.max(order_values))
    order_scale = 0.5 * (np.max(order_values) - np.min(order_values))
    yn = (y_reference - y_center) / y_scale
    mn = (order - order_center) / order_scale
    y_basis = legvander(yn, degree)
    order_basis = legvander(mn, degree)
    terms = _fibre_legendre_terms(degree)
    design = np.column_stack([
        y_basis[:, iy] * order_basis[:, im] for iy, im in terms
    ])
    return design, terms, y_center, y_scale, order_center, order_scale


def match_fibre_lines_to_summed(
    summed_lines,
    fibre_line_sets,
    *,
    ccd,
    maximum_y_difference=1.0,
    minimum_snr=5.0,
    maximum_snr=100.0,
):
    """Match fibre FibTh peaks directly to measured summed-FibTh peaks."""
    summed = _line_table(summed_lines)
    rows = []
    summed_order = np.asarray(summed["order"], dtype=int)
    for fibre, value in sorted(fibre_line_sets.items()):
        fibre_lines = _line_table(value)
        fibre_order = np.asarray(fibre_lines["order"], dtype=int)
        fibre_snr = np.asarray(fibre_lines["signal_to_noise"], dtype=float)
        good_fibre = (
            np.isfinite(np.asarray(fibre_lines["y"], dtype=float))
            & np.isfinite(fibre_snr)
            & (fibre_snr >= float(minimum_snr))
            & (fibre_snr <= float(maximum_snr))
        )
        if "fit_success" in fibre_lines.colnames:
            good_fibre &= np.asarray(fibre_lines["fit_success"], dtype=bool)

        for order_value in np.intersect1d(np.unique(summed_order), np.unique(fibre_order)):
            summed_index = np.flatnonzero(summed_order == int(order_value))
            fibre_index = np.flatnonzero((fibre_order == int(order_value)) & good_fibre)
            if not len(summed_index) or not len(fibre_index):
                continue
            y_reference = np.asarray(summed["y"][summed_index], dtype=float)
            y_fibre = np.asarray(fibre_lines["y"][fibre_index], dtype=float)
            distance = np.abs(y_reference[:, None] - y_fibre[None, :])
            candidates = np.argwhere(distance < float(maximum_y_difference))
            if len(candidates) == 0:
                continue
            candidates = candidates[np.argsort(distance[candidates[:, 0], candidates[:, 1]])]
            used_reference, used_fibre = set(), set()
            for reference_local, fibre_local in candidates:
                reference_local, fibre_local = int(reference_local), int(fibre_local)
                if reference_local in used_reference or fibre_local in used_fibre:
                    continue
                used_reference.add(reference_local)
                used_fibre.add(fibre_local)
                si = int(summed_index[reference_local])
                fi = int(fibre_index[fibre_local])
                yr = float(summed["y"][si])
                yf = float(fibre_lines["y"][fi])
                rows.append(dict(
                    ccd=str(ccd),
                    fibre=int(fibre),
                    order=int(order_value),
                    summed_peak_id=int(summed["peak_id"][si]),
                    fibre_peak_id=int(fibre_lines["peak_id"][fi]),
                    y_reference=yr,
                    y_fibre=yf,
                    delta_y=yf - yr,
                    signal_to_noise=float(fibre_lines["signal_to_noise"][fi]),
                    summed_signal_to_noise=float(summed["signal_to_noise"][si]),
                    y_uncertainty_fibre=float(fibre_lines["y_uncertainty"][fi]),
                    y_uncertainty_summed=float(summed["y_uncertainty"][si]),
                ))
    return Table(rows=rows)


def _initial_fibre_modes(matched_lines, fibres, rank, degree, used):
    terms = _fibre_legendre_terms(degree)
    n_terms = len(terms)
    features = np.zeros((len(fibres), 3 * n_terms), dtype=float)
    ccd_values = np.asarray(matched_lines["ccd"], dtype=str)
    fibre_values = np.asarray(matched_lines["fibre"], dtype=int)
    delta_y = np.asarray(matched_lines["delta_y"], dtype=float)
    for ccd_index, ccd in enumerate(("1", "2", "3")):
        for fibre_index, fibre in enumerate(fibres):
            select = used & (ccd_values == ccd) & (fibre_values == int(fibre))
            if np.count_nonzero(select) <= n_terms + 1:
                continue
            basis, _, *_ = _fibre_legendre_basis(
                np.asarray(matched_lines["y_reference"], float)[select],
                np.asarray(matched_lines["order"], float)[select],
                ccd,
                degree,
            )
            design = np.column_stack((np.ones(np.count_nonzero(select)), basis))
            coefficients, *_ = np.linalg.lstsq(design, delta_y[select], rcond=None)
            lo, hi = ccd_index * n_terms, (ccd_index + 1) * n_terms
            features[fibre_index, lo:hi] = coefficients[1:]
    features -= np.mean(features, axis=0, keepdims=True)
    u, _, _ = np.linalg.svd(features, full_matrices=False)
    if u.shape[1] < rank:
        raise RuntimeError("Could not initialise the requested fibre-displacement rank")
    return u[:, :rank]


def _fit_fibre_displacement_once(
    matched_lines,
    *,
    fibres,
    rank,
    degree,
    used,
    initial_fibre_modes=None,
    max_iterations=30,
):
    fibres = np.asarray(fibres, dtype=int)
    fibre_modes = (
        _initial_fibre_modes(matched_lines, fibres, rank, degree, used)
        if initial_fibre_modes is None
        else np.asarray(initial_fibre_modes, dtype=float).copy()
    )
    fibre_modes, _ = np.linalg.qr(fibre_modes)
    fibre_modes = fibre_modes[:, :rank]
    fibre_lookup = {int(f): i for i, f in enumerate(fibres)}

    ccd_values = np.asarray(matched_lines["ccd"], dtype=str)
    fibre_values = np.asarray(matched_lines["fibre"], dtype=int)
    y_reference = np.asarray(matched_lines["y_reference"], dtype=float)
    order = np.asarray(matched_lines["order"], dtype=float)
    delta_y = np.asarray(matched_lines["delta_y"], dtype=float)
    terms = _fibre_legendre_terms(degree)
    n_terms = len(terms)
    surfaces = {ccd: np.zeros((rank, n_terms), dtype=float) for ccd in ("1", "2", "3")}
    offsets = {ccd: np.zeros(len(fibres), dtype=float) for ccd in ("1", "2", "3")}

    previous_rms = np.inf
    for _ in range(int(max_iterations)):
        for ccd in ("1", "2", "3"):
            rows = np.flatnonzero(used & (ccd_values == ccd))
            if len(rows) == 0:
                continue
            basis, _, *_ = _fibre_legendre_basis(y_reference[rows], order[rows], ccd, degree)
            fibre_index = np.array([fibre_lookup[int(f)] for f in fibre_values[rows]])
            offset_design = np.zeros((len(rows), len(fibres)), dtype=float)
            offset_design[np.arange(len(rows)), fibre_index] = 1.0
            mode_design = np.hstack([
                fibre_modes[fibre_index, mode, None] * basis for mode in range(rank)
            ])
            coefficients, *_ = np.linalg.lstsq(
                np.hstack((offset_design, mode_design)), delta_y[rows], rcond=None
            )
            offsets[ccd] = coefficients[:len(fibres)]
            surfaces[ccd] = coefficients[len(fibres):].reshape(rank, n_terms)

        new_modes = np.zeros_like(fibre_modes)
        new_offsets = {ccd: offsets[ccd].copy() for ccd in offsets}
        for fibre_index, fibre in enumerate(fibres):
            rows = np.flatnonzero(used & (fibre_values == int(fibre)))
            if len(rows) == 0:
                continue
            intercept = np.zeros((len(rows), 3), dtype=float)
            mode_value = np.zeros((len(rows), rank), dtype=float)
            for ccd_index, ccd in enumerate(("1", "2", "3")):
                local = np.flatnonzero(ccd_values[rows] == ccd)
                if not len(local):
                    continue
                intercept[local, ccd_index] = 1.0
                basis, _, *_ = _fibre_legendre_basis(
                    y_reference[rows[local]], order[rows[local]], ccd, degree
                )
                mode_value[local] = basis @ surfaces[ccd].T
            coefficients, *_ = np.linalg.lstsq(
                np.hstack((intercept, mode_value)), delta_y[rows], rcond=None
            )
            for ccd_index, ccd in enumerate(("1", "2", "3")):
                new_offsets[ccd][fibre_index] = coefficients[ccd_index]
            new_modes[fibre_index] = coefficients[3:]

        q, r = np.linalg.qr(new_modes)
        q, r = q[:, :rank], r[:rank, :rank]
        for mode in range(rank):
            pivot = int(np.argmax(np.abs(q[:, mode])))
            if q[pivot, mode] < 0:
                q[:, mode] *= -1.0
                r[mode, :] *= -1.0
        fibre_modes = q
        offsets = new_offsets
        for ccd in surfaces:
            surfaces[ccd] = r @ surfaces[ccd]

        prediction = np.full(len(matched_lines), np.nan, dtype=float)
        for ccd in ("1", "2", "3"):
            rows = np.flatnonzero(used & (ccd_values == ccd))
            if not len(rows):
                continue
            basis, _, *_ = _fibre_legendre_basis(y_reference[rows], order[rows], ccd, degree)
            fibre_index = np.array([fibre_lookup[int(f)] for f in fibre_values[rows]])
            prediction[rows] = offsets[ccd][fibre_index] + np.sum(
                fibre_modes[fibre_index] * (basis @ surfaces[ccd].T), axis=1
            )
        rms = float(np.sqrt(np.nanmean((delta_y[used] - prediction[used]) ** 2)))
        if abs(previous_rms - rms) < 1e-10:
            break
        previous_rms = rms

    coefficient_matrices = {}
    for ccd in ("1", "2", "3"):
        matrix = np.zeros((rank, degree + 1, degree + 1), dtype=float)
        for term_index, (iy, im) in enumerate(terms):
            matrix[:, iy, im] = surfaces[ccd][:, term_index]
        coefficient_matrices[ccd] = matrix
    return fibre_modes, offsets, coefficient_matrices


def _canonicalise_fibre_displacement_modes(
    fibre_modes,
    surface_coefficients,
    matched_lines,
    used,
):
    """Fix the rank-space rotation/order/sign for reproducible saved modes."""
    fibre_modes = np.asarray(fibre_modes, dtype=float).copy()
    rank = fibre_modes.shape[1]
    mode_values = np.zeros((np.count_nonzero(used), rank), dtype=float)
    used_rows = np.flatnonzero(used)
    row_lookup = {int(row): i for i, row in enumerate(used_rows)}
    ccd_values = np.asarray(matched_lines["ccd"], str)
    y_reference = np.asarray(matched_lines["y_reference"], float)
    order = np.asarray(matched_lines["order"], float)

    for ccd in ("1", "2", "3"):
        rows = np.flatnonzero(used & (ccd_values == ccd))
        if not len(rows):
            continue
        order_values = np.asarray(CCD_ORDERS[ccd], float)
        yn = (y_reference[rows] - 0.5 * (N_DISPERSION - 1)) / (0.5 * (N_DISPERSION - 1))
        mn = (order[rows] - 0.5 * (order_values.min() + order_values.max())) / (0.5 * (order_values.max() - order_values.min()))
        for mode in range(rank):
            values = legval2d(yn, mn, surface_coefficients[ccd][mode])
            mode_values[[row_lookup[int(row)] for row in rows], mode] = values

    gram = mode_values.T @ mode_values
    eigenvalue, rotation = np.linalg.eigh(gram)
    rotation = rotation[:, np.argsort(eigenvalue)[::-1]]
    fibre_modes = fibre_modes @ rotation
    surface_coefficients = {
        ccd: np.einsum("ab,bij->aij", rotation.T, np.asarray(coefficients, float))
        for ccd, coefficients in surface_coefficients.items()
    }
    for mode in range(rank):
        pivot = int(np.argmax(np.abs(fibre_modes[:, mode])))
        if fibre_modes[pivot, mode] < 0:
            fibre_modes[:, mode] *= -1.0
            for ccd in surface_coefficients:
                surface_coefficients[ccd][mode] *= -1.0
    return fibre_modes, surface_coefficients


def fit_fibre_displacement_model(
    summed_line_sets,
    fibre_line_sets,
    *,
    rank=2,
    degree=4,
    maximum_y_difference=1.0,
    minimum_snr=5.0,
    maximum_snr=100.0,
    clip_sigma=6.0,
    max_clip_iterations=4,
    reference_night="",
):
    """Fit the joint low-rank fibre displacement model across CCD1--3."""
    from astropy.table import vstack

    tables = [
        match_fibre_lines_to_summed(
            summed_line_sets[ccd], fibre_line_sets[ccd], ccd=ccd,
            maximum_y_difference=maximum_y_difference,
            minimum_snr=minimum_snr,
            maximum_snr=maximum_snr,
        )
        for ccd in ("1", "2", "3")
    ]
    matched_lines = vstack(tables, metadata_conflicts="silent")
    if len(matched_lines) == 0:
        raise RuntimeError("No fibre FibTh peaks matched the summed-FibTh line sets")

    fibres = np.asarray(SCIENCE_FIBRES, dtype=int)
    used = np.ones(len(matched_lines), dtype=bool)
    fibre_modes = None
    for _ in range(int(max_clip_iterations)):
        fibre_modes, offsets, coefficients = _fit_fibre_displacement_once(
            matched_lines,
            fibres=fibres,
            rank=int(rank),
            degree=int(degree),
            used=used,
            initial_fibre_modes=fibre_modes,
        )
        y_center, y_scale, order_center, order_scale = {}, {}, {}, {}
        for ccd in ("1", "2", "3"):
            _, _, yc, ys, mc, ms = _fibre_legendre_basis(
                np.array([0.0]), np.array([np.mean(CCD_ORDERS[ccd])]), ccd, degree
            )
            y_center[ccd], y_scale[ccd] = yc, ys
            order_center[ccd], order_scale[ccd] = mc, ms
        fibre_displacement_model = FibreDisplacementModel(
            fibres=fibres,
            fibre_modes=fibre_modes,
            offsets=offsets,
            surface_coefficients=coefficients,
            y_center=y_center,
            y_scale=y_scale,
            order_center=order_center,
            order_scale=order_scale,
            degree=int(degree),
            reference_night=str(reference_night),
        )
        prediction = np.array([
            fibre_displacement_model.displacement(ccd, fibre, y_ref, order_value)
            for ccd, fibre, y_ref, order_value in zip(
                matched_lines["ccd"], matched_lines["fibre"],
                matched_lines["y_reference"], matched_lines["order"],
            )
        ], dtype=float)
        residual = np.asarray(matched_lines["delta_y"], float) - prediction
        new_used = used.copy()
        ccd_values = np.asarray(matched_lines["ccd"], str)
        for ccd in ("1", "2", "3"):
            select = used & (ccd_values == ccd)
            scale = _mad_std(residual[select])
            if np.isfinite(scale) and scale > 0:
                new_used &= ~((ccd_values == ccd) & (np.abs(residual) > float(clip_sigma) * scale))
        if np.array_equal(new_used, used):
            break
        used = new_used

    # Refit once on the final clipped mask so the persisted coefficients and
    # diagnostics correspond to exactly the same accepted lines.
    fibre_modes, offsets, coefficients = _fit_fibre_displacement_once(
        matched_lines, fibres=fibres, rank=int(rank), degree=int(degree),
        used=used, initial_fibre_modes=fibre_modes,
    )
    fibre_modes, coefficients = _canonicalise_fibre_displacement_modes(
        fibre_modes, coefficients, matched_lines, used
    )
    fibre_displacement_model = FibreDisplacementModel(
        fibres=fibres, fibre_modes=fibre_modes, offsets=offsets,
        surface_coefficients=coefficients, y_center=y_center, y_scale=y_scale,
        order_center=order_center, order_scale=order_scale, degree=int(degree),
        reference_night=str(reference_night),
    )
    prediction = np.array([
        fibre_displacement_model.displacement(ccd, fibre, y_ref, order_value)
        for ccd, fibre, y_ref, order_value in zip(
            matched_lines["ccd"], matched_lines["fibre"],
            matched_lines["y_reference"], matched_lines["order"],
        )
    ], dtype=float)
    residual = np.asarray(matched_lines["delta_y"], float) - prediction

    matched_lines["model_delta_y"] = prediction
    matched_lines["residual_y"] = residual
    matched_lines["used_for_fibre_displacement_fit"] = used
    fit_rms, fit_n_lines = {}, {}
    ccd_values = np.asarray(matched_lines["ccd"], str)
    for ccd in ("1", "2", "3"):
        select = used & (ccd_values == ccd)
        fit_rms[ccd] = float(np.sqrt(np.nanmean(residual[select] ** 2)))
        fit_n_lines[ccd] = int(np.count_nonzero(select))
    fibre_displacement_model.fit_rms_pixel = fit_rms
    fibre_displacement_model.fit_n_lines = fit_n_lines
    return fibre_displacement_model, matched_lines


def write_fibre_displacement_model(fibre_displacement_model, filename, *, overwrite=False):
    """Persist a compact joint fibre displacement reference model."""
    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "velocereduction"
    primary.header["CONTENT"] = "Joint fibre displacement model"
    primary.header["REFNIGHT"] = str(fibre_displacement_model.reference_night)
    primary.header["RANK"] = int(fibre_displacement_model.rank)
    primary.header["DEGREE"] = int(fibre_displacement_model.degree)
    primary.header["DYDEF"] = "Y_FIBRE-Y_REFERENCE"
    primary.header["YFRAME"] = "SUMMED_FIBTH"

    fibre_rows = []
    for fibre_index, fibre in enumerate(fibre_displacement_model.fibres):
        row = dict(fibre=int(fibre))
        for mode in range(fibre_displacement_model.rank):
            row[f"mode_{mode + 1}"] = float(fibre_displacement_model.fibre_modes[fibre_index, mode])
        fibre_rows.append(row)

    offset_rows, surface_rows, scaling_rows = [], [], []
    for ccd in ("1", "2", "3"):
        for fibre_index, fibre in enumerate(fibre_displacement_model.fibres):
            offset_rows.append(dict(
                ccd=ccd,
                fibre=int(fibre),
                offset=float(fibre_displacement_model.offsets[ccd][fibre_index]),
            ))
        coefficients = fibre_displacement_model.surface_coefficients[ccd]
        for mode in range(fibre_displacement_model.rank):
            for iy in range(coefficients.shape[1]):
                for im in range(coefficients.shape[2]):
                    if iy == 0 and im == 0:
                        continue
                    if iy + im > fibre_displacement_model.degree:
                        continue
                    surface_rows.append(dict(
                        ccd=ccd,
                        mode=mode + 1,
                        y_degree=iy,
                        order_degree=im,
                        coefficient=float(coefficients[mode, iy, im]),
                    ))
        scaling_rows.append(dict(
            ccd=ccd,
            y_center=float(fibre_displacement_model.y_center[ccd]),
            y_scale=float(fibre_displacement_model.y_scale[ccd]),
            order_center=float(fibre_displacement_model.order_center[ccd]),
            order_scale=float(fibre_displacement_model.order_scale[ccd]),
            rms_pixel=float((fibre_displacement_model.fit_rms_pixel or {}).get(ccd, np.nan)),
            n_lines=int((fibre_displacement_model.fit_n_lines or {}).get(ccd, 0)),
        ))

    fits.HDUList([
        primary,
        fits.BinTableHDU(Table(rows=fibre_rows), name="FIBRE_MODES"),
        fits.BinTableHDU(Table(rows=offset_rows), name="OFFSETS"),
        fits.BinTableHDU(Table(rows=surface_rows), name="SURFACES"),
        fits.BinTableHDU(Table(rows=scaling_rows), name="SCALING"),
    ]).writeto(filename, overwrite=overwrite)


def read_fibre_displacement_model(filename):
    """Read a joint fibre displacement reference model."""
    with fits.open(filename) as hdul:
        header = hdul[0].header.copy()
        fibre_table = Table(hdul["FIBRE_MODES"].data)
        offset_table = Table(hdul["OFFSETS"].data)
        surface_table = Table(hdul["SURFACES"].data)
        scaling_table = Table(hdul["SCALING"].data)

    rank = int(header["RANK"])
    degree = int(header["DEGREE"])
    fibres = np.asarray(fibre_table["fibre"], dtype=int)
    fibre_modes = np.column_stack([
        np.asarray(fibre_table[f"mode_{mode + 1}"], dtype=float)
        for mode in range(rank)
    ])
    offsets, coefficients = {}, {}
    y_center, y_scale, order_center, order_scale = {}, {}, {}, {}
    fit_rms, fit_n_lines = {}, {}
    for ccd in ("1", "2", "3"):
        rows = offset_table[np.asarray(offset_table["ccd"], str) == ccd]
        by_fibre = {int(row["fibre"]): float(row["offset"]) for row in rows}
        offsets[ccd] = np.array([by_fibre[int(f)] for f in fibres], dtype=float)
        matrix = np.zeros((rank, degree + 1, degree + 1), dtype=float)
        rows = surface_table[np.asarray(surface_table["ccd"], str) == ccd]
        for row in rows:
            matrix[int(row["mode"]) - 1, int(row["y_degree"]), int(row["order_degree"])] = float(row["coefficient"])
        coefficients[ccd] = matrix
        row = scaling_table[np.asarray(scaling_table["ccd"], str) == ccd][0]
        y_center[ccd], y_scale[ccd] = float(row["y_center"]), float(row["y_scale"])
        order_center[ccd], order_scale[ccd] = float(row["order_center"]), float(row["order_scale"])
        fit_rms[ccd], fit_n_lines[ccd] = float(row["rms_pixel"]), int(row["n_lines"])

    return FibreDisplacementModel(
        fibres=fibres,
        fibre_modes=fibre_modes,
        offsets=offsets,
        surface_coefficients=coefficients,
        y_center=y_center,
        y_scale=y_scale,
        order_center=order_center,
        order_scale=order_scale,
        degree=degree,
        reference_night=str(header.get("REFNIGHT", "")),
        fit_rms_pixel=fit_rms,
        fit_n_lines=fit_n_lines,
    )


def ensure_fibre_displacement_model(
    *,
    config,
    paths,
    calibration_exposures,
    summed_fibth_line_sets,
    static_wavelength,
    thorium_atlas,
    static_fibth_reference_index=0,
    rank=2,
    degree=4,
    minimum_snr=5.0,
    maximum_snr=100.0,
    maximum_y_difference=1.0,
):
    """Load the reference fibre model, or construct a local candidate if absent."""
    if config.extraction_mode != "fibre":
        return None

    reference_night = (
        config.fibre_displacement_reference_night or config.reference_night
    )
    reference_file = paths.reference_product("fibre_displacement_model", reference_night)
    if reference_file.exists():
        return read_fibre_displacement_model(reference_file)

    candidate_file = paths.calibrations / f"fibre_displacement_model_{config.night}.fits"
    if candidate_file.exists() and not config.overwrite:
        return read_fibre_displacement_model(candidate_file)

    from . import diagnostics, thorium

    fibre_line_sets = {}
    for ccd in ("1", "2", "3"):
        exposure = calibration_exposures["FibTh"][ccd][int(static_fibth_reference_index)]
        if exposure.fibre_flux is None:
            raise RuntimeError(f"CCD{ccd} reference FibTh has no fibre extraction")
        fibre_line_sets[ccd] = thorium.ensure_fibth_fibre_lines(
            exposure,
            ccd=ccd,
            night=config.night,
            output_directory=paths.calibrations,
            thorium_atlas=thorium_atlas,
            reference_wavelength_function=static_wavelength[ccd].wavelength,
            exposure_index=int(static_fibth_reference_index),
            overwrite=config.overwrite,
            log_level=config.log_level,
        )

    fibre_displacement_model, matched_lines = fit_fibre_displacement_model(
        summed_fibth_line_sets,
        fibre_line_sets,
        rank=rank,
        degree=degree,
        minimum_snr=minimum_snr,
        maximum_snr=maximum_snr,
        maximum_y_difference=maximum_y_difference,
        reference_night=config.night,
    )
    write_fibre_displacement_model(fibre_displacement_model, candidate_file, overwrite=True)

    if config.diagnostics != "none":
        for ccd in ("1", "2", "3"):
            for quantity in ("displacement", "residual"):
                diagnostics.plot_fibre_displacement_qa(
                    matched_lines,
                    ccd=ccd,
                    quantity=quantity,
                    filename=(
                        paths.figures / "calibration"
                        / f"fibth_fibre_{quantity}_{config.night}_ccd{ccd}.png"
                    ),
                    minimum_snr=minimum_snr,
                    maximum_snr=maximum_snr,
                    dpi=300,
                )
    return (fibre_displacement_model, matched_lines)


# Backwards-compatible class name while external callers migrate.
FibreShiftModel = FibreDisplacementModel


@dataclass
class TimeShiftModel:
    """Time-interpolated common detector drift relative to a reference epoch."""

    mjd: np.ndarray
    coefficients: np.ndarray
    template: PixelShiftSurface
    source: str
    reference_mjd: float

    def __post_init__(self):
        self.mjd = np.asarray(self.mjd, dtype=float)
        self.coefficients = np.asarray(self.coefficients, dtype=float)
        idx = np.argsort(self.mjd)
        self.mjd = self.mjd[idx]
        self.coefficients = self.coefficients[idx]
        if self.coefficients.ndim != 3 or self.coefficients.shape[0] != len(self.mjd):
            raise ValueError("coefficients must have shape (n_times, ny, nm)")

    def coefficients_at(self, mjd):
        """Linearly interpolate/extrapolate every shift coefficient in time."""
        t = float(mjd)
        if len(self.mjd) == 1:
            return self.coefficients[0]
        if t <= self.mjd[0]:
            i0, i1 = 0, 1
        elif t >= self.mjd[-1]:
            i0, i1 = len(self.mjd) - 2, len(self.mjd) - 1
        else:
            i1 = int(np.searchsorted(self.mjd, t))
            i0 = i1 - 1
        dt = self.mjd[i1] - self.mjd[i0]
        weight = 0.0 if dt == 0 else (t - self.mjd[i0]) / dt
        return (1.0 - weight) * self.coefficients[i0] + weight * self.coefficients[i1]

    def shift(self, y, order, mjd):
        coefficients = self.coefficients_at(mjd)
        y_n, m_n = self.template._normalised_coordinates(y, order)
        return legval2d(y_n, m_n, coefficients)


def _fit_source_shift_series(
    line_sets,
    static_solution,
    *,
    source,
    reference_mjd,
    y_bounds,
    order_bounds,
    y_degree,
    order_degree,
):
    exposures = []
    for value in line_sets or []:
        table = _line_table(value)
        if len(table) == 0 or "mjd_mid" not in table.colnames:
            continue
        mjd_values = np.asarray(table["mjd_mid"], dtype=float)
        finite_mjd = mjd_values[np.isfinite(mjd_values)]
        if len(finite_mjd) == 0:
            continue
        mjd = float(np.nanmedian(finite_mjd))
        try:
            result, fitted = fit_shift_from_peak_table(
                table,
                static_solution,
                y_bounds=y_bounds,
                order_bounds=order_bounds,
                y_degree=y_degree,
                order_degree=order_degree,
            )
        except (ValueError, RuntimeError):
            continue
        exposures.append((mjd, result, fitted))

    if not exposures:
        return None, [], []
    exposures.sort(key=lambda item: item[0])
    times = np.asarray([item[0] for item in exposures], dtype=float)
    raw_coefficients = np.stack([item[1].surface.coefficients for item in exposures])
    if reference_mjd is None or not np.isfinite(reference_mjd):
        reference_mjd = float(np.nanmedian(times))
    else:
        reference_mjd = float(reference_mjd)

    # Define the temporal zero point at the requested reference epoch, not
    # merely at the nearest exposure.  This uses exactly the same piecewise
    # linear interpolation/extrapolation later used by TimeShiftModel.
    template = exposures[int(np.argmin(np.abs(times - reference_mjd)))][1].surface
    raw_model = TimeShiftModel(
        mjd=times,
        coefficients=raw_coefficients,
        template=template,
        source=str(source),
        reference_mjd=reference_mjd,
    )
    reference_coefficients = raw_model.coefficients_at(reference_mjd)
    relative = raw_coefficients - reference_coefficients[None, :, :]
    model = TimeShiftModel(
        mjd=times,
        coefficients=relative,
        template=template,
        source=str(source),
        reference_mjd=reference_mjd,
    )
    return model, exposures, reference_coefficients


def fit_time_corrections(
    *,
    static_solution: WavelengthSolution,
    simlc_line_sets=None,
    simth_line_sets=None,
    reference_mjd=None,
    y_bounds=None,
    order_bounds=None,
    y_degree=2,
    order_degree=1,
    preferred_source="SimLC",
):
    """Fit common temporal drift, preferring SimLC and using SimTh as fallback.

    Each calibration source is first differenced against its *own* reference
    exposure.  This removes the fixed SimLC/SimTh illumination/fibre offset
    before temporal drift is compared or transferred to science spectra.
    """

    series = {}
    all_exposures = {}
    for source, values in (("SimLC", simlc_line_sets), ("SimTh", simth_line_sets)):
        model, exposures, _ = _fit_source_shift_series(
            values,
            static_solution,
            source=source,
            reference_mjd=reference_mjd,
            y_bounds=y_bounds,
            order_bounds=order_bounds,
            y_degree=y_degree,
            order_degree=order_degree,
        )
        if model is not None:
            series[source] = model
            all_exposures[source] = exposures

    if not series:
        raise RuntimeError("No usable SimLC or SimTh exposure series for temporal drift")

    preferred_source = str(preferred_source)
    eligible = {name: model for name, model in series.items() if len(model.mjd) >= 2}
    if not eligible:
        raise RuntimeError("At least two usable SimLC or SimTh exposures are required for temporal drift")

    if preferred_source in eligible:
        primary = preferred_source
    elif "SimLC" in eligible:
        primary = "SimLC"
    elif "SimTh" in eligible:
        primary = "SimTh"
    else:
        primary = next(iter(eligible))

    rows = []
    for source, model in series.items():
        exposures = all_exposures[source]
        for i, (mjd, result, _) in enumerate(exposures):
            relative_surface = PixelShiftSurface(
                model.coefficients[i],
                model.template.y_center,
                model.template.y_scale,
                model.template.order_center,
                model.template.order_scale,
            )
            rows.append(
                dict(
                    source=source,
                    mjd=float(mjd),
                    reference=(abs(mjd - model.reference_mjd) < 1e-10),
                    n_lines=len(result.used),
                    n_used=int(np.count_nonzero(result.used)),
                    rms_surface_pixel=float(
                        np.sqrt(np.nanmean(result.residual_pixel[result.used] ** 2))
                    ),
                    median_relative_shift_pixel=float(
                        relative_surface.shift(
                            model.template.y_center, model.template.order_center
                        )
                    ),
                )
            )

    return series[primary], Table(rows=rows), series


@dataclass
class WavelengthModel:
    """Hierarchical static + fibre + temporal wavelength calibration."""

    static: WavelengthSolution
    fibre: FibreDisplacementModel | None = None
    time: TimeShiftModel | None = None
    ccd: str | None = None
    fixed_point_iterations: int = 3

    def reference_y(self, y_fibre, order, *, fibre=None, mjd=None):
        """Map current detector/fibre pixels onto the static summed-FibTh frame."""
        y_fibre, order = np.broadcast_arrays(
            np.asarray(y_fibre, dtype=float), np.asarray(order, dtype=float)
        )
        y_reference = y_fibre.copy()
        for _ in range(max(1, int(self.fixed_point_iterations))):
            shift = np.zeros_like(y_reference)
            if self.fibre is not None and fibre is not None:
                if self.ccd is None:
                    raise ValueError("ccd is required when a fibre displacement model is present")
                shift += self.fibre.displacement(self.ccd, int(fibre), y_reference, order)
            if self.time is not None and mjd is not None:
                shift += self.time.shift(y_reference, order, float(mjd))
            y_reference = y_fibre - shift
        return y_reference

    def wavelength(self, y_fibre, order, *, fibre=None, mjd=None):
        y_reference = self.reference_y(y_fibre, order, fibre=fibre, mjd=mjd)
        return self.static.wavelength(y_reference, order)

    def dispersion(self, y_fibre, order, *, fibre=None, mjd=None, step=1e-3):
        y_fibre = np.asarray(y_fibre, dtype=float)
        return (
            self.wavelength(y_fibre + step, order, fibre=fibre, mjd=mjd)
            - self.wavelength(y_fibre - step, order, fibre=fibre, mjd=mjd)
        ) / (2.0 * step)


WavelengthCalibration = WavelengthModel


# -----------------------------------------------------------------------------
# Hierarchical wavelength-model FITS I/O
# -----------------------------------------------------------------------------


def _surface_coefficient_rows(surface, **labels):
    rows = []
    for iy in range(surface.coefficients.shape[0]):
        for im in range(surface.coefficients.shape[1]):
            row = dict(labels)
            row.update(y_degree=iy, order_degree=im, coefficient=float(surface.coefficients[iy, im]))
            rows.append(row)
    return rows


def _set_surface_header(header, surface):
    header["YCENTER"] = float(surface.y_center)
    header["YSCALE"] = float(surface.y_scale)
    header["MCENTER"] = float(surface.order_center)
    header["MSCALE"] = float(surface.order_scale)


def write_wavelength_model_fits(
    model: WavelengthModel,
    filename: str | Path,
    *,
    ccd=None,
    reference_mjd=np.nan,
    overwrite=False,
):
    """Persist the compact hierarchical wavelength calibration."""

    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "velocereduction"
    primary.header["CONTENT"] = "Hierarchical wavelength calibration"
    if ccd is None:
        ccd = model.ccd
    if ccd is not None:
        primary.header["CCD"] = str(ccd)
    if np.isfinite(reference_mjd):
        primary.header["MJDREF"] = float(reference_mjd)
    if model.fibre is not None:
        reference_night = model.fibre.reference_night or "UNKNOWN"
        primary.header["FIBREF"] = f"fibre_displacement_model_{reference_night}.fits"

    static_hdu = fits.BinTableHDU(
        Table(rows=_surface_coefficient_rows(model.static)), name="STATIC"
    )
    _set_surface_header(static_hdu.header, model.static)

    hdus = [primary, static_hdu]
    if model.time is not None:
        rows = []
        for mjd, coefficients in zip(model.time.mjd, model.time.coefficients):
            surface = PixelShiftSurface(
                coefficients,
                model.time.template.y_center,
                model.time.template.y_scale,
                model.time.template.order_center,
                model.time.template.order_scale,
            )
            rows.extend(_surface_coefficient_rows(surface, mjd=float(mjd)))
        hdu = fits.BinTableHDU(Table(rows=rows), name="TIME_SHIFT")
        _set_surface_header(hdu.header, model.time.template)
        hdu.header["SOURCE"] = model.time.source
        hdu.header["MJDREF"] = float(model.time.reference_mjd)
        hdus.append(hdu)

    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)


def _surface_from_table(table, header):
    ny = int(np.max(table["y_degree"])) + 1
    nm = int(np.max(table["order_degree"])) + 1
    coefficients = np.zeros((ny, nm), dtype=float)
    for row in table:
        coefficients[int(row["y_degree"]), int(row["order_degree"])] = float(row["coefficient"])
    return PixelShiftSurface(
        coefficients,
        float(header["YCENTER"]),
        float(header["YSCALE"]),
        float(header["MCENTER"]),
        float(header["MSCALE"]),
    )


def read_wavelength_model_fits(filename: str | Path) -> tuple[WavelengthModel, fits.Header]:
    """Read a hierarchical wavelength calibration written above."""

    with fits.open(filename) as hdul:
        header = hdul[0].header.copy()
        static_table = Table(hdul["STATIC"].data)
        static_shift = _surface_from_table(static_table, hdul["STATIC"].header)
        static = WavelengthSolution(
            static_shift.coefficients,
            static_shift.y_center,
            static_shift.y_scale,
            static_shift.order_center,
            static_shift.order_scale,
        )

        fibre_model = None

        time_model = None
        if "TIME_SHIFT" in hdul:
            table = Table(hdul["TIME_SHIFT"].data)
            times = np.unique(np.asarray(table["mjd"], dtype=float))
            surfaces = [
                _surface_from_table(
                    table[np.asarray(table["mjd"], dtype=float) == t],
                    hdul["TIME_SHIFT"].header,
                )
                for t in times
            ]
            time_model = TimeShiftModel(
                times,
                np.stack([surface.coefficients for surface in surfaces]),
                surfaces[0],
                str(hdul["TIME_SHIFT"].header.get("SOURCE", "")),
                float(hdul["TIME_SHIFT"].header.get("MJDREF", np.nan)),
            )

    return WavelengthModel(
        static=static, fibre=fibre_model, time=time_model,
        ccd=str(header.get("CCD", "")) or None,
    ), header
