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

Peak measurement/identification belongs in ``calibration.py``.  This module
starts from already identified calibration lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legder, legval2d, legvander

from astropy.io import fits
from astropy.table import Table

from .calibration import CalibrationPeakFlag

SPEED_OF_LIGHT_MPS = 299_792_458.0


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
        y = np.asarray(y, dtype=float)
        order = np.asarray(order, dtype=float)

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
    y_degree=4,
    order_degree=3,
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
        y_bounds = (
            float(np.nanmin(y[finite])),
            float(np.nanmax(y[finite])),
        )

    if order_bounds is None:
        order_bounds = (
            float(np.nanmin(order[finite])),
            float(np.nanmax(order[finite])),
        )

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
    y_bounds: tuple[float, float],
    order_bounds: tuple[float, float],
    y_degree: int = 4,
    order_degree: int = 3,
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
