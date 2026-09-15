"""SimLC-specific line-spread-function inference and centroid refinement.

The generic calibration module is responsible for detecting emission peaks and
assigning comb modes.  This module then uses the identified SimLC modes to
infer a shared, order-dependent effective line-spread function (eLSF) and to
remeasure the mode centroids before the wavelength solution is fitted.

Two representations are supported:

``moffat``
    A normalized Moffat profile integrated over detector pixels.  This is the
    robust parametric fallback.

``empirical``
    A pixel-convolved effective LSF reconstructed from many comb modes that
    sample different sub-pixel phases.  A penalized cubic B-spline is used for
    the reconstruction; the calibration product itself stores the resulting
    profile on a regular sub-pixel grid so downstream code is independent of
    the reconstruction method.

The intended reduction flow is

    simlc.measure_simlc_lines()
        -> detect/seed peaks
        -> assign exact comb modes/frequencies
        -> infer and validate the order-dependent LSF
        -> refit mode centroids with the adopted LSF
        -> wavelength.fit_wavelength_from_peak_table()

The empirical eLSF is already convolved with the detector-pixel response and
must therefore *not* be integrated over the pixel again when fitting a mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse import lil_matrix
from scipy.special import stdtr

from .calibration import (
    CalibrationLineSet,
    CalibrationPeakConfig,
    CalibrationPeakFlag,
    _clear_identification,
    _match_peak_table_to_predicted_lines,
    _predict_reference_lines_for_order,
    _refresh_used_for_wavelength_fit,
    measure_calibration_peaks,
)


@dataclass(frozen=True)
class SimLCLSFConfig:
    """Settings controlling SimLC eLSF inference and validation."""

    # Lines used to infer the shared LSF.
    minimum_lsf_snr: float = 30.0
    minimum_neighbour_distance: float = 7.2
    minimum_lines: int = 12
    maximum_lines_per_order: int = 60

    # Local line windows and centroid freedom.
    fit_half_width: int = 3
    maximum_centroid_shift: float = 0.55

    # Deterministic held-out validation.
    holdout_modulo: int = 5
    inner_validation_modulo: int = 4
    minimum_holdout_lines: int = 3

    # Empirical eLSF representation.
    lsf_half_width: float = 3.5
    lsf_grid_step: float = 0.025
    spline_knot_spacing: float = 0.25
    smoothing_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    empirical_iterations: int = 3

    # Quality cuts after re-fitting with the adopted LSF.  We intentionally do
    # not impose a reduced-chi2 cut here because the validation shows that the
    # extracted variance does not capture all residual structure.
    minimum_fit_snr: float = 5.0
    maximum_y_uncertainty: float = 0.30

    # Adopt the empirical eLSF only if it predicts held-out modes at least this
    # much better than the Moffat.  Zero gives a conservative no-regret switch.
    minimum_empirical_improvement_percent: float = 0.0

    # Orders with too few suitable modes may borrow the nearest successful LSF.
    allow_nearest_order_fallback: bool = True


@dataclass
class SimLCLSF:
    """Order-dependent sampled effective LSF calibration."""

    orders: np.ndarray
    offset: np.ndarray
    profile: np.ndarray
    metadata: Table

    def __post_init__(self):
        self.orders = np.asarray(self.orders, dtype=int)
        self.offset = np.asarray(self.offset, dtype=float)
        self.profile = np.asarray(self.profile, dtype=float)
        if self.profile.shape != (len(self.orders), len(self.offset)):
            raise ValueError("profile must have shape (n_orders, n_offset)")

    def order_index(self, order: int) -> int:
        index = np.flatnonzero(self.orders == int(order))
        if len(index) != 1:
            raise KeyError(f"No unique SimLC LSF for order {order}")
        return int(index[0])

    def evaluate(self, order: int, offset: np.ndarray | float) -> np.ndarray:
        """Evaluate the sampled eLSF, returning zero outside the stored grid."""

        i = self.order_index(order)
        return np.interp(
            np.asarray(offset, dtype=float),
            self.offset,
            self.profile[i],
            left=0.0,
            right=0.0,
        )

    def model_name(self, order: int) -> str:
        i = self.order_index(order)
        return str(self.metadata["model"][i])


# -----------------------------------------------------------------------------
# Profile definitions
# -----------------------------------------------------------------------------


def pixel_integrated_moffat(
    offset: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Normalized 1-D Moffat LSF integrated over unit-width detector pixels.

    The intrinsic profile is proportional to

        [1 + (u / alpha)**2]**(-beta),

    with beta > 1/2.  Rewriting it as a rescaled Student-t distribution makes
    the finite-pixel integral analytic.
    """

    offset = np.asarray(offset, dtype=float)
    if alpha <= 0 or beta <= 0.5:
        return np.full_like(offset, np.nan)

    nu = 2.0 * beta - 1.0
    scale = alpha / np.sqrt(nu)
    lo = (offset - 0.5) / scale
    hi = (offset + 0.5) / scale
    return stdtr(nu, hi) - stdtr(nu, lo)


def _normalise_sampled_profile(offset: np.ndarray, profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    profile = np.where(np.isfinite(profile), profile, 0.0)
    profile = np.clip(profile, 0.0, None)
    area = float(np.trapezoid(profile, offset))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("LSF profile has non-positive normalization")
    return profile / area


def effective_fwhm(offset: np.ndarray, profile: np.ndarray) -> float:
    """FWHM of a sampled pixel-convolved LSF."""

    offset = np.asarray(offset, dtype=float)
    profile = np.asarray(profile, dtype=float)
    if len(offset) < 3 or not np.any(np.isfinite(profile)):
        return np.nan

    peak_index = int(np.nanargmax(profile))
    half = 0.5 * float(profile[peak_index])

    left = np.where(profile[: peak_index + 1] <= half)[0]
    right = np.where(profile[peak_index:] <= half)[0]
    if len(left) == 0 or len(right) == 0:
        return np.nan

    il = int(left[-1])
    ir = int(peak_index + right[0])
    if il + 1 >= len(offset) or ir - 1 < 0:
        return np.nan

    x_left = np.interp(half, profile[il : il + 2], offset[il : il + 2])
    # Reverse the declining branch so np.interp sees an increasing x array.
    x_right = np.interp(
        half,
        profile[ir - 1 : ir + 1][::-1],
        offset[ir - 1 : ir + 1][::-1],
    )
    return float(x_right - x_left)


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------


def _order_lookup(orders: np.ndarray) -> dict[int, int]:
    orders = np.asarray(orders, dtype=int)
    if len(np.unique(orders)) != len(orders):
        raise ValueError("orders must contain unique physical order numbers")
    return {int(order): i for i, order in enumerate(orders)}


def _line_data(
    counts: np.ndarray,
    variance: np.ndarray | None,
    order_lookup: dict[int, int],
    order: int,
    y0: float,
    half_width: int,
):
    j = order_lookup[int(order)]
    center_pixel = int(round(float(y0)))
    pixels = np.arange(center_pixel - half_width, center_pixel + half_width + 1)
    if pixels[0] < 0 or pixels[-1] >= counts.shape[1]:
        return None

    signal = np.asarray(counts[j, pixels], dtype=float)
    if variance is None:
        edge = np.r_[signal[:2], signal[-2:]]
        sigma = 1.4826 * np.nanmedian(np.abs(edge - np.nanmedian(edge)))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max(np.sqrt(max(np.nanmedian(np.abs(signal)), 1.0)), 1.0)
        var = np.full(len(signal), sigma**2)
    else:
        var = np.asarray(variance[j, pixels], dtype=float)

    valid = np.isfinite(signal) & np.isfinite(var) & (var > 0)
    if np.count_nonzero(valid) != len(signal):
        return None

    return pixels.astype(float), signal, var


def _cap_evenly(indices: np.ndarray, n: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= n:
        return indices
    take = np.unique(np.linspace(0, len(indices) - 1, n).round().astype(int))
    return indices[take]


def select_lsf_lines(
    peak_table: Table,
    order: int,
    *,
    config: SimLCLSFConfig | None = None,
) -> np.ndarray:
    """Indices of bright, isolated, identified modes suitable for LSF inference."""

    if config is None:
        config = SimLCLSFConfig()

    q = np.asarray(peak_table["order"], dtype=int) == int(order)
    q &= np.asarray(peak_table["comb_mode"], dtype=np.int64) >= 0
    q &= np.isfinite(np.asarray(peak_table["y"], dtype=float))
    q &= np.asarray(peak_table["signal_to_noise"], dtype=float) >= config.minimum_lsf_snr

    if "nearest_peak_distance_pixel" in peak_table.colnames:
        q &= (
            np.asarray(peak_table["nearest_peak_distance_pixel"], dtype=float)
            >= config.minimum_neighbour_distance
        )

    # Do not let a poor *Gaussian* profile fit exclude an otherwise excellent
    # comb line.  Only reject flags that are profile-independent here.
    forbidden = int(
        CalibrationPeakFlag.EDGE
        | CalibrationPeakFlag.SATURATED
        | CalibrationPeakFlag.BLEND_CANDIDATE
        | CalibrationPeakFlag.AMBIGUOUS_MATCH
        | CalibrationPeakFlag.UNMATCHED
    )
    flags = np.asarray(peak_table["quality_flag"], dtype=np.int64)
    q &= (flags & forbidden) == 0

    indices = np.flatnonzero(q)
    if len(indices) == 0:
        return indices

    y = np.asarray(peak_table["y"][indices], dtype=float)
    indices = indices[np.argsort(y)]
    return _cap_evenly(indices, config.maximum_lines_per_order)


def _split_train_test(
    peak_table: Table,
    indices: np.ndarray,
    *,
    modulo: int,
    minimum_test: int,
) -> tuple[np.ndarray, np.ndarray]:
    modes = np.asarray(peak_table["comb_mode"][indices], dtype=np.int64)
    test_mask = (modes % modulo) == 0
    train, test = indices[~test_mask], indices[test_mask]

    if len(test) < minimum_test or len(train) < 2 * minimum_test:
        # Deterministic spatial fallback if the mode-number split is sparse.
        test_mask = np.zeros(len(indices), dtype=bool)
        test_mask[::modulo] = True
        train, test = indices[~test_mask], indices[test_mask]

    return train, test


# -----------------------------------------------------------------------------
# Shared Moffat fit
# -----------------------------------------------------------------------------


def _fit_shared_moffat(
    counts: np.ndarray,
    variance: np.ndarray | None,
    orders: np.ndarray,
    peak_table: Table,
    indices: np.ndarray,
    *,
    config: SimLCLSFConfig,
) -> tuple[float, float]:
    """Fit one Moffat shape shared by all selected lines in an order."""

    lookup = _order_lookup(orders)
    arrays = []
    kept_indices = []
    for i in indices:
        arr = _line_data(
            counts,
            variance,
            lookup,
            int(peak_table["order"][i]),
            float(peak_table["y"][i]),
            config.fit_half_width,
        )
        if arr is not None:
            arrays.append(arr)
            kept_indices.append(i)

    if len(arrays) < config.minimum_lines:
        raise ValueError("Too few usable lines for shared Moffat fit")

    nuisance = []
    for i, (pixels, signal, _) in zip(kept_indices, arrays):
        background = float(np.nanmedian(np.r_[signal[:2], signal[-2:]]))
        amplitude = max(float(np.sum(np.clip(signal - background, 0.0, None))), 1.0)
        nuisance.extend([float(peak_table["y"][i]), amplitude, background])

    nline = len(arrays)
    x0 = np.r_[0.8, 2.5, np.asarray(nuisance)]
    lo = np.r_[0.10, 0.55, np.tile([-np.inf, 0.0, -np.inf], nline)]
    hi = np.r_[4.00, 30.0, np.tile([np.inf, np.inf, np.inf], nline)]

    for j, i in enumerate(kept_indices):
        y0 = float(peak_table["y"][i])
        lo[2 + 3 * j] = y0 - config.maximum_centroid_shift
        hi[2 + 3 * j] = y0 + config.maximum_centroid_shift

    def residuals(parameters):
        alpha, beta = parameters[:2]
        out = []
        for j, (pixels, signal, var) in enumerate(arrays):
            center, amplitude, background = parameters[2 + 3 * j : 5 + 3 * j]
            model = background + amplitude * pixel_integrated_moffat(
                pixels - center, alpha, beta
            )
            out.append((signal - model) / np.sqrt(var))
        return np.concatenate(out)

    nobs = sum(len(a[0]) for a in arrays)
    sparsity = lil_matrix((nobs, len(x0)), dtype=int)
    row0 = 0
    for j, arr in enumerate(arrays):
        n = len(arr[0])
        sparsity[row0 : row0 + n, :2] = 1
        sparsity[row0 : row0 + n, 2 + 3 * j : 5 + 3 * j] = 1
        row0 += n

    fit = least_squares(
        residuals,
        x0,
        bounds=(lo, hi),
        jac_sparsity=sparsity.tocsr(),
        x_scale="jac",
        max_nfev=250,
        ftol=2e-8,
        xtol=2e-8,
        gtol=2e-8,
    )
    if not fit.success or not np.all(np.isfinite(fit.x[:2])):
        raise RuntimeError("Shared Moffat fit failed")
    return float(fit.x[0]), float(fit.x[1])


# -----------------------------------------------------------------------------
# Fixed-profile line fits and empirical eLSF
# -----------------------------------------------------------------------------


def _fit_line_simple(
    pixels: np.ndarray,
    signal: np.ndarray,
    variance: np.ndarray,
    y0: float,
    profile_function: Callable[[np.ndarray], np.ndarray],
    *,
    maximum_centroid_shift: float,
):
    """Fast center/amplitude/background fit used during LSF reconstruction."""

    weight = 1.0 / np.sqrt(variance)

    def solve(center):
        p = profile_function(pixels - center)
        design = np.c_[p, np.ones_like(p)]
        coefficients = np.linalg.lstsq(
            design * weight[:, None], signal * weight, rcond=None
        )[0]
        amplitude, background = coefficients
        if amplitude < 0:
            return 1e50, (center, amplitude, background)
        residual = (signal - design @ coefficients) * weight
        return float(residual @ residual), (center, amplitude, background)

    optimum = minimize_scalar(
        lambda center: solve(center)[0],
        bounds=(y0 - maximum_centroid_shift, y0 + maximum_centroid_shift),
        method="bounded",
        options={"xatol": 2e-5},
    )
    return solve(float(optimum.x))


def _fit_penalized_spline(
    offset: np.ndarray,
    profile_value: np.ndarray,
    weight: np.ndarray,
    *,
    smoothing: float,
    knot_spacing: float,
    half_width: float,
    degree: int = 3,
) -> BSpline:
    xmin, xmax = -half_width - 0.1, half_width + 0.1
    internal = np.arange(xmin + knot_spacing, xmax - 0.5 * knot_spacing, knot_spacing)
    knots = np.r_[np.repeat(xmin, degree + 1), internal, np.repeat(xmax, degree + 1)]
    design = BSpline.design_matrix(
        np.clip(offset, xmin + 1e-9, xmax - 1e-9), knots, degree
    ).toarray()

    second_difference = np.diff(np.eye(design.shape[1]), n=2, axis=0)
    sqrt_weight = np.sqrt(np.clip(weight, 1e-12, None))
    weighted_design = design * sqrt_weight[:, None]
    weighted_value = profile_value * sqrt_weight

    matrix = (
        weighted_design.T @ weighted_design
        + smoothing * (second_difference.T @ second_difference)
    )
    rhs = weighted_design.T @ weighted_value
    coefficients = np.linalg.solve(matrix, rhs)
    return BSpline(knots, coefficients, degree, extrapolate=False)


def _build_empirical_profile(
    counts: np.ndarray,
    variance: np.ndarray | None,
    orders: np.ndarray,
    peak_table: Table,
    indices: np.ndarray,
    moffat_shape: tuple[float, float],
    *,
    smoothing: float,
    config: SimLCLSFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a shared, pixel-convolved eLSF from many sub-pixel phases."""

    lookup = _order_lookup(orders)
    line_data = []
    kept = []
    for i in indices:
        arr = _line_data(
            counts,
            variance,
            lookup,
            int(peak_table["order"][i]),
            float(peak_table["y"][i]),
            config.fit_half_width,
        )
        if arr is not None:
            line_data.append(arr)
            kept.append(i)

    alpha, beta = moffat_shape
    base = lambda u: pixel_integrated_moffat(u, alpha, beta)
    fits = []
    for i, (pixels, signal, var) in zip(kept, line_data):
        fits.append(
            _fit_line_simple(
                pixels,
                signal,
                var,
                float(peak_table["y"][i]),
                base,
                maximum_centroid_shift=config.maximum_centroid_shift,
            )[1]
        )

    spline = None
    for _ in range(config.empirical_iterations):
        all_offset, all_profile, all_weight = [], [], []
        for (pixels, signal, var), (center, amplitude, background) in zip(line_data, fits):
            if amplitude <= 0:
                continue
            all_offset.extend(pixels - center)
            all_profile.extend((signal - background) / amplitude)
            all_weight.extend((amplitude * amplitude) / var)

        spline = _fit_penalized_spline(
            np.asarray(all_offset),
            np.asarray(all_profile),
            np.asarray(all_weight),
            smoothing=smoothing,
            knot_spacing=config.spline_knot_spacing,
            half_width=config.lsf_half_width,
        )

        empirical = lambda u, s=spline: np.nan_to_num(s(u), nan=0.0)
        fits = []
        for i, (pixels, signal, var) in zip(kept, line_data):
            fits.append(
                _fit_line_simple(
                    pixels,
                    signal,
                    var,
                    float(peak_table["y"][i]),
                    empirical,
                    maximum_centroid_shift=config.maximum_centroid_shift,
                )[1]
            )

    offset_grid = np.arange(
        -config.lsf_half_width,
        config.lsf_half_width + 0.5 * config.lsf_grid_step,
        config.lsf_grid_step,
    )
    profile_grid = np.nan_to_num(spline(offset_grid), nan=0.0)
    profile_grid = _normalise_sampled_profile(offset_grid, profile_grid)
    return offset_grid, profile_grid


def _score_profile(
    counts: np.ndarray,
    variance: np.ndarray | None,
    orders: np.ndarray,
    peak_table: Table,
    indices: np.ndarray,
    profile_function: Callable[[np.ndarray], np.ndarray],
    *,
    config: SimLCLSFConfig,
) -> float:
    lookup = _order_lookup(orders)
    total = 0.0
    for i in indices:
        arr = _line_data(
            counts,
            variance,
            lookup,
            int(peak_table["order"][i]),
            float(peak_table["y"][i]),
            config.fit_half_width,
        )
        if arr is None:
            continue
        pixels, signal, var = arr
        chi2, _ = _fit_line_simple(
            pixels,
            signal,
            var,
            float(peak_table["y"][i]),
            profile_function,
            maximum_centroid_shift=config.maximum_centroid_shift,
        )
        total += chi2
    return float(total)


def _select_smoothing(
    counts: np.ndarray,
    variance: np.ndarray | None,
    orders: np.ndarray,
    peak_table: Table,
    train: np.ndarray,
    moffat_shape: tuple[float, float],
    *,
    config: SimLCLSFConfig,
) -> float:
    inner_train, inner_test = _split_train_test(
        peak_table,
        train,
        modulo=config.inner_validation_modulo,
        minimum_test=3,
    )
    if len(inner_train) < config.minimum_lines or len(inner_test) < 3:
        return float(config.smoothing_grid[len(config.smoothing_grid) // 2])

    scores = []
    for smoothing in config.smoothing_grid:
        grid, profile = _build_empirical_profile(
            counts,
            variance,
            orders,
            peak_table,
            inner_train,
            moffat_shape,
            smoothing=float(smoothing),
            config=config,
        )
        f = lambda u, x=grid, p=profile: np.interp(u, x, p, left=0.0, right=0.0)
        scores.append(
            _score_profile(
                counts,
                variance,
                orders,
                peak_table,
                inner_test,
                f,
                config=config,
            )
        )
    return float(config.smoothing_grid[int(np.argmin(scores))])


# -----------------------------------------------------------------------------
# Public LSF inference
# -----------------------------------------------------------------------------


def fit_simlc_lsf(
    counts: np.ndarray,
    orders: np.ndarray,
    peak_table: Table,
    *,
    variance: np.ndarray | None = None,
    config: SimLCLSFConfig | None = None,
) -> SimLCLSF:
    """Infer an order-dependent SimLC eLSF with held-out model validation.

    ``peak_table`` should already have comb modes assigned by
    ``identify_simlc_peaks``.  The Moffat is always fitted first.  Where enough
    suitable modes are available, an empirical eLSF is reconstructed from the
    training modes and adopted only if it improves prediction of held-out modes.
    The final adopted profile is then refitted using all suitable modes.
    """

    if config is None:
        config = SimLCLSFConfig()

    counts = np.asarray(counts, dtype=float)
    orders = np.asarray(orders, dtype=int)
    if counts.ndim != 2 or counts.shape[0] != len(orders):
        raise ValueError("counts must have shape (n_orders, n_pixels)")
    if variance is not None:
        variance = np.asarray(variance, dtype=float)
        if variance.shape != counts.shape:
            raise ValueError("variance must match counts")

    offset_grid = np.arange(
        -config.lsf_half_width,
        config.lsf_half_width + 0.5 * config.lsf_grid_step,
        config.lsf_grid_step,
    )

    profiles: dict[int, np.ndarray] = {}
    metadata: dict[int, dict] = {}

    for order in orders:
        indices = select_lsf_lines(peak_table, int(order), config=config)
        if len(indices) < config.minimum_lines:
            continue

        train, test = _split_train_test(
            peak_table,
            indices,
            modulo=config.holdout_modulo,
            minimum_test=config.minimum_holdout_lines,
        )
        if len(train) < config.minimum_lines or len(test) < config.minimum_holdout_lines:
            continue

        try:
            moffat_train = _fit_shared_moffat(
                counts, variance, orders, peak_table, train, config=config
            )
        except (ValueError, RuntimeError):
            continue

        moffat_function = lambda u, p=moffat_train: pixel_integrated_moffat(u, *p)
        moffat_chi2 = _score_profile(
            counts,
            variance,
            orders,
            peak_table,
            test,
            moffat_function,
            config=config,
        )

        smoothing = np.nan
        empirical_chi2 = np.nan
        empirical_improvement = np.nan
        adopted_model = "moffat"

        try:
            smoothing = _select_smoothing(
                counts,
                variance,
                orders,
                peak_table,
                train,
                moffat_train,
                config=config,
            )
            empirical_grid, empirical_train = _build_empirical_profile(
                counts,
                variance,
                orders,
                peak_table,
                train,
                moffat_train,
                smoothing=smoothing,
                config=config,
            )
            empirical_function = lambda u, x=empirical_grid, p=empirical_train: np.interp(
                u, x, p, left=0.0, right=0.0
            )
            empirical_chi2 = _score_profile(
                counts,
                variance,
                orders,
                peak_table,
                test,
                empirical_function,
                config=config,
            )
            empirical_improvement = 100.0 * (moffat_chi2 - empirical_chi2) / moffat_chi2
            if empirical_improvement >= config.minimum_empirical_improvement_percent:
                adopted_model = "empirical"
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            adopted_model = "moffat"

        # Refit the adopted profile using every suitable line after model choice.
        moffat_all = _fit_shared_moffat(
            counts, variance, orders, peak_table, indices, config=config
        )
        if adopted_model == "empirical":
            final_grid, final_profile = _build_empirical_profile(
                counts,
                variance,
                orders,
                peak_table,
                indices,
                moffat_all,
                smoothing=smoothing,
                config=config,
            )
            final_profile = np.interp(offset_grid, final_grid, final_profile)
        else:
            final_profile = pixel_integrated_moffat(offset_grid, *moffat_all)
            final_profile = _normalise_sampled_profile(offset_grid, final_profile)

        profiles[int(order)] = final_profile
        metadata[int(order)] = dict(
            order=int(order),
            model=adopted_model,
            source_order=int(order),
            n_lines=int(len(indices)),
            n_train=int(len(train)),
            n_test=int(len(test)),
            moffat_alpha=float(moffat_all[0]),
            moffat_beta=float(moffat_all[1]),
            smoothing=float(smoothing),
            heldout_chi2_moffat=float(moffat_chi2),
            heldout_chi2_empirical=float(empirical_chi2),
            empirical_improvement_percent=float(empirical_improvement),
            effective_fwhm=effective_fwhm(offset_grid, final_profile),
        )

    if not profiles:
        raise RuntimeError("No SimLC orders contained enough suitable modes to infer an LSF")

    # Fill sparse orders by borrowing the nearest measured order.  The metadata
    # records this explicitly so it remains visible in QA and FITS products.
    if config.allow_nearest_order_fallback:
        fitted_orders = np.array(sorted(profiles), dtype=int)
        for order in orders:
            order = int(order)
            if order in profiles:
                continue
            source = int(fitted_orders[np.argmin(np.abs(fitted_orders - order))])
            profiles[order] = profiles[source].copy()
            row = dict(metadata[source])
            row.update(
                order=order,
                model="nearest_order",
                source_order=source,
                n_lines=0,
                n_train=0,
                n_test=0,
                empirical_improvement_percent=np.nan,
            )
            metadata[order] = row

    output_orders = np.asarray(orders, dtype=int)
    output_profiles = np.vstack([profiles[int(order)] for order in output_orders])
    output_meta = Table(rows=[metadata[int(order)] for order in output_orders])
    return SimLCLSF(output_orders, offset_grid, output_profiles, output_meta)


# -----------------------------------------------------------------------------
# Refit all identified SimLC modes with the adopted LSF
# -----------------------------------------------------------------------------


def _fit_fixed_profile_line(
    pixels: np.ndarray,
    signal: np.ndarray,
    variance: np.ndarray,
    y0: float,
    profile_function: Callable[[np.ndarray], np.ndarray],
    *,
    maximum_centroid_shift: float,
) -> dict:
    y_reference = float(round(y0))
    edge_background = float(np.nanmedian(np.r_[signal[:2], signal[-2:]]))
    amplitude = max(float(np.sum(np.clip(signal - edge_background, 0.0, None))), 1.0)

    initial = np.array([amplitude, y0, edge_background, 0.0])
    lower = np.array([0.0, y0 - maximum_centroid_shift, -np.inf, -np.inf])
    upper = np.array([np.inf, y0 + maximum_centroid_shift, np.inf, np.inf])
    sigma = np.sqrt(variance)

    def residuals(parameters):
        amp, center, background, slope = parameters
        model = (
            background
            + slope * (pixels - y_reference)
            + amp * profile_function(pixels - center)
        )
        return (model - signal) / sigma

    robust = least_squares(
        residuals,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
    )
    fit = least_squares(
        residuals,
        robust.x,
        bounds=(lower, upper),
        loss="linear",
    )

    amp, center, background, slope = fit.x
    model = (
        background
        + slope * (pixels - y_reference)
        + amp * profile_function(pixels - center)
    )
    residual = signal - model
    dof = max(1, len(signal) - len(fit.x))
    chi2 = float(np.sum((residual / sigma) ** 2))
    reduced_chi2 = chi2 / dof

    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
    covariance *= max(1.0, reduced_chi2)
    uncertainty = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    return dict(
        integrated_counts=float(amp),
        integrated_counts_uncertainty=float(uncertainty[0]),
        y=float(center),
        y_uncertainty=float(uncertainty[1]),
        background=float(background),
        background_slope=float(slope),
        reduced_chi2=float(reduced_chi2),
        fit_rms=float(np.sqrt(np.mean(residual**2))),
        fit_success=bool(fit.success),
    )


def refit_simlc_peaks(
    counts: np.ndarray,
    orders: np.ndarray,
    peak_table: Table,
    lsf: SimLCLSF,
    *,
    variance: np.ndarray | None = None,
    config: SimLCLSFConfig | None = None,
) -> Table:
    """Remeasure identified comb centroids using the adopted shared eLSF.

    The returned table preserves the generic calibration schema: ``y`` and
    ``y_uncertainty`` are replaced by the eLSF-based measurements so
    ``wavelength.py`` requires no special SimLC handling.  The original values
    and applied centroid correction are retained in additional columns.
    """

    if config is None:
        config = SimLCLSFConfig()

    counts = np.asarray(counts, dtype=float)
    orders = np.asarray(orders, dtype=int)
    if variance is not None:
        variance = np.asarray(variance, dtype=float)

    output = peak_table.copy(copy_data=True)
    n = len(output)
    if "y_initial" not in output.colnames:
        output["y_initial"] = np.asarray(output["y"], dtype=float).copy()
    if "y_uncertainty_initial" not in output.colnames:
        output["y_uncertainty_initial"] = np.asarray(
            output["y_uncertainty"], dtype=float
        ).copy()
    if "lsf_y_shift" not in output.colnames:
        output["lsf_y_shift"] = np.full(n, np.nan)
    if "lsf_model" not in output.colnames:
        output["lsf_model"] = np.full(n, "", dtype="U16")
    if "lsf_source_order" not in output.colnames:
        output["lsf_source_order"] = np.full(n, -1, dtype=int)

    lookup = _order_lookup(orders)

    # Clear flags that were based on the initial Gaussian profile fit.  Retain
    # detector, saturation, blend and identification flags.
    refit_bits = int(
        CalibrationPeakFlag.BAD_PROFILE_FIT
        | CalibrationPeakFlag.LARGE_CENTROID_ERROR
        | CalibrationPeakFlag.WIDTH_OUTLIER
        | CalibrationPeakFlag.LOW_SNR
    )

    for i in range(n):
        order = int(output["order"][i])
        mode = int(output["comb_mode"][i])
        if mode < 0 or order not in lookup:
            continue

        y0 = float(output["y_initial"][i])
        arr = _line_data(
            counts,
            variance,
            lookup,
            order,
            y0,
            config.fit_half_width,
        )
        if arr is None:
            continue
        pixels, signal, var = arr

        profile_function = lambda u, o=order: lsf.evaluate(o, u)
        result = _fit_fixed_profile_line(
            pixels,
            signal,
            var,
            y0,
            profile_function,
            maximum_centroid_shift=config.maximum_centroid_shift,
        )

        output["y"][i] = result["y"]
        output["y_uncertainty"][i] = result["y_uncertainty"]
        output["pixel_phase"][i] = result["y"] - np.round(result["y"])
        output["integrated_counts"][i] = result["integrated_counts"]
        output["integrated_counts_uncertainty"][i] = result[
            "integrated_counts_uncertainty"
        ]
        output["background"][i] = result["background"]
        output["background_slope"][i] = result["background_slope"]
        output["reduced_chi2"][i] = result["reduced_chi2"]
        output["fit_rms"][i] = result["fit_rms"]
        output["fit_success"][i] = result["fit_success"]
        output["lsf_y_shift"][i] = result["y"] - y0

        j = lsf.order_index(order)
        output["lsf_model"][i] = str(lsf.metadata["model"][j])
        output["lsf_source_order"][i] = int(lsf.metadata["source_order"][j])
        effective_width = float(lsf.metadata["effective_fwhm"][j])
        if "fwhm" in output.colnames:
            output["fwhm"][i] = effective_width
        if "fwhm_uncertainty" in output.colnames:
            output["fwhm_uncertainty"][i] = np.nan
        if "fwhm_pixel" not in output.colnames:
            output["fwhm_pixel"] = np.full(n, np.nan)
        if "fwhm_uncertainty_pixel" not in output.colnames:
            output["fwhm_uncertainty_pixel"] = np.full(n, np.nan)
        output["fwhm_pixel"][i] = effective_width
        output["fwhm_uncertainty_pixel"][i] = np.nan
        if "intrinsic_peak_amplitude" in output.colnames:
            output["intrinsic_peak_amplitude"][i] = np.nan

        amplitude_uncertainty = result["integrated_counts_uncertainty"]
        if amplitude_uncertainty > 0:
            output["signal_to_noise"][i] = (
                result["integrated_counts"] / amplitude_uncertainty
            )

        flag = int(output["quality_flag"][i]) & ~refit_bits
        if not result["fit_success"] or not np.isfinite(result["reduced_chi2"]):
            flag |= int(CalibrationPeakFlag.BAD_PROFILE_FIT)
        if (
            not np.isfinite(result["y_uncertainty"])
            or result["y_uncertainty"] > config.maximum_y_uncertainty
        ):
            flag |= int(CalibrationPeakFlag.LARGE_CENTROID_ERROR)
        if (
            not np.isfinite(float(output["signal_to_noise"][i]))
            or float(output["signal_to_noise"][i]) < config.minimum_fit_snr
        ):
            flag |= int(CalibrationPeakFlag.LOW_SNR)
        output["quality_flag"][i] = flag

    identified = np.isfinite(np.asarray(output["wavelength_nm"], dtype=float))
    good = np.asarray(output["quality_flag"], dtype=np.int64) == 0
    finite = (
        np.isfinite(np.asarray(output["y"], dtype=float))
        & np.isfinite(np.asarray(output["y_uncertainty"], dtype=float))
    )
    output["used_for_wavelength_fit"] = identified & good & finite
    return output


# -----------------------------------------------------------------------------
# FITS calibration product
# -----------------------------------------------------------------------------


def write_simlc_lsf_fits(
    lsf: SimLCLSF,
    filename: str | Path,
    *,
    ccd: int | str | None = None,
    mjd_mid: float = np.nan,
    source_peak_file: str | Path | None = None,
    overwrite: bool = True,
) -> None:
    """Write an order-dependent SimLC eLSF calibration product."""

    header = fits.Header()
    header["PRODUCT"] = "SIMLC_LSF"
    header["LSFTYPE"] = "EFFECTIVE"
    header["PIXCONV"] = True
    if ccd is not None:
        header["CCD"] = str(ccd)
    if np.isfinite(mjd_mid):
        header["MJD-MID"] = float(mjd_mid)
    if source_peak_file is not None:
        header["PEAKFILE"] = Path(source_peak_file).name

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.BinTableHDU(lsf.metadata, name="ORDERS"),
            fits.ImageHDU(np.asarray(lsf.offset, dtype=np.float64), name="OFFSET"),
            fits.ImageHDU(np.asarray(lsf.profile, dtype=np.float32), name="PROFILE"),
        ]
    )
    hdul.writeto(filename, overwrite=overwrite)


def read_simlc_lsf_fits(filename: str | Path) -> tuple[SimLCLSF, fits.Header]:
    """Read a SimLC eLSF calibration product."""

    with fits.open(filename) as hdul:
        header = hdul[0].header.copy()
        metadata = Table(hdul["ORDERS"].data)
        offset = np.asarray(hdul["OFFSET"].data, dtype=float)
        profile = np.asarray(hdul["PROFILE"].data, dtype=float)
    orders = np.asarray(metadata["order"], dtype=int)
    return SimLCLSF(orders, offset, profile, metadata), header


# -----------------------------------------------------------------------------
# Routine QA plots
# -----------------------------------------------------------------------------


def plot_simlc_lsf_qa(
    lsf: SimLCLSF,
    peak_table: Table | None = None,
    *,
    filename: str | Path | None = None,
):
    """Compact four-panel QA summary for one SimLC LSF calibration."""

    order = np.asarray(lsf.metadata["order"], dtype=int)
    direct = np.asarray(lsf.metadata["source_order"], dtype=int) == order

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    ax = axes[0, 0]
    ax.plot(order[direct], np.asarray(lsf.metadata["moffat_alpha"])[direct], "o-", ms=3)
    ax.set(xlabel="Echelle order", ylabel=r"Moffat $\alpha$ [pixel]", title="Parametric LSF scale")

    ax = axes[0, 1]
    ax.plot(order[direct], np.asarray(lsf.metadata["moffat_beta"])[direct], "o-", ms=3)
    ax.set(xlabel="Echelle order", ylabel=r"Moffat $\beta$", title="Wing shape")

    ax = axes[1, 0]
    improvement = np.asarray(lsf.metadata["empirical_improvement_percent"], dtype=float)
    ax.axhline(0.0, ls="--", lw=1)
    ax.plot(order[direct], improvement[direct], "o", ms=4)
    ax.set(
        xlabel="Echelle order",
        ylabel=r"Held-out $\chi^2$ improvement [%]",
        title="Empirical eLSF vs. Moffat",
    )

    ax = axes[1, 1]
    if peak_table is not None and "lsf_y_shift" in peak_table.colnames:
        phase = np.asarray(peak_table["y_initial"], dtype=float)
        phase = phase - np.round(phase)
        shift = np.asarray(peak_table["lsf_y_shift"], dtype=float)
        good = np.isfinite(phase) & np.isfinite(shift)
        ax.scatter(phase[good], shift[good], s=7, alpha=0.25)
        ax.axhline(0.0, ls="--", lw=1)
        ax.set(
            xlabel="Initial pixel phase",
            ylabel=r"$y_{\rm LSF}-y_{\rm initial}$ [pixel]",
            title="Centroid correction",
        )
    else:
        n_lines = np.asarray(lsf.metadata["n_lines"], dtype=float)
        ax.plot(order, n_lines, "o-", ms=3)
        ax.set(
            xlabel="Echelle order",
            ylabel="LSF lines",
            title="Modes used for LSF inference",
        )

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=180)
    return fig


def plot_simlc_order_qa(
    counts: np.ndarray,
    orders: np.ndarray,
    peak_table: Table,
    lsf: SimLCLSF,
    order: int,
    *,
    variance: np.ndarray | None = None,
    maximum_lines: int = 15,
    filename: str | Path | None = None,
):
    """Overlay normalized modes and the adopted eLSF for one order."""

    lookup = _order_lookup(orders)
    q = np.flatnonzero(
        (np.asarray(peak_table["order"], dtype=int) == int(order))
        & (np.asarray(peak_table["comb_mode"], dtype=np.int64) >= 0)
        & np.isfinite(np.asarray(peak_table["y"], dtype=float))
    )
    q = _cap_evenly(q, maximum_lines)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for i in q:
        arr = _line_data(counts, variance, lookup, order, float(peak_table["y"][i]), 3)
        if arr is None:
            continue
        pixels, signal, _ = arr
        center = float(peak_table["y"][i])
        background = float(peak_table["background"][i])
        amplitude = float(peak_table["integrated_counts"][i])
        if not np.isfinite(amplitude) or amplitude <= 0:
            continue
        ax.plot(
            pixels - center,
            (signal - background) / amplitude,
            "o-",
            ms=3,
            lw=0.8,
            alpha=0.25,
        )

    ax.plot(lsf.offset, lsf.evaluate(order, lsf.offset), lw=2.2, label=lsf.model_name(order))
    ax.set(
        xlim=(-3.2, 3.2),
        xlabel="Pixel coordinate relative to fitted line centre",
        ylabel="Flux / integrated line counts",
        title=f"SimLC order {order}",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=180)
    return fig


def plot_simlc_lsf_stability(
    lsfs: list[SimLCLSF],
    *,
    labels: list[str] | None = None,
    filename: str | Path | None = None,
):
    """Compare repeated SimLC eLSFs to the first exposure as a stability QA."""

    if len(lsfs) < 2:
        raise ValueError("At least two SimLC LSF products are required")
    if labels is None:
        labels = [f"exposure {i}" for i in range(len(lsfs))]
    if len(labels) != len(lsfs):
        raise ValueError("labels must match lsfs")

    reference = lsfs[0]
    common_orders = set(reference.orders)
    for lsf in lsfs[1:]:
        common_orders &= set(lsf.orders)
    common_orders = np.array(sorted(common_orders), dtype=int)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for label, lsf in zip(labels[1:], lsfs[1:]):
        difference = []
        for order in common_orders:
            ref = reference.evaluate(order, reference.offset)
            current = lsf.evaluate(order, reference.offset)
            scale = np.nanmax(ref)
            difference.append(
                100.0 * np.sqrt(np.nanmean((current - ref) ** 2)) / scale
            )
        ax.plot(common_orders, difference, "o-", ms=3, label=label)

    ax.set(
        xlabel="Echelle order",
        ylabel="RMS eLSF difference [% reference peak]",
        title=f"SimLC LSF stability relative to {labels[0]}",
    )
    ax.axhline(0.0, lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=180)
    return fig


# -----------------------------------------------------------------------------
# Laser-comb frequencies, mode identification, and high-level measurement
# -----------------------------------------------------------------------------

SPEED_OF_LIGHT_MPS = 299_792_458.0
SPEED_OF_LIGHT_NM_S = SPEED_OF_LIGHT_MPS * 1e9
SPEED_OF_LIGHT_ANGSTROM_GHZ = SPEED_OF_LIGHT_MPS * 10.0


def lasercomb_wavelength_from_numbers(
    n,
    repeat_frequency_ghz=25.00000000,
    offset_frequency_ghz=9.56000000000,
):
    """Return comb-mode vacuum wavelength in Angstrom."""
    n = np.asarray(n)
    frequency_ghz = n * repeat_frequency_ghz + offset_frequency_ghz
    return SPEED_OF_LIGHT_ANGSTROM_GHZ / frequency_ghz


def lasercomb_numbers_from_wavelength(
    wavelength_angstrom,
    repeat_frequency_ghz=25.00000000,
    offset_frequency_ghz=9.56000000000,
):
    """Return the (generally non-integer) comb mode implied by wavelength."""
    wavelength_angstrom = np.asarray(wavelength_angstrom, dtype=float)
    return (
        SPEED_OF_LIGHT_ANGSTROM_GHZ / wavelength_angstrom
        - offset_frequency_ghz
    ) / repeat_frequency_ghz


def make_comb_reference_table(
    wavelength_min_nm,
    wavelength_max_nm,
    *,
    repetition_rate_hz=25.0e9,
    offset_frequency_hz=9.56e9,
):
    """Generate exact comb frequencies/wavelengths spanning an interval."""
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
    """Create exact comb references from explicitly supplied frequencies."""
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
    detector_shift_y,
    y_bounds,
    comb_reference_table=None,
    repetition_rate_hz=25.0e9,
    offset_frequency_hz=9.56e9,
    initial_matching_radius_pixel=2.5,
    final_matching_radius_pixel=1.2,
):
    """Assign exact LFC modes using a detector-shifted bootstrap solution."""
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
            reference = reference[(wave >= np.nanmin(edge_wave)) & (wave <= np.nanmax(edge_wave))]
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


def measure_simlc_lines(
    counts,
    orders,
    *,
    reference_wavelength_function,
    detector_shift_y=0.0,
    y_bounds=(0.0, 4111.0),
    variance=None,
    ccd=None,
    exposure_index=-1,
    mjd_mid=np.nan,
    fibre=-1,
    trace_x_function=None,
    peak_config=None,
    lsf_config=None,
    comb_reference_table=None,
    repetition_rate_hz=25.0e9,
    offset_frequency_hz=9.56e9,
):
    """Measure SimLC lines, infer the LSF, and return final comb centroids.

    The first pixel-integrated Gaussian pass is used only for detection and mode
    assignment.  Identified modes are then remeasured with the validated
    order-dependent Moffat/empirical eLSF before wavelength fitting.
    """
    if peak_config is None:
        peak_config = CalibrationPeakConfig()
    if lsf_config is None:
        lsf_config = SimLCLSFConfig()
    lines = measure_calibration_peaks(
        counts,
        orders,
        variance=variance,
        calibration_type="SimLC",
        ccd=ccd,
        exposure_index=exposure_index,
        mjd_mid=mjd_mid,
        fibre=fibre,
        trace_x_function=trace_x_function,
        config=peak_config,
    )
    if len(lines) == 0:
        return CalibrationLineSet(lines, "SimLC", np.nan, None)
    lines, shift = identify_simlc_lines(
        lines,
        reference_wavelength_function=reference_wavelength_function,
        detector_shift_y=detector_shift_y,
        y_bounds=y_bounds,
        comb_reference_table=comb_reference_table,
        repetition_rate_hz=repetition_rate_hz,
        offset_frequency_hz=offset_frequency_hz,
    )
    lsf = fit_simlc_lsf(
        counts,
        orders,
        lines,
        variance=variance,
        config=lsf_config,
    )
    lines = refit_simlc_peaks(
        counts,
        orders,
        lines,
        lsf,
        variance=variance,
        config=lsf_config,
    )
    if "fwhm_pixel" not in lines.colnames:
        lines["fwhm_pixel"] = np.asarray(lines["fwhm"], dtype=float)
    if "fwhm_uncertainty_pixel" not in lines.colnames:
        lines["fwhm_uncertainty_pixel"] = np.asarray(lines["fwhm_uncertainty"], dtype=float)
    return CalibrationLineSet(lines, "SimLC", shift, lsf)


# Backwards-compatible name while notebooks are migrated.
identify_simlc_peaks = identify_simlc_lines
