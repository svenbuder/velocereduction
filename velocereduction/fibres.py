"""Fit and persist compact Flat-derived fibre geometry.

The saved data product stores only the smooth model coefficients and one
fibre-specific offset per illuminated science/sky fibre. Evaluated 4112-row
centres/widths are reconstructed only when spectra are extracted.
"""
from pathlib import Path
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import least_squares
from scipy.special import ndtr

from .constants import FIBRE_COMPONENTS, FIBRE_SLOTS, N_DISPERSION
from .models import FibreGeometry, OrderMatrix
from . import diagnostics, utils

logger = logging.getLogger(__name__)


def initial_fibre_parameters(ccd, order):
    """Approximate Veloce fibre spacing/width used only to initialise fits."""
    ccd, order = str(ccd), int(order)
    if ccd == "1":
        separation = np.interp(order, [140, 165], [2.50, 2.40])
        sigma = 1.20
    elif ccd == "2":
        separation = np.interp(order, [104, 140], [2.425, 2.20])
        sigma = 0.80
    elif ccd == "3":
        separation = 2.15
        sigma = 0.55
    else:
        raise ValueError(f"Unknown CCD: {ccd}")
    return float(separation), float(sigma)


def integrated_gaussian_matrix(x, centres, sigma, normalise=True):
    x, centres = np.asarray(x, float)[:, None], np.asarray(centres, float)[None, :]
    p = ndtr((x + 0.5 - centres) / float(sigma)) - ndtr((x - 0.5 - centres) / float(sigma))
    if normalise:
        total = p.sum(axis=0)
        p[:, total > 0] /= total[total > 0]
    return p


def _linear_profile_fit(profile, x, centres, sigma):
    profiles = integrated_gaussian_matrix(x, centres, sigma)
    design = np.column_stack((profiles, np.ones(len(x))))
    good = np.isfinite(profile)
    coeff, *_ = np.linalg.lstsq(design[good], np.asarray(profile)[good], rcond=None)
    return coeff, design @ coeff


def _fit_regular_geometry(profile, x, x0, separation0, sigma0):
    def residual(p):
        centres = p[0] + p[1] * FIBRE_SLOTS
        _, model = _linear_profile_fit(profile, x, centres, p[2])
        return model - profile

    fit = least_squares(
        residual, [x0, separation0, sigma0],
        bounds=([x0 - 1.5, separation0 - 0.35, max(0.30, sigma0 - 0.45)],
                [x0 + 1.5, separation0 + 0.35, sigma0 + 0.60]),
        max_nfev=4000,
    )
    return fit.x


def fit_collapsed_fibre_profile(order_matrix, maximum_centre_shift=0.40, reference_geometry=None):
    """Fit one high-S/N collapsed profile to initialise all local fibre fits."""
    matrix = np.asarray(order_matrix.flux, float)
    profile = np.nanmedian(matrix, axis=0)
    x = np.asarray(order_matrix.relative_x, float)
    x00 = float(np.nanmedian(order_matrix.trace_offset))
    if reference_geometry is None:
        separation0, sigma0 = initial_fibre_parameters(order_matrix.ccd, order_matrix.order)
        reference_offsets = np.zeros(len(FIBRE_SLOTS))
    else:
        separation0 = float(reference_geometry.separation(reference_geometry.y_reference))
        sigma0 = float(reference_geometry.sigma(reference_geometry.y_reference))
        reference_offsets = np.asarray(reference_geometry.fibre_offsets, float)
    x0, separation, sigma = _fit_regular_geometry(profile, x, x00, separation0, sigma0)
    regular = x0 + separation * FIBRE_SLOTS

    def residual(offset_correction):
        centres = regular + reference_offsets + offset_correction
        _, model = _linear_profile_fit(profile, x, centres, sigma)
        return model - profile

    fit = least_squares(
        residual, np.zeros(len(FIBRE_SLOTS)),
        bounds=(-maximum_centre_shift, maximum_centre_shift), max_nfev=5000,
    )
    centres = regular + reference_offsets + fit.x
    separation, x0 = np.polyfit(FIBRE_SLOTS, centres, 1)
    offsets = centres - (x0 + separation * FIBRE_SLOTS)
    coeff, model = _linear_profile_fit(profile, x, centres, sigma)
    rms = float(np.sqrt(np.nanmean((profile - model) ** 2)))
    return {
        "profile": profile, "model": model, "x": x, "background": coeff[-1],
        "amplitudes": coeff[:-1], "sigma": float(sigma), "x0": float(x0),
        "separation": float(separation), "centres": centres,
        "fibre_offsets": offsets, "rms": rms,
        "trace_offset_median": float(np.nanmedian(order_matrix.trace_offset)),
    }


def _fit_local_profile(profile, x, trace_offset, collapsed):
    bundle0 = collapsed["x0"] - collapsed["trace_offset_median"]
    initial = np.array([
        trace_offset + bundle0,
        collapsed["separation"],
        collapsed["sigma"],
    ])
    offsets = collapsed["fibre_offsets"]

    def residual(p):
        centres = p[0] + p[1] * FIBRE_SLOTS + offsets
        _, model = _linear_profile_fit(profile, x, centres, p[2])
        return model - profile

    lower = [initial[0] - 0.6, initial[1] - 0.25, max(0.30, initial[2] - 0.35)]
    upper = [initial[0] + 0.6, initial[1] + 0.25, initial[2] + 0.35]
    fit = least_squares(residual, initial, bounds=(lower, upper), max_nfev=1500)
    return fit.x, float(np.sqrt(np.nanmean(residual(fit.x) ** 2)))


def _normalised_coordinate(y, y_reference, y_scale):
    return (np.asarray(y, float) - y_reference) / y_scale


def _robust_polynomial(y, values, degree, y_reference, y_scale, clip=4.0, iterations=4):
    y, values = np.asarray(y, float), np.asarray(values, float)
    u = _normalised_coordinate(y, y_reference, y_scale)
    good = np.isfinite(u) & np.isfinite(values)
    if good.sum() == 0:
        raise RuntimeError("No valid samples for fibre-geometry polynomial")
    degree = min(int(degree), good.sum() - 1)
    use = good.copy()
    for _ in range(iterations):
        coeff = np.polynomial.polynomial.polyfit(u[use], values[use], degree)
        residual = values - np.polynomial.polynomial.polyval(u, coeff)
        sigma = utils.robust_sigma(residual[use])
        if not np.isfinite(sigma) or sigma == 0:
            break
        new = good & (np.abs(residual - np.nanmedian(residual[use])) < clip * sigma)
        if new.sum() <= degree or np.array_equal(new, use):
            break
        use = new
    coeff = np.polynomial.polynomial.polyfit(u[use], values[use], degree)
    residual = values[use] - np.polynomial.polynomial.polyval(u[use], coeff)
    return coeff, float(np.sqrt(np.nanmean(residual ** 2))), use


def fit_fibre_geometry(order_matrix, sample_step=16, sample_half_width=4, degree=3, reference_geometry=None):
    """Measure sparse local profiles and return a compact smooth FibreGeometry."""
    if not isinstance(order_matrix, OrderMatrix):
        raise TypeError("fit_fibre_geometry expects an OrderMatrix")
    matrix = np.asarray(order_matrix.flux, float)
    collapsed = fit_collapsed_fibre_profile(order_matrix, reference_geometry=reference_geometry)
    n_dispersion = matrix.shape[0]
    sampled_y = np.arange(sample_half_width, n_dispersion - sample_half_width, sample_step, dtype=int)
    bundles = np.full(sampled_y.size, np.nan)
    separations = np.full(sampled_y.size, np.nan)
    sigmas = np.full(sampled_y.size, np.nan)
    local_rms = np.full(sampled_y.size, np.nan)

    for j, y in enumerate(sampled_y):
        profile = np.nanmedian(matrix[y - sample_half_width:y + sample_half_width + 1], axis=0)
        dynamic = np.nanpercentile(profile, 95) - np.nanpercentile(profile, 5)
        if np.isfinite(profile).sum() < len(FIBRE_COMPONENTS) + 2 or not np.isfinite(dynamic) or dynamic <= 0:
            continue
        try:
            p, local_rms[j] = _fit_local_profile(profile, order_matrix.relative_x, order_matrix.trace_offset[y], collapsed)
            bundles[j] = p[0] - order_matrix.trace_offset[y]
            separations[j] = p[1]
            sigmas[j] = p[2]
        except (ValueError, np.linalg.LinAlgError):
            continue

    y_reference = 0.5 * (n_dispersion - 1)
    y_scale = y_reference
    bundle_coeff, bundle_rms, use_bundle = _robust_polynomial(sampled_y, bundles, degree, y_reference, y_scale)
    separation_coeff, separation_rms, use_sep = _robust_polynomial(sampled_y, separations, degree, y_reference, y_scale)
    sigma_coeff, sigma_rms, use_sigma = _robust_polynomial(sampled_y, sigmas, degree, y_reference, y_scale)
    use = use_bundle & use_sep & use_sigma

    geometry = FibreGeometry(
        ccd=order_matrix.ccd,
        order=order_matrix.order,
        components=FIBRE_COMPONENTS,
        slots=FIBRE_SLOTS.copy(),
        fibre_offsets=np.asarray(collapsed["fibre_offsets"], float),
        bundle_coefficients=np.asarray(bundle_coeff, float),
        separation_coefficients=np.asarray(separation_coeff, float),
        sigma_coefficients=np.asarray(sigma_coeff, float),
        y_reference=y_reference,
        y_scale=y_scale,
        fit_rms=float(np.nanmedian(local_rms[use])) if np.any(use) else np.nan,
        fit_npoints=int(np.sum(use)),
        sampled_y=sampled_y,
        sampled_bundle=bundles,
        sampled_separation=separations,
        sampled_sigma=sigmas,
    )

    centres, sigma, separation, _ = geometry.evaluate(n_dispersion, order_matrix.trace_offset)
    if not (np.all(np.isfinite(centres)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(separation))):
        raise RuntimeError(f"{geometry.name}: compact fibre model does not cover the full order")
    logger.debug(
        "%s fibre geometry: %d/%d samples; separation %.3f--%.3f px; sigma %.3f--%.3f px",
        geometry.name, geometry.fit_npoints, len(sampled_y),
        np.nanmin(separation), np.nanmax(separation), np.nanmin(sigma), np.nanmax(sigma),
    )
    return geometry


def fit_fibre_geometries(flat_order_matrices, config, reference_geometries=None):
    """Fit all orders, using compact reference geometry as the starting model when available."""
    reference_geometries = reference_geometries or {}
    geometries = {}
    for name, matrix in flat_order_matrices.items():
        geometries[name] = fit_fibre_geometry(
            matrix,
            sample_step=config.fibre_sample_step,
            sample_half_width=config.fibre_sample_half_width,
            degree=config.fibre_geometry_degree,
            reference_geometry=reference_geometries.get(name),
        )
    return geometries


def fibre_geometry_tables(geometries):
    """Return compact order-model and per-fibre-offset tables."""
    values = list(geometries.values()) if isinstance(geometries, dict) else list(geometries)
    max_degree = max(g.degree for g in values)
    order_rows, fibre_rows = [], []
    for g in values:
        row = {
            "CCD": np.int16(g.ccd),
            "ORDER": np.int16(g.order),
            "DEGREE": np.int8(g.degree),
            "Y_REFERENCE": np.float64(g.y_reference),
            "Y_SCALE": np.float64(g.y_scale),
            "FIT_RMS": np.float32(g.fit_rms),
            "FIT_NPOINTS": np.int16(g.fit_npoints),
        }
        for prefix, coeff in (
            ("BUNDLE", g.bundle_coefficients),
            ("SEPARATION", g.separation_coefficients),
            ("SIGMA", g.sigma_coefficients),
        ):
            for k in range(max_degree + 1):
                row[f"{prefix}_C{k}"] = np.float64(coeff[k]) if k < len(coeff) else np.float64(np.nan)
        order_rows.append(row)
        for component, slot, offset in zip(g.components, g.slots, g.fibre_offsets):
            fibre_rows.append({
                "CCD": np.int16(g.ccd),
                "ORDER": np.int16(g.order),
                "FIBRE": str(component),
                "SLOT": np.int8(slot),
                "OFFSET": np.float64(offset),
            })
    return Table(rows=order_rows), Table(rows=fibre_rows)


def save_fibre_geometry(filename, geometries, config=None):
    """Write only compact functional-form parameters; never 4112-row evaluations."""
    order_table, fibre_table = fibre_geometry_tables(geometries)
    primary = fits.PrimaryHDU()
    primary.header["PRODUCT"] = "FIBRE_GEOMETRY"
    primary.header["MODEL"] = "centre=trace_offset+bundle+separation*slot+offset"
    primary.header["COORD"] = "u=(y-Y_REFERENCE)/Y_SCALE"
    primary.header["NORDER"] = len(order_table)
    if config is not None:
        primary.header["NIGHT"] = config.night
        primary.header["REFNIGHT"] = config.reference_night
    fits.HDUList([
        primary,
        fits.BinTableHDU(order_table, name="ORDER_MODEL"),
        fits.BinTableHDU(fibre_table, name="FIBRE_OFFSETS"),
    ]).writeto(filename, overwrite=True)
    return Path(filename)


def load_fibre_geometry(filename):
    """Read the compact two-table FibreGeometry format."""
    with fits.open(filename, memmap=False) as hdul:
        if "ORDER_MODEL" not in hdul or "FIBRE_OFFSETS" not in hdul:
            raise ValueError(
                "This is the old 4112-row fibre_geometry format. Refit the Flat and save the compact model instead."
            )
        orders = Table(hdul["ORDER_MODEL"].data)
        offsets = Table(hdul["FIBRE_OFFSETS"].data)

    geometries = {}
    for row in orders:
        ccd, order, degree = str(row["CCD"]), int(row["ORDER"]), int(row["DEGREE"])
        use = (np.asarray(offsets["CCD"]) == int(ccd)) & (np.asarray(offsets["ORDER"]) == order)
        fibre_rows = offsets[use]
        components = tuple(str(value).strip() for value in fibre_rows["FIBRE"])
        g = FibreGeometry(
            ccd=ccd, order=order, components=components,
            slots=np.asarray(fibre_rows["SLOT"], float),
            fibre_offsets=np.asarray(fibre_rows["OFFSET"], float),
            bundle_coefficients=np.array([row[f"BUNDLE_C{k}"] for k in range(degree + 1)], float),
            separation_coefficients=np.array([row[f"SEPARATION_C{k}"] for k in range(degree + 1)], float),
            sigma_coefficients=np.array([row[f"SIGMA_C{k}"] for k in range(degree + 1)], float),
            y_reference=float(row["Y_REFERENCE"]), y_scale=float(row["Y_SCALE"]),
            fit_rms=float(row["FIT_RMS"]), fit_npoints=int(row["FIT_NPOINTS"]),
        )
        geometries[g.name] = g
    return geometries


def summarise_fibre_geometry(geometries, n_dispersion=N_DISPERSION):
    """Return one QA row per order with central/range separation and sigma."""
    rows = []
    values = geometries.values() if isinstance(geometries, dict) else geometries
    y = np.arange(n_dispersion)
    for g in values:
        separation, sigma = g.separation(y), g.sigma(y)
        expected_sep, expected_sigma = initial_fibre_parameters(g.ccd, g.order)
        rows.append({
            "ccd": np.int16(g.ccd),
            "order": np.int16(g.order),
            "separation_mid": np.float32(g.separation(g.y_reference)),
            "separation_min": np.float32(np.nanmin(separation)),
            "separation_max": np.float32(np.nanmax(separation)),
            "sigma_mid": np.float32(g.sigma(g.y_reference)),
            "sigma_min": np.float32(np.nanmin(sigma)),
            "sigma_max": np.float32(np.nanmax(sigma)),
            "expected_separation": np.float32(expected_sep),
            "expected_sigma": np.float32(expected_sigma),
            "fit_rms": np.float32(g.fit_rms),
            "fit_npoints": np.int16(g.fit_npoints),
        })
    return Table(rows=rows)


def save_fibre_diagnostics(geometries, flat_order_matrices, config, paths):
    if config.diagnostics == "none":
        return
    diagnostics.plot_fibre_geometry_summary(
        geometries, paths.figures / f"fibre_geometry_{config.night}.png"
    )
    diagnostics.plot_fibre_profile_summary(
        geometries, flat_order_matrices, paths.figures / f"fibre_profiles_{config.night}.png"
    )
    if config.diagnostics == "full":
        for name, geometry in geometries.items():
            diagnostics.plot_fibre_geometry_order(
                geometry, flat_order_matrices[name],
                paths.debug / "fibre_geometry" / f"{name}.png",
            )
