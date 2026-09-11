"""Trace echelle orders and convert DetectorFrames into OrderMatrices."""
from pathlib import Path
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .constants import TRACE_DEGREE, TRACE_HALF_WINDOW
from .models import OrderGeometry, OrderMatrix
from . import detector, diagnostics, observations, utils

logger = logging.getLogger(__name__)
REGIONS = ("SimTh", "Sky_1", "Science", "Sky_2", "SimLC")
CALIBRATION_HALF_WIDTH = 3.0


def order_name(geometry):
    return geometry.name if isinstance(geometry, OrderGeometry) else str(geometry)


def _legacy_row_to_geometry(row):
    name = row["order_name"].decode() if isinstance(row["order_name"], bytes) else str(row["order_name"])
    ccd, order = name.split("_")[1], int(name.split("_")[-1])
    regions = {
        region: (float(row[f"{region}_begin"]), float(row[f"{region}_end"]))
        for region in REGIONS if f"{region}_begin" in row.colnames
    }
    available = {
        "SimTh": bool(row["SimTh_available"]) if "SimTh_available" in row.colnames else True,
        "SimLC": bool(row["SimLC_available"]) if "SimLC_available" in row.colnames else True,
    }
    rms_name = "trace_rms" if "trace_rms" in row.colnames else "tramline_fit_rms"
    n_name = "trace_npoints" if "trace_npoints" in row.colnames else "tramline_fit_npoints"
    return OrderGeometry(
        ccd=ccd,
        order=order,
        trace_coefficients=np.array([float(row[f"tramline_coeff_{i}"]) for i in range(TRACE_DEGREE + 1)]),
        extraction_half_window=int(row["extraction_half_window"]),
        regions=regions,
        available=available,
        trace_rms=float(row[rms_name]) if rms_name in row.colnames else np.nan,
        trace_npoints=int(row[n_name]) if n_name in row.colnames else 0,
    )


def load_order_geometry(filename):
    """Read compact order geometry; legacy tramline tables are accepted."""
    filename = Path(filename)
    with fits.open(filename, memmap=False) as hdul:
        table = Table(hdul["ORDER_GEOMETRY"].data) if "ORDER_GEOMETRY" in hdul else Table.read(filename)
    if "order_name" in table.colnames:
        return [_legacy_row_to_geometry(row) for row in table]

    geometries = []
    for row in table:
        regions = {}
        for region in REGIONS:
            b, e = f"{region.upper()}_BEGIN", f"{region.upper()}_END"
            if b in table.colnames and e in table.colnames and np.isfinite(row[b]) and np.isfinite(row[e]):
                regions[region] = (float(row[b]), float(row[e]))
        available = {
            "SimTh": bool(row["SIMTH_USE"]) if "SIMTH_USE" in table.colnames else "SimTh" in regions,
            "SimLC": bool(row["SIMLC_USE"]) if "SIMLC_USE" in table.colnames else "SimLC" in regions,
        }
        geometries.append(OrderGeometry(
            ccd=str(row["CCD"]), order=int(row["ORDER"]),
            trace_coefficients=np.array([row[f"TRACE_C{i}"] for i in range(TRACE_DEGREE + 1)], float),
            extraction_half_window=int(row["HALF_WINDOW"]), regions=regions, available=available,
            trace_rms=float(row["TRACE_RMS"]), trace_npoints=int(row["TRACE_NPOINTS"]),
        ))
    return geometries


def order_geometry_table(geometries):
    """Return a human-readable one-row-per-order table."""
    rows = []
    for g in geometries:
        row = {
            "CCD": np.int16(g.ccd),
            "ORDER": np.int16(g.order),
            "HALF_WINDOW": np.int16(g.extraction_half_window),
            "TRACE_RMS": np.float32(g.trace_rms),
            "TRACE_NPOINTS": np.int16(g.trace_npoints),
        }
        for k, value in enumerate(g.trace_coefficients):
            row[f"TRACE_C{k}"] = np.float64(value)
        for region in REGIONS:
            begin, end = g.regions.get(region, (np.nan, np.nan))
            row[f"{region.upper()}_BEGIN"] = np.float32(begin)
            row[f"{region.upper()}_END"] = np.float32(end)
        row["SIMTH_USE"] = bool(g.available.get("SimTh", False))
        row["SIMLC_USE"] = bool(g.available.get("SimLC", False))
        rows.append(row)
    return Table(rows=rows)


def save_order_geometry(filename, geometries, config=None):
    """Write compact order geometry as one readable FITS table."""
    primary = fits.PrimaryHDU()
    primary.header["PRODUCT"] = "ORDER_GEOMETRY"
    primary.header["MODEL"] = "x(y)=sum TRACE_Ck*y**k"
    primary.header["NORDER"] = len(geometries)
    primary.header["CALSRC"] = ("SKYEDGE", "SimTh/SimLC centres anchored to outer sky edges")
    primary.header["CALHW"] = (CALIBRATION_HALF_WIDTH, "Calibration aperture half-width / pixel")
    if config is not None:
        primary.header["NIGHT"] = config.night
        primary.header["REFNIGHT"] = config.reference_night
    table = fits.BinTableHDU(order_geometry_table(geometries), name="ORDER_GEOMETRY")
    table.header["COMMENT"] = "Region coordinates are relative cross-dispersion pixel edges."
    fits.HDUList([primary, table]).writeto(filename, overwrite=True)
    return Path(filename)


def geometry_for_ccd(geometries, ccd):
    return [g for g in geometries if str(g.ccd) == str(ccd)]


def extract_trace(image, trace, half_window=TRACE_HALF_WINDOW):
    """Extract a curved trace into a rectangular (dispersion, cross-dispersion) matrix."""
    image, trace = np.asarray(image), np.asarray(trace, float)
    centres = np.rint(trace).astype(int)
    out = np.full((image.shape[0], 2 * half_window + 1), np.nan, dtype=image.dtype if np.issubdtype(image.dtype, np.floating) else float)
    for y, centre in enumerate(centres):
        x0, x1 = centre - half_window, centre + half_window + 1
        s0, s1 = max(0, x0), min(image.shape[1], x1)
        if s0 < s1:
            d0 = s0 - x0
            out[y, d0:d0 + s1 - s0] = image[y, s0:s1]
    return out, centres


def extract_order_matrix(frame, geometry):
    """Apply fixed OrderGeometry to a DetectorFrame and return an OrderMatrix."""
    trace = geometry.trace(frame.image.shape[0])
    half = geometry.extraction_half_window
    flux, centres = extract_trace(frame.image, trace, half)
    variance, _ = extract_trace(frame.variance, trace, half)
    quality, _ = extract_trace(frame.quality_mask, trace, half)
    return OrderMatrix(
        geometry=geometry,
        flux=np.asarray(flux, np.float32),
        variance=np.asarray(variance, np.float32),
        quality_mask=np.nan_to_num(quality, nan=0).astype(np.uint16),
        relative_x=np.arange(-half, half + 1, dtype=float),
        trace_offset=trace - centres,
    )


def extract_order_matrices(frame, geometries):
    """Return all OrderMatrices belonging to one CCD DetectorFrame."""
    return {g.name: extract_order_matrix(frame, g) for g in geometry_for_ccd(geometries, frame.ccd)}


def collapsed_profile(matrix):
    data = matrix.flux if isinstance(matrix, OrderMatrix) else np.asarray(matrix, float)
    return np.nanmedian(data, axis=0)


def _find_trough(profile, x, expected, radius=6, smooth=1.5):
    use = np.isfinite(profile) & (np.abs(x - expected) <= radius)
    if use.sum() < 5:
        return np.nan
    xx, yy = x[use], gaussian_filter1d(profile[use], smooth)
    dynamic = np.nanpercentile(yy, 95) - np.nanpercentile(yy, 5)
    peaks, _ = find_peaks(-yy, prominence=max(0.05 * dynamic, 0))
    if not len(peaks):
        return np.nan
    i = peaks[np.argmin(np.abs(xx[peaks] - expected))]
    if 0 < i < len(xx) - 1:
        a, b, _ = np.polyfit(xx[i - 1:i + 2], yy[i - 1:i + 2], 2)
        vertex = -b / (2 * a) if a > 0 else np.nan
        if np.isfinite(vertex) and xx[i - 1] <= vertex <= xx[i + 1]:
            return float(vertex)
    return float(xx[i])


def _robust_polyfit(x, values, degree=TRACE_DEGREE, clip=4.0, iterations=5):
    x, values = np.asarray(x, float), np.asarray(values, float)
    good = np.isfinite(x) & np.isfinite(values)
    if good.sum() <= degree:
        raise ValueError("Insufficient points for order-trace fit")
    use = good.copy()
    for _ in range(iterations):
        coeff = np.polynomial.polynomial.polyfit(x[use], values[use], degree)
        residual = values - np.polynomial.polynomial.polyval(x, coeff)
        sigma = utils.robust_sigma(residual[use])
        if not np.isfinite(sigma) or sigma == 0:
            break
        new = good & (np.abs(residual - np.nanmedian(residual[use])) < clip * sigma)
        if new.sum() <= degree or np.array_equal(new, use):
            break
        use = new
    coeff = np.polynomial.polynomial.polyfit(x[use], values[use], degree)
    residual = values[use] - np.polynomial.polynomial.polyval(x[use], coeff)
    return coeff, float(np.sqrt(np.nanmean(residual ** 2))), int(use.sum())


def _extract_array(image, geometry):
    trace = geometry.trace(image.shape[0])
    data, centres = extract_trace(image, trace, geometry.extraction_half_window)
    x = np.arange(-geometry.extraction_half_window, geometry.extraction_half_window + 1, dtype=float)
    return data, x, trace - centres


def _set_calibration_regions(geometry):
    """Anchor SimTh and SimLC apertures to the outer sky edges."""
    simth_centre = geometry.regions["Sky_1"][0]
    simlc_centre = geometry.regions["Sky_2"][1]
    geometry.regions["SimTh"] = (simth_centre - CALIBRATION_HALF_WIDTH, simth_centre + CALIBRATION_HALF_WIDTH)
    geometry.regions["SimLC"] = (simlc_centre - CALIBRATION_HALF_WIDTH, simlc_centre + CALIBRATION_HALF_WIDTH)


def _fit_flat_order(image, reference, dx, dy, step=8):
    """Refine one reference OrderGeometry on a current-night combined Flat."""
    y = np.arange(image.shape[0], dtype=float)
    shifted = utils.shifted_coefficients(reference.trace_coefficients, dx, dy)
    initial = OrderGeometry(
        reference.ccd, reference.order, shifted, reference.extraction_half_window,
        dict(reference.regions), dict(reference.available), reference.trace_rms, reference.trace_npoints,
    )
    matrix, x, offset = _extract_array(image, initial)
    left0 = 0.5 * (initial.region("Sky_1")[1] + initial.region("Science")[0])
    right0 = 0.5 * (initial.region("Science")[1] + initial.region("Sky_2")[0])
    ys, centres = [], []
    trace = initial.trace(image.shape[0])
    rounded = np.rint(trace)
    for y0 in range(2, len(matrix) - 2, step):
        profile = np.nanmedian(matrix[y0 - 2:y0 + 3], axis=0)
        left = _find_trough(profile, x, left0 + offset[y0])
        right = _find_trough(profile, x, right0 + offset[y0])
        if np.isfinite(left) and np.isfinite(right) and right > left:
            ys.append(y0)
            centres.append(rounded[y0] + 0.5 * (left + right))
    if len(ys) > TRACE_DEGREE + 3:
        coeff, rms, npoints = _robust_polyfit(ys, centres)
    else:
        coeff, rms, npoints = shifted, np.nan, len(ys)
        logger.warning("%s: only %d usable trace samples; using shifted reference", reference.name, len(ys))

    result = OrderGeometry(
        reference.ccd, reference.order, np.asarray(coeff), reference.extraction_half_window,
        dict(reference.regions), dict(reference.available), rms, npoints,
    )
    final, x, final_offset = _extract_array(image, result)
    profile = np.nanmedian(final, axis=0)
    left = _find_trough(profile, x, left0 + np.nanmedian(final_offset))
    right = _find_trough(profile, x, right0 + np.nanmedian(final_offset))
    if np.isfinite(left) and np.isfinite(right):
        result.regions["Sky_1"] = (result.regions["Sky_1"][0], left)
        result.regions["Science"] = (left, right)
        result.regions["Sky_2"] = (right, result.regions["Sky_2"][1])
    _set_calibration_regions(result)
    return result


def determine_order_geometry(reduction_input, combined_flats, detector_shifts, config, paths):
    """Refine the reference-night geometry and save one compact nightly table."""
    if paths.order_geometry.exists() and not config.overwrite:
        geometry = load_order_geometry(paths.order_geometry)
        logger.info("Loaded %d cached order geometries from %s", len(geometry), paths.order_geometry)
        return geometry

    reference_file = paths.reference_product("order_geometry", config.reference_night)
    if not reference_file.exists():
        legacy = paths.reference_data / f"tramline_reference_{config.reference_night}.fits"
        if legacy.exists():
            reference_file = legacy
        else:
            raise FileNotFoundError(
                f"No reference order geometry found. Copy the {config.reference_night} product to {reference_file}."
            )
    reference = load_order_geometry(reference_file)
    available = {
        (kind, ccd): len(observations.select(reduction_input, kind, ccd)) > 0
        for kind in ("SimTh", "SimLC") for ccd in ("1", "2", "3")
    }

    geometry = []
    for ref in reference:
        dx, dy = detector.detector_shift(detector_shifts, ref.ccd)
        current = _fit_flat_order(combined_flats[ref.ccd].image, ref, dx, dy)
        current.available["SimTh"] = available["SimTh", ref.ccd]
        current.available["SimLC"] = available["SimLC", ref.ccd]
        geometry.append(current)

    save_order_geometry(paths.order_geometry, geometry, config)
    for ccd in ("1", "2", "3"):
        subset = geometry_for_ccd(geometry, ccd)
        rms = np.array([g.trace_rms for g in subset], float)
        widths = np.array([np.diff(g.region("Science"))[0] for g in subset], float)
        logger.info(
            "CCD%s order geometry: %d orders; median trace RMS %.3f px; max %.3f px; median Science width %.2f px",
            ccd, len(subset), np.nanmedian(rms), np.nanmax(rms), np.nanmedian(widths),
        )
    if config.diagnostics != "none":
        diagnostics.plot_order_geometry_summary(geometry, paths.figures / f"order_geometry_{config.night}.png")
        diagnostics.plot_order_matrix_examples(combined_flats, geometry, paths.figures / f"order_matrices_{config.night}.png")
    return geometry
