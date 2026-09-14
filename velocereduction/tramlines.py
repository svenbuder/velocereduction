import logging
import numpy as np
from astropy.table import Table
from numpy.polynomial import Polynomial
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .constants import TRACE_DEGREE
from . import diagnostics, observations, detector, utils

logger = logging.getLogger(__name__)

def order_name(row):
    value = row["order_name"]
    return value.decode() if isinstance(value, bytes) else str(value)

def ccd_from_order_name(name):
    return str(name).split("_")[1]

def physical_order(name):
    return int(str(name).split("_")[-1])

def extract_trace(image, trace, half_window):
    image, trace = np.asarray(image), np.asarray(trace, float)
    centres = np.rint(trace).astype(int)
    out = np.full((image.shape[0], 2 * half_window + 1), np.nan, np.float32)
    for x, centre in enumerate(centres):
        y0, y1 = centre - half_window, centre + half_window + 1
        s0, s1 = max(0, y0), min(image.shape[1], y1)
        if s0 < s1:
            d0 = s0 - y0
            out[x, d0:d0 + s1 - s0] = image[x, s0:s1]
    return out, centres


def trace_from_row(row, n_dispersion):
    coeffs = np.array([float(row[f"tramline_coeff_{i}"]) for i in range(TRACE_DEGREE + 1)])
    return Polynomial(coeffs)(np.arange(n_dispersion, dtype=float))


def extract_order_matrix(image, row, return_trace_offset=False):
    trace = trace_from_row(row, image.shape[0])
    half = int(row["extraction_half_window"])
    matrix, centres = extract_trace(image, trace, half)
    m = np.arange(-half, half + 1, dtype=float)
    return (matrix, m, trace - centres) if return_trace_offset else (matrix, m)


def collapsed_profile(matrix):
    matrix = np.asarray(matrix, float)
    valid = np.isfinite(matrix)
    n = valid.sum(axis=0)
    profile = np.full(matrix.shape[1], np.nan)
    good = n > 0
    if np.any(good):
        profile[good] = np.nansum(matrix, axis=0)[good] / n[good] * np.nanmedian(n[good])
    return profile


def _find_trough(profile, m, expected, radius=6, smooth=1.5):
    use = np.isfinite(profile) & (np.abs(m - expected) <= radius)
    if use.sum() < 5:
        return np.nan
    x, y = m[use], gaussian_filter1d(profile[use], smooth)
    dynamic = np.nanpercentile(y, 95) - np.nanpercentile(y, 5)
    peaks, _ = find_peaks(-y, prominence=max(0.05 * dynamic, 0))
    if not len(peaks):
        return np.nan
    i = peaks[np.argmin(np.abs(x[peaks] - expected))]
    if 0 < i < len(x) - 1:
        a, b, _ = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
        vertex = -b / (2 * a) if a > 0 else np.nan
        if np.isfinite(vertex) and x[i - 1] <= vertex <= x[i + 1]:
            return float(vertex)
    return float(x[i])


def _robust_polyfit(x, y, degree=TRACE_DEGREE, clip=4, iterations=5):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() <= degree:
        raise ValueError("Insufficient points for trace fit")
    use = good.copy()
    for _ in range(iterations):
        coeffs = np.polynomial.polynomial.polyfit(x[use], y[use], degree)
        residual = y - np.polynomial.polynomial.polyval(x, coeffs)
        sigma = utils.robust_sigma(residual[use])
        if not np.isfinite(sigma) or sigma == 0:
            break
        new = good & (np.abs(residual - np.nanmedian(residual[use])) < clip * sigma)
        if new.sum() <= degree or np.array_equal(new, use):
            break
        use = new
    return np.polynomial.polynomial.polyfit(x[use], y[use], degree)


def _fit_flat_order(image, row, dx, dy, step=8):
    half, x = int(row["extraction_half_window"]), np.arange(image.shape[0], dtype=float)
    reference = np.array([float(row[f"tramline_coeff_{i}"]) for i in range(TRACE_DEGREE + 1)])
    initial = Polynomial(utils.shifted_coefficients(reference, dx, dy))(x)
    matrix, centres = extract_trace(image, initial, half)
    m = np.arange(-half, half + 1, dtype=float)
    left0 = 0.5 * (float(row["Sky_1_end"]) + float(row["Science_begin"]))
    right0 = 0.5 * (float(row["Science_end"]) + float(row["Sky_2_begin"]))
    xs, ys = [], []
    for x0 in range(2, len(matrix) - 2, step):
        profile = np.nanmedian(matrix[x0 - 2:x0 + 3], axis=0)
        offset = initial[x0] - centres[x0]
        left, right = _find_trough(profile, m, left0 + offset), _find_trough(profile, m, right0 + offset)
        if np.isfinite(left) and np.isfinite(right) and right > left:
            xs.append(x0)
            ys.append(centres[x0] + 0.5 * (left + right))
    enough = len(xs) > TRACE_DEGREE + 3
    coeffs = _robust_polyfit(xs, ys) if enough else utils.shifted_coefficients(reference, dx, dy)
    for i, value in enumerate(coeffs):
        row[f"tramline_coeff_{i}"] = value
    if "trace_npoints" in row.colnames:
        row["trace_npoints"] = len(xs)
    if "trace_rms" in row.colnames:
        residual = np.asarray(ys) - np.polynomial.polynomial.polyval(np.asarray(xs), coeffs) if len(xs) else np.array([])
        row["trace_rms"] = float(np.sqrt(np.nanmean(residual ** 2))) if len(residual) else np.nan
    if not enough:
        logger.debug("%s: only %d gap measurements; using shifted reference trace", order_name(row), len(xs))

    final, m, offset = extract_order_matrix(image, row, True)
    profile = collapsed_profile(final)
    left = _find_trough(profile, m, left0 + np.nanmedian(offset))
    right = _find_trough(profile, m, right0 + np.nanmedian(offset))
    if np.isfinite(left) and np.isfinite(right):
        row["Sky_1_end"] = row["Science_begin"] = left
        row["Science_end"] = row["Sky_2_begin"] = right
    return row


def _find_bright_interval(profile, m, begin, end):
    centre, width = 0.5 * (begin + end), max(abs(end - begin), 1)
    use = np.isfinite(profile) & (m >= centre - width - 6) & (m <= centre + width + 6)
    if use.sum() < 5:
        return begin, end
    x, y = m[use], gaussian_filter1d(profile[use], 1.2)
    dark, bright = np.nanpercentile(y, [10, 90])
    on = y > dark + 0.2 * (bright - dark)
    if not np.any(on):
        return begin, end
    groups = np.split(np.where(on)[0], np.where(np.diff(np.where(on)[0]) > 1)[0] + 1)
    group = min(groups, key=lambda g: abs(np.nanmean(x[g]) - centre))
    return float(x[group[0]] - 0.5), float(x[group[-1]] + 0.5)


def _representative_image(reduction_input, kind, ccd, config):
    rows = observations.select(reduction_input, kind, ccd)
    return None if not len(rows) else detector.preprocess_image(rows[0][f"file_ccd{ccd}"], ccd, config).image


def fit_nightly_tramlines(reduction_input, master_flat, detector_shifts, config, paths):
    filename = paths.flat / "tramlines.fits"
    calibration_images = None
    if filename.exists() and not config.overwrite:
        table = Table.read(filename)
        logger.info("Loaded cached nightly tramlines from %s", filename)
        return table
    reference_file = paths.repository / "velocereduction" / "veloce_reference_data" / "tramline_reference_001122.fits"
    table = Table.read(reference_file).copy()
    if "trace_rms" not in table.colnames:
        table["trace_rms"] = np.full(len(table), np.nan)
    if "trace_npoints" not in table.colnames:
        table["trace_npoints"] = np.zeros(len(table), int)
    calibration_images = {
        (kind, ccd): _representative_image(reduction_input, kind, ccd, config)
        for kind in ("SimTh", "SimLC") for ccd in ("1", "2", "3")
    }
    for i in range(len(table)):
        name = order_name(table[i])
        ccd = ccd_from_order_name(name)
        dx, dy = detector_shift(detector_shifts, ccd)
        table[i] = _fit_flat_order(master_flat[f"ccd_{ccd}"], table[i], dx, dy)
        for kind in ("SimTh", "SimLC"):
            image = calibration_images[(kind, ccd)]
            if image is None or f"{kind}_begin" not in table.colnames:
                continue
            matrix, m = extract_order_matrix(image, table[i])
            begin, end = _find_bright_interval(
                collapsed_profile(matrix), m, float(table[i][f"{kind}_begin"]), float(table[i][f"{kind}_end"])
            )
            table[i][f"{kind}_begin"], table[i][f"{kind}_end"] = begin, end
    table.write(filename, overwrite=True)
    for ccd in ("1", "2", "3"):
        use = np.array([ccd_from_order_name(order_name(row)) == ccd for row in table])
        if not np.any(use):
            continue
        rms = np.asarray(table["trace_rms"], float)[use]
        widths = np.asarray(table["Science_end"], float)[use] - np.asarray(table["Science_begin"], float)[use]
        finite_rms = rms[np.isfinite(rms)]
        finite_widths = widths[np.isfinite(widths)]
        max_rms = (float(np.nanmax(rms)) if np.any(np.isfinite(rms)) else np.nan)
        median_rms = (float(np.median(finite_rms)) if finite_rms.size else np.nan)
        median_widths = (float(np.median(finite_widths)) if finite_widths.size else np.nan)

        logger.info(
            "CCD%s tramlines: %d orders; median trace RMS %.3f px; max %.3f px; median Science width %.2f px",
            ccd, int(use.sum()), median_rms, max_rms, median_widths,
        )
    if config.diagnostics != "none":
        diagnostics.save_tramline_diagnostics(master_flat, calibration_images, table, config, paths)
    logger.info("Fitted %d nightly tramlines", len(table))
    return table
