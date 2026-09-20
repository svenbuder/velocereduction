"""Measure 1D spectra from OrderMatrices using fixed extraction geometry."""
from pathlib import Path
from dataclasses import dataclass
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.special import ndtr

from .constants import SCIENCE_FIBRES
from .models import DetectorFrame, ExtractionResult, ExtractedExposure, OrderMatrix
from . import detector, fibres, observations, orders

logger = logging.getLogger(__name__)


def aperture_weights(relative_x, begin, end):
    """Fractional pixel overlap with a continuous cross-dispersion aperture."""
    x = np.asarray(relative_x, float)
    return np.clip(np.minimum(x + 0.5, end) - np.maximum(x - 0.5, begin), 0, 1)


def extract_apertures(order_matrix, apertures):
    """Sum named apertures of an OrderMatrix with variance propagation."""
    if not isinstance(order_matrix, OrderMatrix):
        raise TypeError("extract_apertures expects an OrderMatrix")
    matrix = np.asarray(order_matrix.flux, float)
    variance = np.asarray(order_matrix.variance, float)
    flux = np.full((matrix.shape[0], len(apertures)), np.nan)
    var = np.full_like(flux, np.nan)
    for j, (name, begin, end) in enumerate(apertures):
        w = aperture_weights(order_matrix.relative_x, float(begin), float(end))[None, :]
        good = np.isfinite(matrix) & np.isfinite(variance) & (variance >= 0)
        flux[:, j] = np.nansum(np.where(good, matrix * w, np.nan), axis=1)
        var[:, j] = np.nansum(np.where(good, variance * w ** 2, np.nan), axis=1)
    return ExtractionResult(flux, var, tuple(a[0] for a in apertures), "summed")


def extract_summed_order(order_matrix, components=("Science", "Sky_1", "Sky_2")):
    """Extract named OrderGeometry regions by direct aperture summation."""
    apertures = []
    for component in components:
        begin, end = order_matrix.geometry.region(component)
        apertures.append((component, begin, end))
    return extract_apertures(order_matrix, apertures)


def integrated_gaussian_cube(relative_x, centres, sigma, normalise=True):
    """Pixel-integrated Gaussian profiles: (dispersion, cross-dispersion, fibre)."""
    x = np.asarray(relative_x, float)[None, :, None]
    centres = np.asarray(centres, float)[:, None, :]
    sigma = np.asarray(sigma, float)[:, None, None]
    p = ndtr((x + 0.5 - centres) / sigma) - ndtr((x - 0.5 - centres) / sigma)
    if normalise:
        total = p.sum(axis=1, keepdims=True)
        p = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
    return p


def extract_fibre_order(order_matrix, fibre_geometry, fit_background=True, return_covariance=False, batch_size=512):
    """Simultaneously deblend all Flat-defined fibre profiles in one order."""
    if not isinstance(order_matrix, OrderMatrix):
        raise TypeError("extract_fibre_order expects an OrderMatrix")
    if fibre_geometry.name != order_matrix.name:
        raise ValueError(f"Geometry mismatch: {fibre_geometry.name} != {order_matrix.name}")

    matrix = np.asarray(order_matrix.flux, float)
    variance = np.asarray(order_matrix.variance, float)
    n_dispersion, n_cross = matrix.shape
    centres, sigma, _, _ = fibre_geometry.evaluate(n_dispersion, order_matrix.trace_offset)
    n_fibre = len(fibre_geometry.components)
    flux = np.full((n_dispersion, n_fibre), np.nan)
    var = np.full_like(flux, np.nan)
    background = np.full(n_dispersion, np.nan) if fit_background else None
    covariance = np.full((n_dispersion, n_fibre, n_fibre), np.nan) if return_covariance else None

    for start in range(0, n_dispersion, batch_size):
        stop = min(start + batch_size, n_dispersion)
        slc = slice(start, stop)
        profiles = integrated_gaussian_cube(order_matrix.relative_x, centres[slc], sigma[slc])
        design = (
            np.concatenate((profiles, np.ones((stop - start, n_cross, 1))), axis=2)
            if fit_background else profiles
        )
        good = np.isfinite(matrix[slc]) & np.isfinite(variance[slc]) & (variance[slc] > 0)
        weight = np.divide(1.0, variance[slc], out=np.zeros_like(variance[slc]), where=good)
        data = np.where(good, matrix[slc], 0.0)
        normal = np.einsum("xyi,xy,xyj->xij", design, weight, design, optimize=True)
        rhs = np.einsum("xyi,xy,xy->xi", design, weight, data, optimize=True)
        local = np.where(good.sum(axis=1) >= design.shape[2])[0]
        if not len(local):
            continue
        matrices = normal[local]
        try:
            inverse = np.linalg.inv(matrices)
        except np.linalg.LinAlgError:
            inverse = np.stack([np.linalg.pinv(item, hermitian=True) for item in matrices])
        coeff = np.einsum("xij,xj->xi", inverse, rhs[local])
        idx = start + local
        flux[idx] = coeff[:, :n_fibre]
        var[idx] = np.diagonal(inverse[:, :n_fibre, :n_fibre], axis1=1, axis2=2)
        if fit_background:
            background[idx] = coeff[:, -1]
        if return_covariance:
            covariance[idx] = inverse[:, :n_fibre, :n_fibre]

    return ExtractionResult(
        flux=flux, variance=var, components=fibre_geometry.components,
        extraction_mode="fibre", covariance=covariance, background=background,
    )


def extract_order(order_matrix, extraction_mode, fibre_geometry=None, return_covariance=False):
    if extraction_mode == "summed":
        return extract_summed_order(order_matrix)
    if extraction_mode == "fibre":
        if fibre_geometry is None:
            raise ValueError("fibre_geometry is required for extraction_mode='fibre'")
        return extract_fibre_order(order_matrix, fibre_geometry, return_covariance=return_covariance)
    raise ValueError(f"Unknown extraction mode: {extraction_mode}")


def apply_response(result, response):
    """Apply a 1D post-extraction response and propagate variance/covariance."""
    response = np.asarray(response, float)
    if response.ndim == 1:
        response = response[:, None]
    if response.shape != result.flux.shape:
        if response.shape[0] != result.flux.shape[0] or response.shape[1] not in (1, result.flux.shape[1]):
            raise ValueError(f"Response shape {response.shape} is incompatible with flux {result.flux.shape}")
    good = np.isfinite(response) & (response > 0)
    flux = np.divide(result.flux, response, out=np.full_like(result.flux, np.nan), where=good)
    variance = np.divide(result.variance, response ** 2, out=np.full_like(result.variance, np.nan), where=good)
    covariance = result.covariance
    if covariance is not None:
        scale = response[:, :, None] * response[:, None, :]
        covariance = np.divide(covariance, scale, out=np.full_like(covariance, np.nan), where=np.isfinite(scale) & (scale > 0))
    return ExtractionResult(
        flux, variance, result.components, result.extraction_mode, covariance, result.background,
    )


def recombine_fibres(result, components=SCIENCE_FIBRES):
    """Recombine selected fibres, including off-diagonal covariance when available."""
    idx = result.component_indices(components)
    flux = np.nansum(result.flux[:, idx], axis=1)
    if result.covariance is None:
        variance = np.nansum(result.variance[:, idx], axis=1)
    else:
        covariance = result.covariance[:, idx][:, :, idx]
        variance = np.nansum(covariance, axis=(1, 2))
    return flux, variance


def _reference_pixel_overlaps(left, right, n_output):
    """Return output-pixel overlap fractions for one transformed source pixel."""
    lo, hi = sorted((float(left), float(right)))
    width = hi - lo
    if not np.isfinite(width) or width <= 0:
        return []
    first = max(0, int(np.floor(lo + 0.5)))
    last = min(int(n_output) - 1, int(np.floor(hi + 0.5)))
    result = []
    for pixel in range(first, last + 1):
        overlap = max(0.0, min(hi, pixel + 0.5) - max(lo, pixel - 0.5))
        if overlap > 0:
            result.append((pixel, overlap / width))
    return result


def recombine_fibres_on_reference_grid(
    result,
    fibre_displacement_model,
    *,
    ccd,
    order,
    components=SCIENCE_FIBRES,
    n_output=None,
):
    """Flux-conservingly align and recombine fibres on the summed-FibTh grid.

    The native extracted spectra remain untouched.  This is the single
    interpolation/resampling step used when a recombined spectrum is requested.
    Same-row inter-fibre extraction covariance is propagated when available.
    """
    idx = result.component_indices(components)
    fibres = tuple(int(value) for value in components)
    flux = np.asarray(result.flux[:, idx], float)
    variance = np.asarray(result.variance[:, idx], float)
    n_dispersion = flux.shape[0]
    n_output = n_dispersion if n_output is None else int(n_output)
    output_flux = np.zeros(n_output, dtype=float)
    output_variance = np.zeros(n_output, dtype=float)
    output_weight = np.zeros(n_output, dtype=float)

    edge = np.arange(n_dispersion + 1, dtype=float) - 0.5
    overlaps = [[None] * n_dispersion for _ in fibres]
    for local_fibre, fibre in enumerate(fibres):
        reference_edge = fibre_displacement_model.fibre_to_reference_y(
            str(ccd), fibre, edge, int(order)
        )
        for pixel in range(n_dispersion):
            overlaps[local_fibre][pixel] = _reference_pixel_overlaps(
                reference_edge[pixel], reference_edge[pixel + 1], n_output
            )
            if not np.isfinite(flux[pixel, local_fibre]):
                continue
            for output_pixel, weight in overlaps[local_fibre][pixel]:
                output_flux[output_pixel] += weight * flux[pixel, local_fibre]
                output_weight[output_pixel] += weight

    covariance = None
    if result.covariance is not None:
        covariance = np.asarray(result.covariance[:, idx][:, :, idx], float)

    for pixel in range(n_dispersion):
        per_output = {}
        for local_fibre in range(len(fibres)):
            if not np.isfinite(flux[pixel, local_fibre]):
                continue
            for output_pixel, weight in overlaps[local_fibre][pixel]:
                per_output.setdefault(
                    output_pixel, np.zeros(len(fibres), float)
                )[local_fibre] = weight
        for output_pixel, weight_vector in per_output.items():
            if covariance is None:
                good = np.isfinite(variance[pixel])
                output_variance[output_pixel] += np.nansum(
                    variance[pixel, good] * weight_vector[good] ** 2
                )
            else:
                local_covariance = np.where(
                    np.isfinite(covariance[pixel]), covariance[pixel], 0.0
                )
                output_variance[output_pixel] += float(
                    weight_vector @ local_covariance @ weight_vector
                )

    empty = output_weight == 0
    output_flux[empty] = np.nan
    output_variance[empty] = np.nan
    return output_flux, output_variance


def fibre_recombination_qa(summed, fibre, summed_component="Science"):
    """Compare direct summed flux with independently recombined science fibres."""
    summed_flux = summed.flux[:, summed.component_indices([summed_component])[0]]
    recombined, recombined_variance = recombine_fibres(fibre)
    ratio = np.divide(recombined, summed_flux, out=np.full_like(recombined, np.nan), where=np.isfinite(summed_flux) & (summed_flux != 0))
    finite = np.isfinite(ratio)
    median = np.nanmedian(ratio[finite]) if np.any(finite) else np.nan
    fractional = ratio / median - 1 if np.isfinite(median) and median != 0 else np.full_like(ratio, np.nan)
    robust_rms = 1.4826 * np.nanmedian(np.abs(fractional - np.nanmedian(fractional)))
    return {
        "summed_flux": summed_flux,
        "recombined_flux": recombined,
        "recombined_variance": recombined_variance,
        "ratio": ratio,
        "fractional_structure": fractional,
        "robust_rms_fractional_structure": float(robust_rms),
    }


PRIMARY_SIGNAL_REGION = {
    "Flat": "Science",
    "FibTh": "Science",
    "Science": "Science",
    "SimTh": "SimTh",
    "SimLC": "SimLC",
}


@dataclass(frozen=True)
class SignalCheck:
    """Compact CCD3 signal-presence diagnostic for one extraction region."""

    detected: bool
    region: str
    mode: str
    score: float
    median_snr: float
    p95_snr: float
    n_significant: int
    n_orders: int
    n_orders_detected: int


def measure_region_signal(
    frame,
    order_geometries,
    region,
    *,
    mode="line",
    continuum_snr=3.0,
    line_sigma=5.0,
    min_line_pixels=3,
    min_orders=3,
):
    """Measure whether an expected extraction region contains useful signal.

    ``mode='continuum'`` is intended for Science/Flat light and uses the median
    aperture S/N in each order. ``mode='line'`` subtracts a low-percentile
    baseline order by order and looks for repeated positive narrow-line signal,
    appropriate for FibTh, SimTh and SimLC. The returned metrics are retained so
    the thresholds can be tuned from real nights rather than treated as magic.
    """
    order_scores, all_snr = [], []
    n_significant = n_orders_detected = n_orders = 0

    for geometry in orders.geometry_for_ccd(order_geometries, str(frame.ccd)):
        if hasattr(geometry, "available") and region in geometry.available:
            if not geometry.available.get(region, False):
                continue
        try:
            order_matrix = orders.extract_order_matrix(frame, geometry)
            result = extract_summed_order(order_matrix, components=(region,))
        except (KeyError, ValueError):
            continue

        flux = np.asarray(result.flux[:, 0], float)
        variance = np.asarray(result.variance[:, 0], float)
        good = np.isfinite(flux) & np.isfinite(variance) & (variance > 0)
        if np.count_nonzero(good) < 10:
            continue

        n_orders += 1
        sigma = np.sqrt(variance[good])
        values = flux[good]
        if mode == "continuum":
            snr = values / sigma
            score = float(np.nanmedian(snr))
            detected = np.isfinite(score) and score >= continuum_snr
        elif mode == "line":
            baseline = float(np.nanpercentile(values, 20.0))
            snr = (values - baseline) / sigma
            significant = int(np.count_nonzero(snr >= line_sigma))
            n_significant += significant
            score = float(np.nanpercentile(snr, 99.0))
            detected = significant >= min_line_pixels and score >= line_sigma
        else:
            raise ValueError(f"Unknown signal-check mode: {mode}")

        all_snr.append(snr)
        order_scores.append(score)
        n_orders_detected += int(detected)

    if all_snr:
        snr = np.concatenate(all_snr)
        median_snr = float(np.nanmedian(snr))
        p95_snr = float(np.nanpercentile(snr, 95.0))
    else:
        median_snr = p95_snr = np.nan

    required_orders = min(min_orders, n_orders) if n_orders else min_orders
    detected = n_orders > 0 and n_orders_detected >= required_orders
    score = float(np.nanmedian(order_scores)) if order_scores else np.nan
    return SignalCheck(
        detected=bool(detected), region=region, mode=mode, score=score,
        median_snr=median_snr, p95_snr=p95_snr,
        n_significant=int(n_significant), n_orders=int(n_orders),
        n_orders_detected=int(n_orders_detected),
    )


def _signal_check(frame, order_geometries, region, mode, config):
    """Run a signal check using optional ReductionConfig threshold overrides."""
    return measure_region_signal(
        frame,
        order_geometries,
        region,
        mode=mode,
        continuum_snr=float(getattr(config, "signal_qa_continuum_snr", 3.0)),
        line_sigma=float(getattr(config, "signal_qa_line_sigma", 5.0)),
        min_line_pixels=int(getattr(config, "signal_qa_min_line_pixels", 3)),
        min_orders=int(getattr(config, "signal_qa_min_orders", 3)),
    )


def _set_issue(table, index, message):
    current = str(table["issue"][index]).strip()
    text = f"{current}; {message}" if current else message
    table["issue"][index] = text[:255]


def inspect_ccd3_signals(reduction_input, order_geometries, config, *, verbose=True, force=False):
    """Inspect CCD3 before calibration co-addition and record per-run signal QA.

    Primary signal is checked in Science for Flat/FibTh/Science, in SimTh for
    SimTh, and in SimLC for dedicated SimLC. Failed calibration exposures are
    excluded from later combinations. Science exposures are always retained
    unless already marked bad, but a weak primary signal is recorded as QA.

    SimLC presence is a separate image-based decision: the SimLC region is
    inspected for every Science exposure, every 15-s SimTh candidate, and every
    dedicated SimLC exposure. Recorded LC flags are not consulted.
    """
    table = reduction_input
    if "primary_signal_status" in table.colnames and not force:
        observations.update_calibration_blocks_from_qa(table)
        return table

    n = len(table)
    table["primary_signal_status"] = np.full(n, "not_checked", dtype="U16")
    table["primary_signal_score"] = np.full(n, np.nan, float)
    table["primary_signal_orders"] = np.zeros(n, np.int16)
    table["simlc_signal_status"] = np.full(n, "not_checked", dtype="U16")
    table["simlc_signal_score"] = np.full(n, np.nan, float)
    table["simlc_signal_orders"] = np.zeros(n, np.int16)
    table["simlc_signal"] = np.zeros(n, bool)
    base_use = np.asarray(
        table["calibration_use"] if "calibration_use" in table.colnames else table["use"],
        bool,
    )
    table["signal_use"] = base_use.copy()

    for i, row in enumerate(table):
        kind = str(row["type"])
        if kind not in PRIMARY_SIGNAL_REGION or not bool(row["use"]):
            continue
        if kind != "Science" and "calibration_use" in table.colnames and not bool(row["calibration_use"]):
            table["signal_use"][i] = False
            continue
        # Do not spend CCD3 QA time on calibration exposure times that are not
        # candidates for any CCD product (e.g. the intermediate 10-s Flats).
        if kind in ("Flat", "FibTh", "SimTh") and not any(
            bool(row[f"use_ccd{ccd}"]) for ccd in ("1", "2", "3")
        ):
            continue

        if not bool(row["has_ccd3"]):
            table["primary_signal_status"][i] = "missing"
            if kind != "Science":
                table["signal_use"][i] = False
                _set_issue(table, i, "CCD3 signal QA unavailable")
            continue

        try:
            frame = detector.preprocess_image(row["file_ccd3"], "3", config)
            primary_region = PRIMARY_SIGNAL_REGION[kind]
            primary_mode = "continuum" if kind in ("Flat", "Science") else "line"
            primary = _signal_check(frame, order_geometries, primary_region, primary_mode, config)
            table["primary_signal_status"][i] = "present" if primary.detected else "absent"
            table["primary_signal_score"][i] = primary.score
            table["primary_signal_orders"][i] = primary.n_orders_detected

            if kind != "Science" and not primary.detected:
                table["signal_use"][i] = False
                _set_issue(table, i, f"No significant {primary_region} signal on CCD3")
            elif kind == "Science" and not primary.detected:
                _set_issue(table, i, "Low/absent Science-region signal on CCD3; Science retained")

            lc_candidate = kind == "SimLC" or (
                "block_SimLC" in table.colnames and bool(str(row["block_SimLC"]).strip())
            )
            if lc_candidate:
                lc = primary if kind == "SimLC" else _signal_check(
                    frame, order_geometries, "SimLC", "line", config
                )
                table["simlc_signal_status"][i] = "present" if lc.detected else "absent"
                table["simlc_signal_score"][i] = lc.score
                table["simlc_signal_orders"][i] = lc.n_orders_detected
                table["simlc_signal"][i] = lc.detected
        except Exception as exc:
            table["primary_signal_status"][i] = "error"
            if kind != "Science":
                table["signal_use"][i] = False
            _set_issue(table, i, f"CCD3 signal QA failed: {type(exc).__name__}")
            logger.warning("CCD3 signal QA failed for run %s: %s", row["run"], exc)

    table.meta["signal_qa_done"] = True
    observations.update_calibration_blocks_from_qa(table)
    if verbose:
        _print_signal_qa_summary(table)
    return table


def _print_signal_qa_summary(table):
    """Print concise exposure-level QA after the CCD3 inspection pass."""
    print("\nCCD3 exposure signal QA")
    kinds = np.asarray(table["type"]).astype(str)
    calibration_use = np.asarray(
        table["calibration_use"] if "calibration_use" in table.colnames else table["use"], bool
    )
    signal_use = np.asarray(table["signal_use"], bool)

    rejected = calibration_use & ~signal_use & np.isin(kinds, ["Flat", "FibTh", "SimTh", "SimLC"])
    if np.any(rejected):
        for kind in ("Flat", "FibTh", "SimTh", "SimLC"):
            rows = table[rejected & (kinds == kind)]
            if len(rows):
                print(f"  rejected {kind:<6}: {observations._format_runs(rows['run'])}")
    else:
        print("  primary calibration signal: no failed exposures")

    science_low = (
        (kinds == "Science")
        & np.asarray(table["use"], bool)
        & (np.asarray(table["primary_signal_status"]).astype(str) == "absent")
    )
    if np.any(science_low):
        print(f"  low-signal Science retained: {observations._format_runs(table[science_low]['run'])}")

    detected = np.asarray(table["simlc_signal"], bool)
    if np.any(detected):
        print(f"  SimLC detected from CCD3: {observations._format_runs(table[detected]['run'])}")
    candidates = np.asarray(table["block_SimLC"]).astype(str) != "" if "block_SimLC" in table.colnames else np.zeros(len(table), bool)
    absent = candidates & ~detected & (np.asarray(table["simlc_signal_status"]).astype(str) == "absent")
    if np.any(absent):
        print(f"  SimLC candidates without detected signal: {observations._format_runs(table[absent]['run'])}")

def _calibration_region(kind):
    return "Science" if kind == "FibTh" else kind


def _combine_detector_frames(frames):
    """Streaming equal-weight mean of frames with variance propagation."""
    sum_image = sum_variance = n_good = quality = None
    shape = ccd = readout_mode = header = None
    n_frames = 0
    overscan_median, overscan_rms = {}, {}

    for frame in frames:
        if n_frames == 0:
            shape = frame.image.shape
            ccd, readout_mode = frame.ccd, frame.readout_mode
            header = frame.header.copy()
            sum_image = np.zeros(shape, float)
            sum_variance = np.zeros(shape, float)
            n_good = np.zeros(shape, np.uint16)
            quality = np.zeros(shape, np.uint16)
        else:
            if frame.ccd != ccd:
                raise ValueError("Cannot combine calibration frames from different CCDs")
            if frame.readout_mode != readout_mode:
                raise ValueError("Cannot combine calibration frames with different readout modes")
            if frame.image.shape != shape:
                raise ValueError("Cannot combine calibration frames with different image shapes")

        image = np.asarray(frame.image, float)
        variance = np.asarray(frame.variance, float)
        good = np.isfinite(image) & np.isfinite(variance) & (variance >= 0)
        sum_image[good] += image[good]
        sum_variance[good] += variance[good]
        n_good[good] += 1
        quality |= np.asarray(frame.quality_mask, np.uint16)
        for key, value in frame.overscan_median.items():
            overscan_median.setdefault(key, []).append(value)
        for key, value in frame.overscan_rms.items():
            overscan_rms.setdefault(key, []).append(value)
        n_frames += 1

    if n_frames == 0:
        raise ValueError("No detector frames supplied for calibration block")
    del frame, image, variance, good

    # Reuse the accumulators for the final mean/variance to keep the peak memory
    # close to one preprocessed detector frame plus four accumulator arrays.
    np.divide(sum_image, n_good, out=sum_image, where=n_good > 0)
    sum_image[n_good == 0] = np.nan
    np.divide(sum_variance, n_good, out=sum_variance, where=n_good > 0)
    np.divide(sum_variance, n_good, out=sum_variance, where=n_good > 0)
    sum_variance[n_good == 0] = np.nan

    header["NCOMBINE"] = n_frames
    return DetectorFrame(
        image=sum_image,
        variance=sum_variance,
        header=header,
        ccd=ccd,
        readout_mode=readout_mode,
        overscan_median={key: float(np.nanmean(value)) for key, value in overscan_median.items()},
        overscan_rms={key: float(np.nanmean(value)) for key, value in overscan_rms.items()},
        quality_mask=quality,
    )

def _extract_calibration_block(
    rows,
    block,
    kind,
    ccd,
    order_geometries,
    fibre_geometries,
    config,
    *,
    candidate_rows=None,
):
    """Combine accepted members of one block, extract them, and retain QA provenance."""
    exptimes = np.asarray(rows["exptime"], float)
    finite = exptimes[np.isfinite(exptimes)]
    if kind != "SimLC" and len(finite) and np.nanmax(finite) - np.nanmin(finite) > 0.01:
        raise ValueError(f"{block} CCD{ccd} contains mixed exposure times: {finite.tolist()}")

    frame = _combine_detector_frames(
        detector.preprocess_image(row[f"file_ccd{ccd}"], ccd, config)
        for row in rows
    )
    region = _calibration_region(kind)
    physical_orders, names = [], []
    summed_flux, summed_variance = [], []
    fibre_flux, fibre_variance = [], []

    for geometry in orders.geometry_for_ccd(order_geometries, ccd):
        if region in ("SimTh", "SimLC") and not geometry.available.get(region, False):
            continue
        order_matrix = orders.extract_order_matrix(frame, geometry)
        summed = extract_summed_order(order_matrix, components=(region,))
        physical_orders.append(geometry.order)
        names.append(geometry.name)
        summed_flux.append(summed.flux[:, 0])
        summed_variance.append(summed.variance[:, 0])
        if kind == "FibTh" and config.extraction_mode == "fibre":
            fibre_result = extract_fibre_order(order_matrix, fibre_geometries[geometry.name])
            science = fibre_result.select(SCIENCE_FIBRES)
            fibre_flux.append(science.flux)
            fibre_variance.append(science.variance)

    if not physical_orders:
        raise RuntimeError(f"No usable {kind} orders found for {block} CCD{ccd}")

    base = ExtractionResult(
        np.column_stack(summed_flux), np.column_stack(summed_variance),
        tuple(physical_orders), "summed",
    )
    mjds = np.asarray(rows["mjd_mid"], float)
    mjd_mid = float(np.nanmean(mjds)) if np.any(np.isfinite(mjds)) else np.nan
    exptime = float(np.nanmean(finite)) if len(finite) else np.nan
    exposure = ExtractedExposure(
        run=str(rows["run"][0]), kind=kind, ccd=ccd,
        mjd_mid=mjd_mid, exptime=exptime,
        orders=np.asarray(physical_orders, dtype=np.int16), summed=base,
        order_names=tuple(names),
        fibre_flux=np.stack(fibre_flux, axis=1) if fibre_flux else None,
        fibre_variance=np.stack(fibre_variance, axis=1) if fibre_variance else None,
    )

    candidate_rows = rows if candidate_rows is None else candidate_rows
    used_runs = tuple(str(run) for run in rows["run"])
    candidate_runs = tuple(str(run) for run in candidate_rows["run"])
    used_set = set(used_runs)
    exposure.block = str(block)
    exposure.runs = used_runs
    exposure.candidate_runs = candidate_runs
    exposure.rejected_runs = tuple(run for run in candidate_runs if run not in used_set)
    exposure.source_types = tuple(sorted(set(np.asarray(candidate_rows["type"]).astype(str))))
    exposure.member_mjd_mid = tuple(float(value) for value in rows["mjd_mid"])
    exposure.member_exptime = tuple(float(value) for value in rows["exptime"])
    exposure.qa_rows = candidate_rows.copy()
    return exposure


def _accepted_block_rows(rows, kind):
    """Filter CCD-specific block members using the image-based CCD3 QA."""
    if not len(rows):
        return rows
    if kind == "SimLC":
        if "simlc_signal" not in rows.colnames:
            raise RuntimeError("SimLC signal QA has not been run")
        return rows[np.asarray(rows["simlc_signal"], bool)]
    if "signal_use" not in rows.colnames:
        raise RuntimeError("Primary signal QA has not been run")
    return rows[np.asarray(rows["signal_use"], bool)]


def _print_extracted_calibration(exposure):
    """Print the final members and mean time of one CCD calibration product."""
    mjd = f"{float(exposure.mjd_mid):.6f}" if np.isfinite(exposure.mjd_mid) else "n/a"
    runs = observations._format_runs(getattr(exposure, "runs", (exposure.run,)))
    rejected = getattr(exposure, "rejected_runs", ())
    suffix = f"  rejected {observations._format_runs(rejected)}" if rejected else ""
    source = "/".join(getattr(exposure, "source_types", ()))
    source = f"  source {source}" if source and source != exposure.kind else ""
    print(f"  {exposure.block:<16} CCD{exposure.ccd}  MJD {mjd}  runs {runs}{suffix}{source}")


def extract_calibration_exposures(reduction_input, order_geometries, fibre_geometries, config):
    """Validate, combine, extract, and save calibration products block by block.

    The editable block structure lives in
    ``reduction_input.meta["calibration_blocks"]`` as
    ``kind -> CCD -> block_id -> metadata``. Flat/FibTh/SimTh candidates are
    already split by their CCD-specific exposure times; SimLC CCD2/3 candidates
    share the image-based CCD3 LC decision.

    CCD3 signal QA updates ``used_runs`` and ``rejected_runs`` in place. Deleted
    block keys and ``manual_rejected_runs`` are respected. Accepted raw frames
    are averaged with equal weight before extraction, with independent variance
    propagated as ``sum(V_i) / N**2``. SimLC frames are never exposure-time
    scaled. Products are saved immediately after each logical block.
    """
    inspect_ccd3_signals(reduction_input, order_geometries, config)
    blocks = observations.get_calibration_blocks(reduction_input)
    observations.update_calibration_blocks_from_qa(reduction_input, blocks)

    # Preserve the existing returned-product layout for downstream code while
    # using the nested block dictionary as the single source of membership.
    output = {
        kind: {ccd: [] for ccd in ("1", "2", "3")}
        for kind in ("SimLC", "SimTh", "FibTh")
    }
    directory = reduction_input.meta.get("calibration_directory")
    if directory is None:
        logger.warning(
            "No calibration_directory metadata on reduction_input; "
            "calibration products will not be saved automatically"
        )

    for kind in output:
        logger.info("Extracting %s calibration products", kind)

        # Process a logical block once, collecting any CCD-specific products.
        block_ids = []
        for ccd_blocks in blocks.get(kind, {}).values():
            for block in ccd_blocks:
                if block not in block_ids:
                    block_ids.append(block)

        for block in block_ids:
            block_output = {kind: {ccd: [] for ccd in ("1", "2", "3")}}
            used_any = False
            for ccd in ("1", "2", "3"):
                info = blocks.get(kind, {}).get(ccd, {}).get(block)
                if info is None or not info.get("use", True):
                    continue

                candidate_rows = observations._rows_for_runs(
                    reduction_input, info.get("candidate_runs", [])
                )
                rows = observations._rows_for_runs(
                    reduction_input, info.get("used_runs", [])
                )
                if not len(rows):
                    continue

                exposure = _extract_calibration_block(
                    rows,
                    block,
                    kind,
                    ccd,
                    order_geometries,
                    fibre_geometries,
                    config,
                    candidate_rows=candidate_rows,
                )
                # Use the editable block provenance verbatim. This preserves
                # manual rejections as well as automatic CCD3 QA decisions.
                exposure.candidate_runs = tuple(info.get("candidate_runs", ()))
                exposure.runs = tuple(info.get("used_runs", ()))
                exposure.rejected_runs = tuple(info.get("rejected_runs", ()))

                output[kind][ccd].append(exposure)
                block_output[kind][ccd].append(exposure)
                _print_extracted_calibration(exposure)
                used_any = True

            if not used_any:
                logger.info("    -> Skipping %s: no members passed calibration QA", block)
                continue
            if directory is not None:
                save_calibration_exposures(
                    block_output,
                    directory,
                    config.night,
                    overwrite=getattr(config, "overwrite", True),
                )
    return output

def _metadata_hdus(exposure, mode):
    primary = fits.PrimaryHDU()
    primary.header["PRODUCT"] = f"{exposure.kind.upper()}_{mode.upper()}"
    primary.header["KIND"] = exposure.kind
    primary.header["RUN"] = exposure.run
    primary.header["CCD"] = int(exposure.ccd)
    primary.header["MJD-MID"] = exposure.mjd_mid
    primary.header["EXPTIME"] = exposure.exptime
    primary.header["EXTRMODE"] = mode

    block = getattr(exposure, "block", "")
    runs = tuple(getattr(exposure, "runs", (exposure.run,)))
    candidates = tuple(getattr(exposure, "candidate_runs", runs))
    rejected = tuple(getattr(exposure, "rejected_runs", ()))
    source_types = tuple(getattr(exposure, "source_types", ()))
    if block:
        primary.header["BLOCK"] = block
    primary.header["NRUNS"] = len(runs)
    primary.header["NCAND"] = len(candidates)
    primary.header["NREJECT"] = len(rejected)
    primary.header["RUNFIRST"] = str(runs[0])
    primary.header["RUNLAST"] = str(runs[-1])
    member_exptime = np.asarray(getattr(exposure, "member_exptime", ()), float)
    primary.header["TOTEXP"] = float(np.nansum(member_exptime)) if len(member_exptime) else float(exposure.exptime) * len(runs)
    primary.header["COMBINE"] = "MEAN"
    primary.header["WEIGHT"] = "EQUAL"
    primary.header["SIGQA"] = "CCD3"
    if source_types:
        primary.header["SOURCE"] = ",".join(source_types)
    run_text = ",".join(runs)
    if len(run_text) <= 60:
        primary.header["RUNS"] = run_text

    order_table = Table({
        "COLUMN": np.arange(len(exposure.orders), dtype=np.int16),
        "ORDER": np.asarray(exposure.orders, dtype=np.int16),
    })
    return primary, fits.BinTableHDU(order_table, name="ORDERS")


def _qa_hdu(exposure):
    """Return per-candidate signal QA and inclusion provenance for a product."""
    rows = getattr(exposure, "qa_rows", None)
    if rows is None or not len(rows):
        return None
    used = set(getattr(exposure, "runs", ()))
    qa = Table()
    qa["RUN"] = np.asarray(rows["run"]).astype("U8")
    qa["TYPE"] = np.asarray(rows["type"]).astype("U12")
    qa["MJD_MID"] = np.asarray(rows["mjd_mid"], float)
    qa["EXPTIME"] = np.asarray(rows["exptime"], float)
    qa["USED"] = np.array([str(run) in used for run in rows["run"]], bool)
    for source, target in (
        ("primary_signal_status", "PRIMARY"),
        ("primary_signal_score", "PRIMSCORE"),
        ("simlc_signal_status", "SIMLC"),
        ("simlc_signal_score", "LCSCORE"),
    ):
        if source in rows.colnames:
            qa[target] = rows[source]
    return fits.BinTableHDU(qa, name="SIGNAL_QA")

def _calibration_filename(exposure, directory, night, mode):
    block = getattr(exposure, "block", "")
    tag = block.lower() if block else f"run{int(exposure.run):04d}"
    return Path(directory) / f"{exposure.kind.lower()}_{mode}_{night}_{tag}_ccd{exposure.ccd}.fits"


def save_extracted_exposure(exposure, directory, night, overwrite=True):
    """Write separate, self-describing summed and fibre calibration files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []

    primary, order_hdu = _metadata_hdus(exposure, "summed")
    filename = _calibration_filename(exposure, directory, night, "summed")
    hdus = [
        primary, order_hdu,
        fits.ImageHDU(np.asarray(exposure.summed.flux, np.float32), name="FLUX"),
        fits.ImageHDU(np.asarray(exposure.summed.variance, np.float32), name="VARIANCE"),
    ]
    qa_hdu = _qa_hdu(exposure)
    if qa_hdu is not None:
        hdus.append(qa_hdu)
    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
    files.append(filename)

    if exposure.fibre_flux is not None:
        primary, order_hdu = _metadata_hdus(exposure, "fibres")
        fibre_table = Table({
            "INDEX": np.arange(len(SCIENCE_FIBRES), dtype=np.int16),
            "FIBRE": np.array([str(f) for f in SCIENCE_FIBRES], dtype="U4"),
        })
        filename = _calibration_filename(exposure, directory, night, "fibres")
        hdus = [
            primary, order_hdu, fits.BinTableHDU(fibre_table, name="FIBRES"),
            fits.ImageHDU(np.asarray(exposure.fibre_flux, np.float32), name="FLUX"),
            fits.ImageHDU(np.asarray(exposure.fibre_variance, np.float32), name="VARIANCE"),
        ]
        qa_hdu = _qa_hdu(exposure)
        if qa_hdu is not None:
            hdus.append(qa_hdu)
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
        files.append(filename)
    return files


def save_calibration_exposures(exposures, directory, night, overwrite=True):
    """Save all extracted products in a calibration block."""
    files = []
    for kind in exposures.values():
        for ccd_exposures in kind.values():
            for exposure in ccd_exposures:
                files.extend(save_extracted_exposure(exposure, directory, night, overwrite))
    return files
