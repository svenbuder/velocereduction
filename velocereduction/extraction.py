"""Measure 1D spectra from OrderMatrices using fixed extraction geometry."""
from pathlib import Path
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.special import ndtr

from .constants import SCIENCE_FIBRES
from .models import ExtractionResult, ExtractedExposure, OrderMatrix
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


def _calibration_region(kind):
    return "Science" if kind == "FibTh" else kind


def _calibration_rows(reduction_input, kind):
    if kind != "SimLC":
        return observations.select(reduction_input, kind)
    mask = np.asarray(reduction_input["use"], bool) & (
        (np.asarray(reduction_input["type"]).astype(str) == "SimLC")
        | np.asarray(reduction_input["lc_requested"], bool)
    )
    return reduction_input[mask]


def extract_calibration_exposures(reduction_input, order_geometries, fibre_geometries, config):
    """Extract detector-coordinate calibration spectra without Flat-response correction."""
    output = {kind: {ccd: [] for ccd in ("1", "2", "3")} for kind in ("SimLC", "SimTh", "FibTh")}
    for kind in output:
        region = _calibration_region(kind)
        for obs in _calibration_rows(reduction_input, kind):
            for ccd in ("1", "2", "3"):
                if kind == "SimLC" and ccd == "1":
                    continue
                if not obs[f"use_ccd{ccd}"]:
                    continue
                frame = detector.preprocess_image(obs[f"file_ccd{ccd}"], ccd, config)
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

                base = ExtractionResult(
                    np.column_stack(summed_flux), np.column_stack(summed_variance),
                    tuple(physical_orders), "summed",
                )
                exposure = ExtractedExposure(
                    run=str(obs["run"]), kind=kind, ccd=ccd,
                    mjd_mid=float(obs["mjd_mid"]), exptime=float(obs["exptime"]),
                    orders=np.asarray(physical_orders, dtype=np.int16), summed=base, order_names=tuple(names),
                    fibre_flux=np.stack(fibre_flux, axis=1) if fibre_flux else None,
                    fibre_variance=np.stack(fibre_variance, axis=1) if fibre_variance else None,
                )
                output[kind][ccd].append(exposure)
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
    order_table = Table({"COLUMN": np.arange(len(exposure.orders), dtype=np.int16), "ORDER": np.asarray(exposure.orders,dtype=np.int16,)})
    return primary, fits.BinTableHDU(order_table, name="ORDERS")


def save_extracted_exposure(exposure, directory, night, overwrite=True):
    """Write separate, self-describing summed and fibre calibration files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []

    primary, order_hdu = _metadata_hdus(exposure, "summed")
    filename = directory / f"{exposure.kind.lower()}_summed_{night}_run{int(exposure.run):04d}_ccd{exposure.ccd}.fits"
    fits.HDUList([
        primary, order_hdu,
        fits.ImageHDU(np.asarray(exposure.summed.flux, np.float32), name="FLUX"),
        fits.ImageHDU(np.asarray(exposure.summed.variance, np.float32), name="VARIANCE"),
    ]).writeto(filename, overwrite=overwrite)
    files.append(filename)

    if exposure.fibre_flux is not None:
        primary, order_hdu = _metadata_hdus(exposure, "fibres")
        fibre_table = Table({
            "INDEX": np.arange(len(SCIENCE_FIBRES), dtype=np.int16),
            "FIBRE": np.array([str(f) for f in SCIENCE_FIBRES], dtype="U4"),
        })
        filename = directory / f"{exposure.kind.lower()}_fibres_{night}_run{int(exposure.run):04d}_ccd{exposure.ccd}.fits"
        fits.HDUList([
            primary, order_hdu, fits.BinTableHDU(fibre_table, name="FIBRES"),
            fits.ImageHDU(np.asarray(exposure.fibre_flux, np.float32), name="FLUX"),
            fits.ImageHDU(np.asarray(exposure.fibre_variance, np.float32), name="VARIANCE"),
        ]).writeto(filename, overwrite=overwrite)
        files.append(filename)
    return files


def save_calibration_exposures(exposures, directory, night, overwrite=True):
    files = []
    for kind in exposures.values():
        for ccd_exposures in kind.values():
            for exposure in ccd_exposures:
                files.extend(save_extracted_exposure(exposure, directory, night, overwrite))
    return files
