from pathlib import Path
import logging
import numpy as np
from astropy.io import fits

from .constants import SCIENCE_FIBRES, SKY_FIBRES
from .models import ScienceExposure, ScienceOrder
from . import detector, diagnostics, extraction, observations, tramlines, velocities

logger = logging.getLogger(__name__)


def _linear_coordinates(wavelength, target):
    wavelength, target = np.asarray(wavelength, float), np.asarray(target, float)
    finite = np.isfinite(wavelength)
    if finite.sum() < 2:
        n = len(target); return np.zeros((n, 2), int), np.zeros((n, 2)), np.zeros(n, bool)
    source = np.where(finite)[0]
    order = np.argsort(wavelength[source]); source = source[order]; wave = wavelength[source]
    unique = np.r_[True, np.diff(wave) > 0]; source, wave = source[unique], wave[unique]
    if len(wave) < 2:
        n = len(target); return np.zeros((n, 2), int), np.zeros((n, 2)), np.zeros(n, bool)
    right = np.searchsorted(wave, target, side="right")
    valid = np.isfinite(target) & (target >= wave[0]) & (target <= wave[-1])
    right = np.clip(right, 1, len(wave) - 1); left = right - 1
    span = wave[right] - wave[left]
    alpha = np.divide(target - wave[left], span, out=np.zeros_like(target), where=span != 0)
    indices = np.column_stack((source[left], source[right]))
    weights = np.column_stack((1 - alpha, alpha)); weights[~valid] = 0
    return indices, weights, valid


def resample_linear(wavelength, flux, variance, target):
    indices, weights, valid = _linear_coordinates(wavelength, target)
    flux, variance = np.asarray(flux, float), np.asarray(variance, float)
    out_flux = np.sum(weights * flux[indices], axis=1)
    out_var = np.sum(weights ** 2 * variance[indices], axis=1)
    out_flux[~valid] = np.nan; out_var[~valid] = np.nan
    return out_flux, out_var


def resample_fibre_bundle(wavelength, flux, covariance, target):
    """Resample each fibre once and propagate same-pixel inter-fibre covariance."""
    wavelength, flux, covariance = map(np.asarray, (wavelength, flux, covariance))
    nt, nf = len(target), flux.shape[1]
    indices, weights, valid = np.zeros((nt, nf, 2), int), np.zeros((nt, nf, 2)), np.zeros((nt, nf), bool)
    out_flux = np.zeros((nt, nf), float)
    for fibre in range(nf):
        indices[:, fibre], weights[:, fibre], valid[:, fibre] = _linear_coordinates(wavelength[:, fibre], target)
        out_flux[:, fibre] = np.sum(weights[:, fibre] * flux[indices[:, fibre], fibre], axis=1)
    out_flux[~valid] = np.nan

    out_cov = np.zeros((nt, nf, nf), float)
    ii = np.arange(nf)[None, :, None]; jj = np.arange(nf)[None, None, :]
    for a in (0, 1):
        ia, wa = indices[:, :, a], weights[:, :, a]
        for b in (0, 1):
            ib, wb = indices[:, :, b], weights[:, :, b]
            same = ia[:, :, None] == ib[:, None, :]
            source_cov = covariance[ia[:, :, None], ii, jj]
            out_cov += wa[:, :, None] * wb[:, None, :] * np.where(same, source_cov, 0.0)
    pair_valid = valid[:, :, None] & valid[:, None, :]
    out_cov[~pair_valid] = np.nan
    return out_flux, out_cov


def correct_fibre_response(flux, covariance, response):
    response = np.asarray(response, float)
    good = np.isfinite(response) & (response > 0)
    flux = np.divide(flux, response, out=np.full_like(flux, np.nan, float), where=good)
    scale = response[:, :, None] * response[:, None, :]
    covariance = np.divide(
        covariance, scale, out=np.full_like(covariance, np.nan, float),
        where=np.isfinite(scale) & (scale > 0),
    )
    return flux, covariance


def _sky_weights(flux, covariance, sky_idx, clip=5.0):
    values = flux[sky_idx]; diagonal = np.diag(covariance)[sky_idx]
    good = np.isfinite(values) & np.isfinite(diagonal) & (diagonal > 0)
    if good.sum() < 1:
        return None, None
    centre = np.nanmedian(values[good]); scatter = 1.4826 * np.nanmedian(np.abs(values[good] - centre))
    if np.isfinite(scatter) and scatter > 0:
        good &= np.abs(values - centre) <= clip * scatter
    keep = sky_idx[good]
    if len(keep) == 0:
        return None, None
    c = covariance[np.ix_(keep, keep)]
    if not np.isfinite(c).all():
        return None, None
    inverse = np.linalg.pinv(c); one = np.ones(len(keep)); denominator = one @ inverse @ one
    if not np.isfinite(denominator) or denominator <= 0:
        return None, None
    return keep, (inverse @ one) / denominator


def combine_fibre_science(flux, covariance, components, clip=5.0):
    """Estimate one sky spectrum, subtract it from each science fibre, then sum."""
    flux, covariance = np.asarray(flux, float), np.asarray(covariance, float)
    science_idx = np.array([components.index(f) for f in SCIENCE_FIBRES], int)
    sky_idx = np.array([components.index(f) for f in SKY_FIBRES], int)
    nx, ns = flux.shape[0], len(science_idx)
    science_flux = np.full((nx, ns), np.nan); science_var = np.full_like(science_flux, np.nan)
    sky = np.full(nx, np.nan); sky_var = np.full(nx, np.nan)
    combined = np.full(nx, np.nan); combined_var = np.full(nx, np.nan)

    for x in range(nx):
        if not np.isfinite(covariance[x]).any():
            continue
        keep, weights = _sky_weights(flux[x], covariance[x], sky_idx, clip)
        if keep is None:
            continue
        sky[x] = weights @ flux[x, keep]
        sky_vector = np.zeros(flux.shape[1]); sky_vector[keep] = weights
        sky_var[x] = sky_vector @ covariance[x] @ sky_vector
        transform = np.zeros((ns, flux.shape[1])); transform[np.arange(ns), science_idx] = 1.0
        transform -= sky_vector[None, :]
        science_flux[x] = transform @ flux[x]
        science_cov = transform @ covariance[x] @ transform.T
        science_var[x] = np.diag(science_cov)
        combined[x] = np.nansum(science_flux[x])
        combined_var[x] = np.nansum(science_cov)
    return combined, combined_var, sky, sky_var, science_flux, science_var


def _summed_order(matrix, variance, row, flat_product, wavelength_model, ccd, order, mjd, berv):
    matrix, variance = detector.apply_response(matrix, variance, flat_product.response)
    result = extraction.extract_summed_order(matrix, variance, row)
    science_flux, sky1, sky2 = result.flux.T; science_var, sky1_var, sky2_var = result.variance.T
    sky = 19.0 * (sky1 + sky2) / 5.0
    sky_var = 19.0 ** 2 * (sky1_var + sky2_var) / 25.0
    flux, var = science_flux - sky, science_var + sky_var
    y = np.arange(len(flux), dtype=float); wave = wavelength_model.wavelength(ccd, order, y, mjd)
    return ScienceOrder(order, wave, velocities.barycentric_wavelength(wave, berv), flux, var, sky)


def _fibre_order(matrix, variance, row, flat_product, wavelength_model, ccd, order, mjd, berv):
    matrix, variance = detector.apply_response(matrix, variance, flat_product.response)
    result = extraction.extract_fibre_order(matrix, variance, flat_product.geometry, return_covariance=True)
    flux, covariance = correct_fibre_response(result.flux, result.covariance, flat_product.fibre_relative_response)
    y = np.arange(matrix.shape[0], dtype=float); target = wavelength_model.wavelength(ccd, order, y, mjd)
    fibre_wave = np.column_stack([
        wavelength_model.component_wavelength(ccd, order, y, mjd, component)
        for component in result.components
    ])
    resampled_flux, resampled_cov = resample_fibre_bundle(fibre_wave, flux, covariance, target)
    combined, combined_var, sky, _, science_flux, science_var = combine_fibre_science(
        resampled_flux, resampled_cov, result.components,
    )
    science_wave = fibre_wave[:, result.indices(SCIENCE_FIBRES)]
    return ScienceOrder(
        order, target, velocities.barycentric_wavelength(target, berv), combined, combined_var,
        len(SCIENCE_FIBRES) * sky, science_flux, science_var, science_wave,
    )


def save_science_exposure(exposure, filename, overwrite=True):
    filename = Path(filename)
    if filename.exists() and not overwrite:
        return filename
    primary = fits.PrimaryHDU(); primary.header["RUN"] = exposure.run; primary.header["OBJECT"] = exposure.object_name
    primary.header["CCD"] = int(exposure.ccd); primary.header["MJD-MID"] = exposure.mjd_mid
    primary.header["EXTRMODE"] = exposure.mode; primary.header["BERV"] = exposure.berv_kms
    arrays = {
        "ORDERS": np.array([o.order for o in exposure.orders], np.int16),
        "WAVELENGTH_NM": np.asarray([o.wavelength_nm for o in exposure.orders], np.float64),
        "BARY_WAVELENGTH_NM": np.asarray([o.barycentric_wavelength_nm for o in exposure.orders], np.float64),
        "FLUX": np.asarray([o.flux for o in exposure.orders], np.float32),
        "VARIANCE": np.asarray([o.variance for o in exposure.orders], np.float32),
        "SKY": np.asarray([o.sky for o in exposure.orders], np.float32),
    }
    hdus = [primary] + [fits.ImageHDU(data, name=name) for name, data in arrays.items()]
    if exposure.mode == "fibre":
        for name, data in (
            ("FIBRE_FLUX", np.asarray([o.fibre_flux for o in exposure.orders], np.float32)),
            ("FIBRE_VARIANCE", np.asarray([o.fibre_variance for o in exposure.orders], np.float32)),
            ("FIBRE_NATIVE_WAVELENGTH_NM", np.asarray([o.fibre_native_wavelength_nm for o in exposure.orders], np.float64)),
        ):
            hdu = fits.ImageHDU(data, name=name)
            for i, fibre in enumerate(SCIENCE_FIBRES): hdu.header[f"FIB{i:02d}"] = fibre
            hdus.append(hdu)
    fits.HDUList(hdus).writeto(filename, overwrite=True)
    return filename


def load_science_exposure(filename):
    with fits.open(filename, memmap=False) as hdul:
        h = hdul[0].header; orders = np.asarray(hdul["ORDERS"].data, int)
        wave, bary = np.asarray(hdul["WAVELENGTH_NM"].data), np.asarray(hdul["BARY_WAVELENGTH_NM"].data)
        flux, var, sky = np.asarray(hdul["FLUX"].data), np.asarray(hdul["VARIANCE"].data), np.asarray(hdul["SKY"].data)
        fibre_flux = np.asarray(hdul["FIBRE_FLUX"].data) if "FIBRE_FLUX" in hdul else None
        fibre_var = np.asarray(hdul["FIBRE_VARIANCE"].data) if "FIBRE_VARIANCE" in hdul else None
        fibre_wave = np.asarray(hdul["FIBRE_NATIVE_WAVELENGTH_NM"].data) if "FIBRE_NATIVE_WAVELENGTH_NM" in hdul else None
        result = [ScienceOrder(
            int(order), wave[i], bary[i], flux[i], var[i], sky[i],
            None if fibre_flux is None else fibre_flux[i], None if fibre_var is None else fibre_var[i],
            None if fibre_wave is None else fibre_wave[i],
        ) for i, order in enumerate(orders)]
        return ScienceExposure(
            str(h.get("RUN", "")), str(h.get("OBJECT", "")), str(h["CCD"]), float(h["MJD-MID"]),
            str(h.get("EXTRMODE", "summed")), result, float(h.get("BERV", np.nan)),
        )


def extract_science_exposures(reduction_input, nightly_tramlines, flat_products, wavelength_model, config, paths):
    exposures = []
    for obs in observations.select(reduction_input, "Science"):
        berv = velocities.barycentric_velocity_correction(obs["ra"], obs["dec"], obs["mjd_mid"]) \
            if np.isfinite(obs["ra"]) and np.isfinite(obs["dec"]) else np.nan
        for ccd in ("1", "2", "3"):
            if not obs[f"use_ccd{ccd}"]:
                continue
            filename = paths.science_products / f"science_{obs['run']}_ccd{ccd}.fits"
            if filename.exists() and not config.overwrite:
                exposure = load_science_exposure(filename)
                exposures.append(exposure)
                logger.info("Loaded cached science run %s CCD%s (%d orders)", exposure.run, ccd, len(exposure.orders))
                continue
            frame = detector.preprocess_image(obs[f"file_ccd{ccd}"], ccd, config); orders = []
            for row in nightly_tramlines:
                name = tramlines.order_name(row)
                if tramlines.ccd_from_order_name(name) != ccd:
                    continue
                matrix, _ = tramlines.extract_order_matrix(frame.image, row)
                variance, _ = tramlines.extract_order_matrix(frame.variance, row)
                args = (matrix, variance, row, flat_products[name], wavelength_model, ccd, tramlines.physical_order(name), float(obs["mjd_mid"]), berv)
                orders.append(_fibre_order(*args) if config.extraction_mode == "fibre" else _summed_order(*args))
            exposure = ScienceExposure(str(obs["run"]), str(obs["object"]), ccd, float(obs["mjd_mid"]), config.extraction_mode, orders, berv)
            save_science_exposure(exposure, filename, overwrite=True); exposures.append(exposure)
            order_snr = [
                np.nanmedian(np.divide(o.flux, np.sqrt(o.variance), out=np.full_like(o.flux, np.nan, float), where=np.asarray(o.variance) > 0))
                for o in orders
            ]
            logger.info(
                "Science run %s CCD%s: %d orders; median S/N %.1f per pixel; BERV %+.3f km/s",
                obs["run"], ccd, len(orders), float(np.nanmedian(order_snr)), berv,
            )
    if config.diagnostics != "none" and exposures:
        diagnostics.plot_science_summary(exposures, paths.figures / f"science_qa_{config.extraction_mode}.png")
    logger.info("Extracted/loaded %d science CCD exposures", len(exposures))
    return exposures
