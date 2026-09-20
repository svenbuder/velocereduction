"""Build compact Flat calibrations after order/fibre geometry is known."""
from pathlib import Path
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d

from .constants import SCIENCE_FIBRES
from .models import DetectorFrame, FlatOrderCalibration
from . import detector, diagnostics, extraction, observations, orders

logger = logging.getLogger(__name__)


def _smooth_nan(values, sigma):
    values = np.asarray(values, float)
    good = np.isfinite(values)
    if not np.any(good):
        return np.full_like(values, np.nan)
    data = gaussian_filter1d(np.where(good, values, 0.0), sigma, mode="nearest")
    weight = gaussian_filter1d(good.astype(float), sigma, mode="nearest")
    return np.divide(data, weight, out=np.full_like(values, np.nan), where=weight > 0.05)


def smooth_spectrum(values, sigma):
    values = np.asarray(values, float)
    if values.ndim == 1:
        return _smooth_nan(values, sigma)
    return np.column_stack([_smooth_nan(values[:, i], sigma) for i in range(values.shape[1])])


def response_from_smooth(flat_flux, smooth_flux):
    return np.divide(
        flat_flux, smooth_flux,
        out=np.full_like(np.asarray(flat_flux, float), np.nan),
        where=np.isfinite(flat_flux) & np.isfinite(smooth_flux) & (smooth_flux > 0),
    )


def combine_flat_frames(reduction_input, config):
    """Combine normalized Flat DetectorFrames in memory"""
    combined = {}
    for ccd in ("1", "2", "3"):
        images, variances, masks, scales, runs, frames = [], [], [], [], [], []
        selected_flat_exposures = observations.select(reduction_input, "Flat", ccd)
        logger.info(
            "Combining CCD%s Flat frames from %d exposures: %s",
            ccd, len(selected_flat_exposures), ",".join([str(row["run"]) for row in selected_flat_exposures])
        )
        for row in selected_flat_exposures:
            frame = detector.preprocess_image(row[f"file_ccd{ccd}"], ccd, config)
            if np.nanpercentile(frame.image, 99) < 5000:
                logger.warning("CCD%s Flat %s rejected as too faint", ccd, row["run"])
                continue
            scale = float(np.nanpercentile(frame.image, 95))
            images.append(np.asarray(frame.image, float) / scale)
            variances.append(np.asarray(frame.variance, float) / scale ** 2)
            masks.append(np.asarray(frame.quality_mask, np.uint16))
            scales.append(scale)
            runs.append(str(row["run"]))
            frames.append(frame)
        if not images:
            raise RuntimeError(f"No usable Flat exposures for CCD{ccd}")

        image_stack = np.stack(images)
        variance_stack = np.stack(variances)
        valid = np.isfinite(image_stack)
        n_valid = valid.sum(axis=0)
        image = np.nanmedian(image_stack, axis=0)
        # Gaussian approximation for the variance of a sample median.
        variance = np.nanmedian(variance_stack, axis=0) * (np.pi / 2.0) / np.maximum(n_valid, 1)
        quality = np.zeros(image.shape, np.uint16)
        quality[n_valid == 0] = np.uint16(1 << 15)
        image[n_valid == 0] = np.nan
        variance[n_valid == 0] = np.nan
        first = frames[0]
        combined[ccd] = DetectorFrame(
            image=np.asarray(image, np.float32), variance=np.asarray(variance, np.float32),
            header=first.header.copy(), ccd=ccd, readout_mode=first.readout_mode,
            overscan_median=first.overscan_median, overscan_rms=first.overscan_rms,
            quality_mask=quality,
        )
        logger.info("CCD%s combined Flat %d exposures with input 95th-percentile scale median %.1f ADU:",ccd, len(images), np.nanmedian(scales))
        logger.info("%s", ",".join(runs))
    return combined


def extract_flat_order_matrices(combined_flats, order_geometries):
    """Create transient OrderMatrices from the in-memory combined Flats."""
    result = {}
    for ccd, frame in combined_flats.items():
        result.update(orders.extract_order_matrices(frame, order_geometries))
    return result


def _fibre_response(fibre_flat, fibre_smooth):
    """Return the small-scale response of each extracted fibre independently."""
    fibre_flat = np.asarray(fibre_flat, float)
    fibre_smooth = np.asarray(fibre_smooth, float)
    return np.divide(
        fibre_flat, fibre_smooth,
        out=np.full_like(fibre_flat, np.nan),
        where=np.isfinite(fibre_flat) & np.isfinite(fibre_smooth) & (fibre_smooth > 0),
    )


def build_flat_calibrations(flat_order_matrices, fibre_geometries, config, paths):
    """Create and save 1D summed Flat products and optional fibre products."""
    if paths.response_summed.exists() and not config.overwrite:
        products = load_flat_calibrations(paths, include_fibres=config.extraction_mode == "fibre")
        logger.info("Loaded cached Flat calibrations for %d orders", len(products))
        return products

    products = {}
    for name, order_matrix in flat_order_matrices.items():
        summed = extraction.extract_summed_order(order_matrix, components=("Science",))
        summed_flat = summed.flux[:, 0]
        summed_smooth = smooth_spectrum(summed_flat, config.flat_smooth_sigma)
        summed_response = response_from_smooth(summed_flat, summed_smooth)
        product = FlatOrderCalibration(
            ccd=order_matrix.ccd, order=order_matrix.order,
            summed_flat=np.asarray(summed_flat, np.float32),
            summed_smooth=np.asarray(summed_smooth, np.float32),
            summed_response=np.asarray(summed_response, np.float32),
        )

        if config.extraction_mode == "fibre":
            geometry = fibre_geometries[name]
            fibre = extraction.extract_fibre_order(order_matrix, geometry)
            fibre_smooth = smooth_spectrum(fibre.flux, config.flat_smooth_sigma)
            product.fibre_flat = np.asarray(fibre.flux, np.float32)
            product.fibre_smooth = np.asarray(fibre_smooth, np.float32)
            product.fibre_response = np.asarray(
                _fibre_response(fibre.flux, fibre_smooth), np.float32
            )
        products[name] = product

    save_flat_calibrations(products, paths, config)
    _log_flat_qa(products)
    if config.diagnostics != "none":
        diagnostics.plot_flat_summed_response(
            products, paths.figures / f"flat_response_summed_{config.night}.png"
        )
        diagnostics.plot_summed_response_image(
            products, paths.figures / f"response_summed_image_{config.night}.png"
        )
        if config.extraction_mode == "fibre":
            diagnostics.plot_flat_fibre_response(
                products, paths.figures / f"flat_response_fibres_{config.night}.png"
            )
            diagnostics.plot_flat_recombination_qa(
                products, fibre_geometries,
                paths.figures / f"flat_recombination_{config.night}.png",
            )
    return products


def _ordered_products(products):
    return sorted(products.values(), key=lambda p: (int(p.ccd), -int(p.order)))


def _order_table(products):
    ordered = _ordered_products(products)
    return Table({
        "INDEX": np.arange(len(ordered), dtype=np.int16),
        "CCD": np.array([int(p.ccd) for p in ordered], dtype=np.int16),
        "ORDER": np.array([p.order for p in ordered], dtype=np.int16),
        "ORDER_NAME": np.array([p.name for p in ordered], dtype="U24"),
    })


def _fibre_table(components):
    return Table({
        "INDEX": np.arange(len(components), dtype=np.int16),
        "FIBRE": np.array([str(component) for component in components], dtype="U4"),
    })


def _write_product(filename, products, attribute, config, components=None):
    ordered = _ordered_products(products)
    data = np.stack([np.asarray(getattr(p, attribute)) for p in ordered])
    primary = fits.PrimaryHDU()
    primary.header["PRODUCT"] = attribute.upper()
    primary.header["NIGHT"] = config.night
    primary.header["REFNIGHT"] = config.reference_night
    primary.header["NORDER"] = len(ordered)
    primary.header["NDISP"] = data.shape[1]
    hdus = [primary, fits.BinTableHDU(_order_table(products), name="ORDERS")]
    if components is not None:
        hdus.append(fits.BinTableHDU(_fibre_table(components), name="FIBRES"))
    image = fits.ImageHDU(np.asarray(data, np.float32), name="DATA")
    image.header["COMMENT"] = "NumPy axes: order, dispersion" + (", fibre" if data.ndim == 3 else "")
    hdus.append(image)
    fits.HDUList(hdus).writeto(filename, overwrite=True)


def save_flat_calibrations(products, paths, config):
    """Persist only compact extracted Flat spectra/smooth models/responses."""
    _write_product(paths.flat_summed, products, "summed_flat", config)
    _write_product(paths.flat_smooth_summed, products, "summed_smooth", config)
    _write_product(paths.response_summed, products, "summed_response", config)
    if config.extraction_mode == "fibre":
        first = next(p for p in products.values() if p.fibre_flat is not None)
        # Fibre component labels are stored in fibre_geometry; repeat them here for easy inspection.
        components = SCIENCE_FIBRES  # overwritten below if 24-component products are present
        n_fibre = first.fibre_flat.shape[1]
        if n_fibre != len(SCIENCE_FIBRES):
            from .constants import FIBRE_COMPONENTS
            components = FIBRE_COMPONENTS
        _write_product(paths.flat_fibres, products, "fibre_flat", config, components)
        _write_product(paths.flat_smooth_fibres, products, "fibre_smooth", config, components)
        _write_product(paths.response_fibres, products, "fibre_response", config, components)


def _read_product(filename):
    with fits.open(filename, memmap=False) as hdul:
        orders_table = Table(hdul["ORDERS"].data)
        data = np.asarray(hdul["DATA"].data, float)
    return orders_table, data


def load_flat_calibrations(paths, include_fibres=False):
    tables = {}
    for name, filename in (
        ("summed_flat", paths.flat_summed),
        ("summed_smooth", paths.flat_smooth_summed),
        ("summed_response", paths.response_summed),
    ):
        tables[name] = _read_product(filename)
    fibre_tables = {}
    if include_fibres:
        for name, filename in (
            ("fibre_flat", paths.flat_fibres),
            ("fibre_smooth", paths.flat_smooth_fibres),
            ("fibre_response", paths.response_fibres),
        ):
            fibre_tables[name] = _read_product(filename)

    order_table = tables["summed_flat"][0]
    products = {}
    for i, row in enumerate(order_table):
        name = str(row["ORDER_NAME"]).strip()
        product = FlatOrderCalibration(
            ccd=str(row["CCD"]), order=int(row["ORDER"]),
            summed_flat=tables["summed_flat"][1][i],
            summed_smooth=tables["summed_smooth"][1][i],
            summed_response=tables["summed_response"][1][i],
        )
        if include_fibres:
            product.fibre_flat = fibre_tables["fibre_flat"][1][i]
            product.fibre_smooth = fibre_tables["fibre_smooth"][1][i]
            product.fibre_response = fibre_tables["fibre_response"][1][i]
        products[name] = product
    return products


def blaze_from_summed_flat(product, normalise=True):
    """Return the smooth summed Flat as the blaze/large-scale illumination model."""
    blaze = np.asarray(product.summed_smooth, float).copy()
    if normalise and np.any(np.isfinite(blaze)):
        maximum = np.nanmax(blaze)
        if maximum > 0:
            blaze /= maximum
    return blaze


def _log_flat_qa(products):
    for ccd in ("1", "2", "3"):
        subset = [p for p in products.values() if p.ccd == ccd]
        scatter = [
            1.4826 * np.nanmedian(np.abs(p.summed_response - np.nanmedian(p.summed_response)))
            for p in subset
        ]
        logger.info(
            "CCD%s summed Flat response: %d orders; median robust response scatter %.4f%%",
            ccd, len(subset), 100 * np.nanmedian(scatter),
        )
        if subset and subset[0].fibre_response is not None:
            fscatter = [
                1.4826 * np.nanmedian(np.abs(p.fibre_response - np.nanmedian(p.fibre_response)))
                for p in subset
            ]
            logger.info(
                "CCD%s fibre Flat response: median robust scatter %.4f%%",
                ccd, 100 * np.nanmedian(fscatter),
            )
