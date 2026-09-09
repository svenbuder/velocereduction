import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d

from .constants import SCIENCE_FIBRES
from .models import FibreGeometry, FlatOrder
from . import detector, diagnostics, extraction, observations, tramlines

logger = logging.getLogger(__name__)


def _smooth_nan(values, sigma):
    values = np.asarray(values, float); good = np.isfinite(values)
    if not np.any(good): return np.full_like(values, np.nan)
    data = gaussian_filter1d(np.where(good, values, 0), sigma, mode="nearest")
    weight = gaussian_filter1d(good.astype(float), sigma, mode="nearest")
    return np.divide(data, weight, out=np.full_like(values, np.nan), where=weight > 0.05)


def smooth_matrix(matrix, sigma):
    return np.column_stack([_smooth_nan(matrix[:, j], sigma) for j in range(matrix.shape[1])]).astype(np.float32)


def create_response(matrix, smooth):
    return np.divide(
        matrix, smooth, out=np.full_like(matrix, np.nan, np.float32),
        where=np.isfinite(matrix) & np.isfinite(smooth) & (smooth > 0),
    )


def create_master_flat(reduction_input, config, paths):
    filename = paths.flat / "master_flat.fits"
    if filename.exists() and not config.overwrite:
        with fits.open(filename, memmap=False) as hdul:
            master = {f"ccd_{ccd}": np.asarray(hdul[f"CCD{ccd}"].data, np.float32) for ccd in ("1", "2", "3")}
        logger.info("Loaded cached master Flat: %s", filename)
        noise_file = paths.detector / "flat_read_noise.ecsv"
        if config.diagnostics != "none" and noise_file.exists():
            diagnostics.plot_read_noise(Table.read(noise_file, format="ascii.ecsv"), paths.figures / "detector_read_noise.png")
        return master

    master, runs, read_noise_rows = {}, {}, []
    for ccd in ("1", "2", "3"):
        images, accepted = [], []
        for row in observations.select(reduction_input, "Flat", ccd):
            frame = detector.preprocess_image(row[f"file_ccd{ccd}"], ccd, config)
            image = frame.image
            if np.nanpercentile(image, 99) < 5000:
                logger.warning("CCD%s Flat %s rejected as too faint", ccd, row["run"])
                continue
            scale = float(np.nanpercentile(image, 95))
            images.append((image / scale).astype(np.float32)); accepted.append(str(row["run"]))
            read_noise_rows.extend({
                "ccd": int(ccd), "run": int(row["run"]), "mjd_mid": float(row["mjd_mid"]),
                "readout_mode": frame.readout_mode, "amplifier": amp, "rms_adu": float(value),
            } for amp, value in frame.overscan_rms.items())
        if not images:
            raise RuntimeError(f"No usable Flat exposures for CCD{ccd}")
        master[f"ccd_{ccd}"] = np.nanmedian(np.stack(images), axis=0).astype(np.float32); runs[ccd] = accepted
        subset = [r["rms_adu"] for r in read_noise_rows if r["ccd"] == int(ccd)]
        logger.info(
            "CCD%s master Flat: combined %d exposures; median overscan RMS %.2f ADU",
            ccd, len(accepted), float(np.nanmedian(subset)),
        )

    hdus = [fits.PrimaryHDU()]; hdus[0].header["NIGHT"] = config.night; hdus[0].header["PRODUCT"] = "MASTER_FLAT"
    for ccd in ("1", "2", "3"):
        hdu = fits.ImageHDU(master[f"ccd_{ccd}"], name=f"CCD{ccd}"); hdu.header["NCOMBINE"] = len(runs[ccd]); hdu.header["RUNS"] = ",".join(runs[ccd]); hdus.append(hdu)
    fits.HDUList(hdus).writeto(filename, overwrite=True)
    read_noise_table = Table(rows=read_noise_rows)
    read_noise_table.write(paths.detector / "flat_read_noise.ecsv", format="ascii.ecsv", overwrite=True)
    if config.diagnostics != "none":
        diagnostics.plot_read_noise(read_noise_table, paths.figures / "detector_read_noise.png")
    return master


def _blaze(smooth, row):
    half = int(row["extraction_half_window"]); m = np.arange(-half, half + 1, dtype=float)
    weights = extraction.aperture_weights(m, float(row["Science_begin"]), float(row["Science_end"]))
    blaze = np.nansum(smooth * weights[None, :], axis=1); maximum = np.nanmax(blaze)
    return (blaze / maximum if maximum > 0 else blaze).astype(np.float32)


def _fibre_relative_response(fibre_smooth, components):
    science = [components.index(f) for f in SCIENCE_FIBRES]; reference = np.nanmedian(fibre_smooth[:, science], axis=1)
    return np.divide(fibre_smooth, reference[:, None], out=np.full_like(fibre_smooth, np.nan), where=reference[:, None] > 0).astype(np.float32)


def reconstruct_fibre_flat(geometry, fibre_smooth, background_smooth, n_crossdispersion=81, batch_size=512):
    m = np.arange(n_crossdispersion, dtype=float) - (n_crossdispersion - 1) / 2
    model = np.full((len(geometry.sigma), n_crossdispersion), np.nan, np.float32)
    for start in range(0, len(model), batch_size):
        stop = min(start + batch_size, len(model)); profiles = extraction.integrated_gaussian_cube(m, geometry.centres[start:stop], geometry.sigma[start:stop])
        model[start:stop] = np.einsum("xyf,xf->xy", profiles, fibre_smooth[start:stop], optimize=True) + background_smooth[start:stop, None]
    return model


def _write_arrays(filename, products, attribute, config):
    hdus = [fits.PrimaryHDU()]; hdus[0].header["NIGHT"] = config.night; hdus[0].header["PRODUCT"] = attribute.upper()
    for name, product in products.items():
        data = getattr(product, attribute)
        if data is not None: hdus.append(fits.ImageHDU(np.asarray(data), name=name))
    fits.HDUList(hdus).writeto(filename, overwrite=True)


def _write_geometry(filename, products, config):
    hdus = [fits.PrimaryHDU()]; hdus[0].header["NIGHT"] = config.night; hdus[0].header["PRODUCT"] = "FIBRE_GEOMETRY"
    for name, product in products.items():
        g = product.geometry
        if g is None: continue
        columns = [
            fits.Column(name="X", format="J", array=np.arange(len(g.sigma))),
            fits.Column(name="SIGMA", format="E", array=g.sigma.astype(np.float32)),
            fits.Column(name="SEPARATION", format="E", array=g.separation.astype(np.float32)),
            fits.Column(name="BUNDLE", format="E", array=g.bundle_offset.astype(np.float32)),
        ] + [fits.Column(name=f"CENTRE_{i:02d}", format="E", array=g.centres[:, i].astype(np.float32)) for i in range(len(g.components))]
        hdu = fits.BinTableHDU.from_columns(columns, name=name); hdu.header["NFIBRE"] = len(g.components)
        for i, component in enumerate(g.components):
            hdu.header[f"FIB{i:02d}"] = str(component); hdu.header[f"SLOT{i:02d}"] = float(g.slots[i]); hdu.header[f"OFF{i:02d}"] = float(g.fibre_offsets[i])
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(filename, overwrite=True)


def create_flat_products(master_flat, nightly_tramlines, config, paths):
    response_file = paths.flat_mode / "response.fits"
    if response_file.exists() and not config.overwrite:
        products = load_flat_products(config, paths)
        logger.info("Loaded cached %s Flat products for %d orders", config.extraction_mode, len(products))
        if config.diagnostics != "none":
            diagnostics.plot_flat_response_summary(products, paths.figures / f"flat_response_{config.extraction_mode}.png")
            if config.extraction_mode == "fibre":
                diagnostics.plot_fibre_geometry_summary(products, paths.figures / "fibre_geometry.png")
                diagnostics.plot_fibre_profile_summary(products, paths.figures / "fibre_profile_fits.png")
                if config.diagnostics == "full":
                    for name, product in products.items():
                        diagnostics.plot_fibre_profile_order(product, paths.debug / "fibre_profiles" / f"{name}.png")
        return products

    products = {}
    for row in nightly_tramlines:
        name = tramlines.order_name(row); ccd = tramlines.ccd_from_order_name(name)
        matrix, _, trace_offset = tramlines.extract_order_matrix(master_flat[f"ccd_{ccd}"], row, True)

        logger.debug(
            "%s geometry input: min=%.3g median=%.3g max=%.3g",
            name,
            np.nanmin(matrix),
            np.nanmedian(matrix),
            np.nanmax(matrix),
        )
        if config.extraction_mode == "fibre":
            geometry = extraction.fit_fibre_geometry(matrix, trace_offset, config.fibre_sample_step, config.fibre_sample_half_width, config.fibre_geometry_degree,label=name)
            fibre = extraction.extract_fibre_order(matrix, np.ones_like(matrix), geometry)
            fibre_smooth = smooth_matrix(fibre.flux, config.flat_smooth_sigma); background_smooth = _smooth_nan(fibre.background, config.flat_smooth_sigma)
            smooth = reconstruct_fibre_flat(geometry, fibre_smooth, background_smooth, matrix.shape[1])
            product = FlatOrder(name, matrix, smooth, create_response(matrix, smooth), _blaze(smooth, row), trace_offset, geometry, fibre.flux.astype(np.float32), fibre_smooth, _fibre_relative_response(fibre_smooth, geometry.components))
        else:
            smooth = smooth_matrix(matrix, config.flat_smooth_sigma)
            product = FlatOrder(name, matrix, smooth, create_response(matrix, smooth), _blaze(smooth, row), trace_offset)
        products[name] = product

    for ccd in ("1", "2", "3"):
        subset = [p for name, p in products.items() if str(name).split("_")[1] == ccd]
        if not subset:
            continue
        response_scatter = [1.4826 * np.nanmedian(np.abs(p.response - np.nanmedian(p.response))) for p in subset]
        if config.extraction_mode == "fibre":
            sigma = [np.nanmedian(p.geometry.sigma) for p in subset]
            separation = [np.nanmedian(p.geometry.separation) for p in subset]
            failed = sum(np.count_nonzero(~np.isfinite(p.geometry.sampled_sigma)) for p in subset)
            total = sum(len(p.geometry.sampled_sigma) for p in subset)
            logger.info(
                "CCD%s fibre geometry: %d orders; median sigma %.3f px; separation %.3f px; sampled failures %d/%d",
                ccd, len(subset), float(np.nanmedian(sigma)), float(np.nanmedian(separation)), failed, total,
            )
        logger.info("CCD%s Flat response: median robust scatter %.4f", ccd, float(np.nanmedian(response_scatter)))

    for attr in ("matrix", "trace_offset"): _write_arrays(paths.flat / f"{attr}.fits", products, attr, config)
    for attr in ("smooth", "response", "blaze"): _write_arrays(paths.flat_mode / f"{attr}.fits", products, attr, config)
    if config.extraction_mode == "fibre":
        for attr in ("fibre_flat", "fibre_smooth", "fibre_relative_response"): _write_arrays(paths.flat_mode / f"{attr}.fits", products, attr, config)
        _write_geometry(paths.flat_mode / "fibre_geometry.fits", products, config)
    if config.diagnostics != "none":
        diagnostics.plot_flat_response_summary(products, paths.figures / f"flat_response_{config.extraction_mode}.png")
        if config.extraction_mode == "fibre":
            diagnostics.plot_fibre_geometry_summary(products, paths.figures / "fibre_geometry.png")
            diagnostics.plot_fibre_profile_summary(products, paths.figures / "fibre_profile_fits.png")
            if config.diagnostics == "full":
                for name, product in products.items():
                    diagnostics.plot_fibre_profile_order(product, paths.debug / "fibre_profiles" / f"{name}.png")
    logger.info("Created %s Flat products for %d orders", config.extraction_mode, len(products))
    return products


def _read_array_file(filename):
    with fits.open(filename, memmap=False) as hdul: return {hdu.name.lower(): np.asarray(hdu.data) for hdu in hdul[1:]}


def _read_geometries(filename):
    geometries = {}
    with fits.open(filename, memmap=False) as hdul:
        for hdu in hdul[1:]:
            n = int(hdu.header["NFIBRE"])
            components = tuple(int(hdu.header[f"FIB{i:02d}"]) if str(hdu.header[f"FIB{i:02d}"]).isdigit() else str(hdu.header[f"FIB{i:02d}"]) for i in range(n))
            centres = np.column_stack([hdu.data[f"CENTRE_{i:02d}"] for i in range(n)])
            geometries[hdu.name.lower()] = FibreGeometry(
                components, np.array([hdu.header[f"SLOT{i:02d}"] for i in range(n)], float), centres,
                np.asarray(hdu.data["SIGMA"]), np.asarray(hdu.data["SEPARATION"]), np.asarray(hdu.data["BUNDLE"]),
                np.array([hdu.header[f"OFF{i:02d}"] for i in range(n)], float),
            )
    return geometries


def load_flat_products(config, paths):
    common = {attr: _read_array_file(paths.flat / f"{attr}.fits") for attr in ("matrix", "trace_offset")}
    mode = {attr: _read_array_file(paths.flat_mode / f"{attr}.fits") for attr in ("smooth", "response", "blaze")}
    fibre = {}; geometries = {}
    if config.extraction_mode == "fibre":
        fibre = {attr: _read_array_file(paths.flat_mode / f"{attr}.fits") for attr in ("fibre_flat", "fibre_smooth", "fibre_relative_response")}
        geometries = _read_geometries(paths.flat_mode / "fibre_geometry.fits")
    products = {}
    for key in common["matrix"]:
        products[key] = FlatOrder(
            key, common["matrix"][key], mode["smooth"][key], mode["response"][key], mode["blaze"][key], common["trace_offset"][key],
            geometries.get(key), fibre.get("fibre_flat", {}).get(key), fibre.get("fibre_smooth", {}).get(key), fibre.get("fibre_relative_response", {}).get(key),
        )
    return products
