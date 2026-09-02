from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import least_squares

from .models import DetectorFrame

logger = logging.getLogger(__name__)
OVERSCAN_BORDER = 32
DEFAULT_GAIN_FILE = Path(__file__).resolve().parent / "veloce_reference_data" / "detector_gains.ecsv"


def _robust_sigma(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return np.nan if values.size == 0 else 1.4826 * np.nanmedian(np.abs(values - np.nanmedian(values)))


def _amplifier(block, border=OVERSCAN_BORDER):
    mask = np.zeros(block.shape, bool)
    mask[:border] = mask[-border:] = True
    mask[:, :border] = mask[:, -border:] = True
    overscan = np.asarray(block, float)[mask]
    median, rms = float(np.nanmedian(overscan)), _robust_sigma(overscan)
    return np.asarray(block[border:-border, border:-border], float) - median, median, float(rms)


def _raw_amplifier_blocks(raw):
    if raw.shape == (4240, 4224):
        return {
            "q2": raw[:2120, :2112], "q1": raw[2120:, :2112],
            "q3": raw[:2120, 2112:], "q4": raw[2120:, 2112:],
        }, "4Amp"
    if raw.shape == (4176, 4224):
        return {"q1": raw[:, :2112], "q2": raw[:, 2112:]}, "2Amp"
    raise ValueError(f"Unexpected Veloce raw CCD shape: {raw.shape}")


def subtract_overscan(raw):
    blocks, readout = _raw_amplifier_blocks(np.asarray(raw))
    result, medians, rms = {}, {}, {}
    for amp, block in blocks.items():
        result[amp], medians[amp], rms[amp] = _amplifier(block)

    if readout == "4Amp":
        image = np.hstack((np.vstack((result["q2"], result["q1"])), np.vstack((result["q3"], result["q4"]))))
    else:
        image = np.hstack((result["q1"], result["q2"]))
    return image.astype(np.float32), medians, rms, readout


def amplifier_slices(shape, readout_mode):
    nx, ny = shape
    if readout_mode == "4Amp":
        nx2, ny2 = nx // 2, ny // 2
        return {
            "q2": (slice(0, nx2), slice(0, ny2)),
            "q1": (slice(nx2, nx), slice(0, ny2)),
            "q3": (slice(0, nx2), slice(ny2, ny)),
            "q4": (slice(nx2, nx), slice(ny2, ny)),
        }
    if readout_mode == "2Amp":
        ny2 = ny // 2
        return {"q1": (slice(0, nx), slice(0, ny2)), "q2": (slice(0, nx), slice(ny2, ny))}
    raise ValueError(f"Unknown readout mode: {readout_mode}")


@lru_cache(maxsize=8)
def _load_gain_table_cached(filename):
    return Table.read(filename, format="ascii.ecsv")


def load_detector_gains(filename=None):
    path = str(Path(filename or DEFAULT_GAIN_FILE).resolve())
    table = _load_gain_table_cached(path).copy()
    logger.debug("Loaded %d detector gain entries from %s", len(table), path)
    return table


def gain_for_amplifier(gains, ccd, readout_mode, amplifier):
    use = (
        (np.asarray(gains["ccd"]).astype(str) == str(ccd))
        & (np.asarray(gains["readout_mode"]).astype(str) == str(readout_mode))
        & (np.asarray(gains["amplifier"]).astype(str) == str(amplifier))
    )
    if use.sum() != 1:
        raise KeyError(f"Expected one gain for CCD{ccd} {readout_mode} {amplifier}; found {use.sum()}")
    return float(gains["gain_e_per_adu"][use][0])


def variance_image(image, overscan_rms, ccd, readout_mode, gains=None, include_poisson=True):
    image = np.asarray(image, float)
    gains = load_detector_gains() if gains is None else gains
    variance = np.empty(image.shape, np.float32)
    for amp, slc in amplifier_slices(image.shape, readout_mode).items():
        value = np.full(image[slc].shape, float(overscan_rms[amp]) ** 2, float)
        if include_poisson:
            value += np.clip(image[slc], 0, None) / gain_for_amplifier(gains, ccd, readout_mode, amp)
        variance[slc] = value
    return variance


def preprocess_image(filename, ccd, config=None, gains=None):
    with fits.open(filename, memmap=False) as hdul:
        raw = np.asarray(hdul[0].data, float)
        header = hdul[0].header.copy()
    image, medians, rms, readout = subtract_overscan(raw)
    include_poisson = True if config is None else config.use_poisson_variance
    if gains is None and include_poisson:
        gain_file = None if config is None else config.gain_file
        gains = load_detector_gains(gain_file)
    variance = variance_image(image, rms, ccd, readout, gains=gains, include_poisson=include_poisson)
    logger.debug(
        "CCD%s %s: readout=%s; overscan RMS [%s] ADU; Poisson variance=%s",
        ccd, Path(filename).name, readout,
        ", ".join(f"{amp}={rms[amp]:.2f}" for amp in sorted(rms)), include_poisson,
    )
    return DetectorFrame(image, variance, header, str(ccd), readout, medians, rms)


def apply_response(image, variance, response):
    image, variance, response = map(lambda x: np.asarray(x, float), (image, variance, response))
    valid = np.isfinite(response) & (response > 0)
    flux = np.full_like(image, np.nan)
    var = np.full_like(variance, np.nan)
    flux[valid] = image[valid] / response[valid]
    var[valid] = variance[valid] / response[valid] ** 2
    return flux, var


# ---- Optional detector characterisation; not part of the nightly reduction. ----

def _read_amplifiers(filename):
    with fits.open(filename, memmap=False) as hdul:
        raw = np.asarray(hdul[0].data, float)
        header = hdul[0].header.copy()
    blocks, readout = _raw_amplifier_blocks(raw)
    amplifiers = {}
    for amp, block in blocks.items():
        image, median, rms = _amplifier(block)
        amplifiers[amp] = {"image": image, "overscan_median": median, "overscan_rms": rms}
    return {
        "filename": Path(filename), "readout_mode": readout,
        "exptime": float(header.get("EXPTIME", np.nan)),
        "mjd": float(header.get("MJD-OBS", np.nan)),
        "run": int(header.get("RUN", -1)), "amplifiers": amplifiers,
    }


def _file_metadata(filename):
    with fits.open(filename, memmap=True) as hdul:
        header, shape = hdul[0].header, hdul[0].shape
    if shape == (4240, 4224):
        readout = "4Amp"
    elif shape == (4176, 4224):
        readout = "2Amp"
    else:
        raise ValueError(f"Unexpected Veloce raw CCD shape: {shape}")
    return {
        "filename": Path(filename), "readout_mode": readout,
        "exptime": float(header.get("EXPTIME", np.nan)),
        "mjd": float(header.get("MJD-OBS", np.nan)), "run": int(header.get("RUN", -1)),
    }


def _pair_files(flat_files):
    groups = defaultdict(list)
    for row in map(_file_metadata, flat_files):
        groups[(row["readout_mode"], round(row["exptime"], 6))].append(row)
    pairs = []
    for (readout, exptime), rows in groups.items():
        rows.sort(key=lambda r: (np.inf if not np.isfinite(r["mjd"]) else r["mjd"], r["run"], str(r["filename"])))
        if len(rows) % 2:
            logger.warning("%s %.6g-s Flats: ignoring unpaired %s", readout, exptime, rows[-1]["filename"].name)
        pairs.extend((rows[i], rows[i + 1]) for i in range(0, len(rows) - 1, 2))
    if not pairs:
        raise ValueError("No usable consecutive Flat pairs were found")
    return pairs


def _pair_binned_statistics(image1, image2, n_bins=30, signal_range=(1000, 50000), max_level_difference=0.05, min_pixels=500):
    image1, image2 = np.asarray(image1, float), np.asarray(image2, float)
    preliminary = 0.5 * (image1 + image2)
    valid = np.isfinite(image1) & np.isfinite(image2) & (image1 > 0) & (image2 > 0)
    valid &= (preliminary >= signal_range[0]) & (preliminary <= signal_range[1])
    if valid.sum() < min_pixels:
        return []
    ratio = float(np.nanmedian(image1[valid] / image2[valid]))
    if not np.isfinite(ratio) or ratio <= 0 or abs(ratio - 1) > max_level_difference:
        return []

    scaled2 = ratio * image2
    signal, difference = 0.5 * (image1 + scaled2), image1 - scaled2
    values = signal[valid]
    edges = np.unique(np.nanquantile(values, np.linspace(0, 1, n_bins + 1)))
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        use = valid & (signal >= left) & (signal < right)
        if use.sum() < min_pixels:
            continue
        d = difference[use]
        centre, sigma = np.nanmedian(d), _robust_sigma(d)
        keep = np.isfinite(d) & (np.abs(d - centre) < 5 * sigma)
        if keep.sum() < min_pixels:
            continue
        rows.append({
            "signal_adu": float(np.nanmedian(signal[use])),
            "variance_difference_adu2": float(_robust_sigma(d[keep]) ** 2),
            "pair_scale": ratio, "n_pixels": int(keep.sum()),
        })
    return rows


def _fit_gain(points):
    signal = np.array([p["signal_adu"] for p in points], float)
    variance = np.array([p["variance_difference_adu2"] for p in points], float)
    ratio = np.array([p["pair_scale"] for p in points], float)
    n_pixels = np.array([p["n_pixels"] for p in points], float)
    design = np.column_stack((signal * (1 + ratio), 1 + ratio ** 2))
    uncertainty = np.maximum(variance * np.sqrt(2 / np.maximum(n_pixels - 1, 1)), np.nanmedian(variance) * 1e-4)
    initial, *_ = np.linalg.lstsq(design, variance, rcond=None)
    initial = np.maximum(initial, [1e-8, 0])
    residual = lambda p: (design @ p - variance) / uncertainty
    robust = least_squares(residual, initial, bounds=([1e-8, 0], [np.inf, np.inf]), loss="soft_l1")
    keep = np.abs(residual(robust.x)) < 5
    if keep.sum() < 4:
        keep[:] = True
    final = least_squares(lambda p: residual(p)[keep], robust.x, bounds=([1e-8, 0], [np.inf, np.inf]))
    inverse_gain, read_noise2 = final.x
    gain, read_noise = 1 / inverse_gain, np.sqrt(read_noise2)
    dof = max(keep.sum() - 2, 1)
    chi2 = np.sum(residual(final.x)[keep] ** 2)
    covariance = np.linalg.pinv(final.jac.T @ final.jac) * chi2 / dof
    gain_err = np.sqrt(max(covariance[0, 0], 0)) / inverse_gain ** 2
    rn2_err = np.sqrt(max(covariance[1, 1], 0))
    rn_err = 0.5 * rn2_err / read_noise if read_noise > 0 else np.nan
    return {
        "gain_e_per_adu": float(gain), "gain_err_e_per_adu": float(gain_err),
        "read_noise_adu": float(read_noise), "read_noise_err_adu": float(rn_err),
        "read_noise_e": float(read_noise * gain), "reduced_chi2": float(chi2 / dof),
        "used": keep, "model_variance_difference_adu2": design @ final.x,
    }


def characterise_detector_gain(flat_files, ccd, output_file=None, n_bins=30, signal_range=(1000, 50000), max_level_difference=0.05, min_pixels=500):
    points, overscan, n_pairs = defaultdict(list), defaultdict(list), defaultdict(int)
    for meta1, meta2 in _pair_files(flat_files):
        a, b = _read_amplifiers(meta1["filename"]), _read_amplifiers(meta2["filename"])
        if a["readout_mode"] != b["readout_mode"]:
            continue
        for amp in a["amplifiers"]:
            stats = _pair_binned_statistics(
                a["amplifiers"][amp]["image"], b["amplifiers"][amp]["image"],
                n_bins, signal_range, max_level_difference, min_pixels,
            )
            if not stats:
                continue
            key = (a["readout_mode"], amp)
            points[key].extend(stats)
            overscan[key].extend([a["amplifiers"][amp]["overscan_rms"], b["amplifiers"][amp]["overscan_rms"]])
            n_pairs[key] += 1

    rows, diagnostics = [], {}
    for (readout, amp), data in sorted(points.items()):
        if len(data) < 4:
            continue
        fit = _fit_gain(data)
        signal = np.array([p["signal_adu"] for p in data])
        ratio = np.array([p["pair_scale"] for p in data])
        rows.append({
            "ccd": str(ccd), "readout_mode": readout, "amplifier": amp,
            "gain_e_per_adu": fit["gain_e_per_adu"], "gain_err_e_per_adu": fit["gain_err_e_per_adu"],
            "read_noise_adu": fit["read_noise_adu"], "read_noise_err_adu": fit["read_noise_err_adu"],
            "read_noise_e": fit["read_noise_e"], "overscan_rms_adu": float(np.nanmedian(overscan[(readout, amp)])),
            "n_pairs": n_pairs[(readout, amp)], "n_bins": len(data),
            "signal_min_adu": float(signal.min()), "signal_max_adu": float(signal.max()),
            "median_pair_scale": float(np.nanmedian(ratio)), "reduced_chi2": fit["reduced_chi2"],
        })
        diagnostics[(readout, amp)] = {
            "signal_adu": signal,
            "variance_difference_adu2": np.array([p["variance_difference_adu2"] for p in data]),
            "pair_scale": ratio, **fit,
        }
    if not rows:
        raise RuntimeError("No amplifier gain could be fitted")
    table = Table(rows=rows)
    table.meta["description"] = "Veloce amplifier photon-transfer characterisation"
    table.meta["source_files"] = [str(Path(f)) for f in flat_files]
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        table.write(output_file, format="ascii.ecsv", overwrite=True)
        _load_gain_table_cached.cache_clear()
    return table, diagnostics


def plot_gain_characterisation(diagnostics, ccd, output_directory=None):
    import matplotlib.pyplot as plt
    output_directory = Path(output_directory) if output_directory else None
    if output_directory:
        output_directory.mkdir(parents=True, exist_ok=True)
    figures = []
    for (readout, amp), d in diagnostics.items():
        order = np.argsort(d["signal_adu"])
        scale = 1 + d["pair_scale"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(d["signal_adu"], d["variance_difference_adu2"] / scale, ".", label="Flat pairs")
        ax.plot(d["signal_adu"][order], d["model_variance_difference_adu2"][order] / scale[order], label="fit")
        ax.set(xlabel="Mean signal / ADU", ylabel="Difference variance / ADU$^2$",
               title=f"CCD{ccd} {readout} {amp}: {d['gain_e_per_adu']:.3f} e-/ADU")
        ax.legend()
        figures.append(fig)
        if output_directory:
            fig.savefig(output_directory / f"gain_ccd{ccd}_{readout}_{amp}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    return figures
