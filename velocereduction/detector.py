from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import least_squares
from skimage.registration import phase_cross_correlation

from .constants import REFERENCE_NIGHT
from . import diagnostics, observations
from .models import DetectorFrame

logger = logging.getLogger(__name__)

REFERENCE_RUNS = {"1": "0001", "2": "0002", "3": "0003"}

OVERSCAN_BORDER = 32
RAW_16BIT_MAX = np.iinfo(np.uint16).max
MASK_SATURATED = np.uint16(1 << 0)
DEFAULT_GAIN_FILE = Path(__file__).resolve().parent / "veloce_reference_data" / "detector_gains.ecsv"


# ---- Nightly detector preprocessing. ----

def _robust_sigma(values):
    """Return a robust 1-sigma scatter estimate (1.4826 x MAD), ignoring non-finite values."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return np.nan if values.size == 0 else 1.4826 * np.nanmedian(np.abs(values - np.nanmedian(values)))


def _overscan_strips(block, border=OVERSCAN_BORDER):
    """
    Return the four non-overlapping overscan strips surrounding an amplifier.

    Corners are excluded so every overscan pixel is used once. Left/right strips
    sample each science row; top/bottom strips sample each science column.
    """
    block = np.asarray(block, float)
    if min(block.shape) <= 2 * border:
        raise ValueError(f"Amplifier block {block.shape} is too small for a {border}-pixel overscan border")
    return (
        block[border:-border, :border],
        block[border:-border, -border:],
        block[:border, border:-border],
        block[-border:, border:-border],
    )


def _overscan_statistics(block, border=OVERSCAN_BORDER, clip_sigma=5.0):
    """
    Measure amplifier bias level and read noise from its overscan border.

    The bias level is the median of all overscan strips. For the read noise,
    row medians are removed from the left/right strips and column medians from
    the top/bottom strips. A global MAD estimate only defines a robust outlier
    clip; the final read noise is sqrt(mean(local sample variances)).
    """
    left, right, top, bottom = _overscan_strips(block, border)
    overscan = np.concatenate([x.ravel() for x in (left, right, top, bottom)])
    finite = overscan[np.isfinite(overscan)]
    bias = float(np.median(finite)) if finite.size else np.nan

    local = []
    residuals = []
    for strip, axis in ((left, 1), (right, 1), (top, 0), (bottom, 0)):
        centre = np.nanmedian(strip, axis=axis, keepdims=True)
        residual = strip - centre
        residuals.append(residual.ravel())
        local.extend(np.moveaxis(residual, axis, -1).reshape(-1, residual.shape[axis]))

    residuals = np.concatenate(residuals)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 2:
        return bias, np.nan

    robust = _robust_sigma(residuals)
    variances = []
    for values in local:
        values = np.asarray(values, float)
        good = np.isfinite(values)
        if np.isfinite(robust) and robust > 0:
            good &= np.abs(values) < clip_sigma * robust
        if good.sum() >= 2:
            variances.append(np.var(values[good], ddof=1))

    read_noise = float(np.sqrt(np.mean(variances))) if variances else np.nan
    return bias, read_noise


def _amplifier(block, border=OVERSCAN_BORDER):
    """
    Overscan-subtract one amplifier and return science pixels, bias, and read noise.

    Each amplifier has a 32-pixel overscan border on every edge. The science
    region is corrected by one robust amplifier bias level; read noise is
    measured from locally de-trended overscan rows/columns, not from the global
    scatter of the complete border.
    """
    median, rms = _overscan_statistics(block, border)
    science = np.asarray(block[border:-border, border:-border], float) - median
    return science, median, rms

def _raw_amplifier_blocks(raw):
    """
    Split a raw Veloce CCD image into its physical amplifier blocks.
    
    4Amp data (4240 x 4224) are split into four 2120 x 2112 blocks and 2Amp data (4176 x 4224) into two 4176 x 2112 halves.
    Each returned block still contains its own 32-pixel overscan border on every edge.
    """
    if raw.shape == (4240, 4224):
        return {
            "q1": raw[:2120, :2112], "q2": raw[2120:, :2112],
            "q3": raw[:2120, 2112:], "q4": raw[2120:, 2112:],
        }, "4Amp"
    if raw.shape == (4176, 4224):
        return {"q1": raw[:, :2112], "q2": raw[:, 2112:]}, "2Amp"
    raise ValueError(f"Unexpected Veloce raw CCD shape: {raw.shape}")


def _reassemble_amplifiers(result, readout):
    """Reassemble trimmed amplifier arrays in the same physical orientation as the science CCD."""
    if readout == "4Amp":
        return np.hstack((
            np.vstack((result["q1"], result["q2"])),
            np.vstack((result["q3"], result["q4"])),
        ))
    if readout == "2Amp":
        return np.hstack((result["q1"], result["q2"]))
    raise ValueError(f"Unknown readout mode: {readout}")


def quality_mask_from_raw(raw, border=OVERSCAN_BORDER, saturation_level=RAW_16BIT_MAX):
    """
    Build the detector quality bitmask from raw pixels before overscan subtraction.

    Bit `MASK_SATURATED` is set where a science pixel reaches the 16-bit ceiling
    (65535 by default). The mask is uint16 so additional quality bits can be
    added later without changing the data model; zero always means unflagged.
    """
    blocks, readout = _raw_amplifier_blocks(np.asarray(raw))
    masks = {}
    for amp, block in blocks.items():
        science = np.asarray(block[border:-border, border:-border])
        mask = np.zeros(science.shape, dtype=np.uint16)
        saturated = np.isfinite(science) & (science >= saturation_level)
        mask[saturated] |= MASK_SATURATED
        masks[amp] = mask
    return _reassemble_amplifiers(masks, readout)


def subtract_overscan(raw):
    """
    Overscan-subtract every amplifier and reconstruct the 4112 x 4096 science image.
    
    Each amplifier loses its 32-pixel border, that is the frame as well as
    the cross (4Amp) and vertical line (2Amp), respectively.
    """
    blocks, readout = _raw_amplifier_blocks(np.asarray(raw))
    result, medians, rms = {}, {}, {}
    for amp, block in blocks.items():
        result[amp], medians[amp], rms[amp] = _amplifier(block)

    image = _reassemble_amplifiers(result, readout)
    return image.astype(np.float32), medians, rms, readout


def amplifier_slices(shape, readout_mode):
    """
    Return the science-image slice belonging to each amplifier after overscan trimming.
    
    The mapping matches the reassembly performed by `subtract_overscan()` and is used to
    apply amplifier-specific gains/read-noise variances to the final 4112 x 4096 image.
    """
    nx, ny = shape
    if readout_mode == "4Amp":
        nx2, ny2 = nx // 2, ny // 2
        return {
            "q1": (slice(0, nx2), slice(0, ny2)),
            "q2": (slice(nx2, nx), slice(0, ny2)),
            "q3": (slice(0, nx2), slice(ny2, ny)),
            "q4": (slice(nx2, nx), slice(ny2, ny)),
        }
    if readout_mode == "2Amp":
        ny2 = ny // 2
        return {"q1": (slice(0, nx), slice(0, ny2)), "q2": (slice(0, nx), slice(ny2, ny))}
    raise ValueError(f"Unknown readout mode: {readout_mode}")


@lru_cache(maxsize=8)
def _load_gain_table_cached(filename):
    """Read and cache the persistent ECSV detector-gain calibration table to avoid repeated disk I/O."""
    return Table.read(filename, format="ascii.ecsv")


def load_detector_gains(filename=None):
    """Load a copy of the detector-gain calibration table, using the packaged reference table by default."""
    path = str(Path(filename or DEFAULT_GAIN_FILE).resolve())
    table = _load_gain_table_cached(path).copy()
    return table


def gain_for_amplifier(gains, ccd, readout_mode, amplifier):
    """Return the characterised gain (e-/ADU) for one CCD, readout mode, and amplifier."""
    use = (
        (np.asarray(gains["ccd"]).astype(str) == str(ccd))
        & (np.asarray(gains["readout_mode"]).astype(str) == str(readout_mode))
        & (np.asarray(gains["amplifier"]).astype(str) == str(amplifier))
    )
    if use.sum() != 1:
        raise KeyError(f"Expected one gain for CCD{ccd} {readout_mode} {amplifier}; found {use.sum()}")
    return float(gains["gain_e_per_adu"][use][0])


def variance_image(image, overscan_rms, ccd, readout_mode, gains=None, include_poisson=True):
    """
    Build the per-pixel variance image in ADU^2 for an overscan-subtracted CCD.
    
    Read variance comes from that exposure's amplifier overscan RMS;
    photon variance is `max(counts, 0) / gain` using the persistent amplifier gain calibration.
    """
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
    """
    Read one raw exposure, overscan-correct it, and build variance and quality mask.

    Saturation is identified from raw science pixels before bias subtraction.
    Flagged pixels retain their measured counts but receive infinite variance,
    so downstream variance-weighted/summed extraction excludes them naturally.
    """
    with fits.open(filename, memmap=False) as hdul:
        raw = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()

    quality_mask = quality_mask_from_raw(raw)
    image, medians, rms, readout = subtract_overscan(raw)
    include_poisson = True if config is None else config.use_poisson_variance
    if gains is None and include_poisson:
        gain_file = None if config is None else config.gain_file
        gains = load_detector_gains(gain_file)

    variance = variance_image(image, rms, ccd, readout, gains=gains, include_poisson=include_poisson)
    variance[quality_mask != 0] = np.inf

    n_saturated = int(np.count_nonzero(quality_mask & MASK_SATURATED))
    logger.debug(
        "CCD%s %s: readout=%s; overscan RMS [%s] ADU; Poisson variance=%s; saturated=%d",
        ccd, Path(filename).name, readout,
        ", ".join(f"{amp}={rms[amp]:.2f}" for amp in sorted(rms)),
        include_poisson, n_saturated,
    )
    if n_saturated:
        logger.warning("CCD%s %s: %d science pixels reached the 16-bit maximum", ccd, Path(filename).name, n_saturated)

    return DetectorFrame(image, variance, header, str(ccd), readout, medians, rms, quality_mask)

def apply_response(image, variance, response):
    """
    Divide an image by a response map and propagate its variance.
    
    For flux `f/r`, the variance becomes `V/r^2`;
    invalid or non-positive response pixels are returned as NaN.
    """
    image, variance, response = map(lambda x: np.asarray(x, float), (image, variance, response))
    valid = np.isfinite(response) & (response > 0)
    flux = np.full_like(image, np.nan)
    var = np.full_like(variance, np.nan)
    flux[valid] = image[valid] / response[valid]
    var[valid] = variance[valid] / response[valid] ** 2
    return flux, var


# ---- Detector gain characterisation; not part of the nightly reduction. ----

def _read_amplifiers(filename):
    """Read one raw FITS file and return each amplifier as an overscan-subtracted science image plus overscan statistics."""
    with fits.open(filename, memmap=False) as hdul:
        raw = np.asarray(hdul[0].data, float)
        header = hdul[0].header.copy()
    blocks, readout = _raw_amplifier_blocks(raw)
    amplifiers = {}
    for amp, block in blocks.items():
        image, median, rms = _amplifier(block)
        raw_science = np.asarray(block[OVERSCAN_BORDER:-OVERSCAN_BORDER, OVERSCAN_BORDER:-OVERSCAN_BORDER])
        quality_mask = np.zeros(raw_science.shape, dtype=np.uint16)
        quality_mask[np.isfinite(raw_science) & (raw_science >= RAW_16BIT_MAX)] |= MASK_SATURATED
        image = image.copy()
        image[quality_mask != 0] = np.nan
        amplifiers[amp] = {
            "image": image, "overscan_median": median, "overscan_rms": rms,
            "quality_mask": quality_mask,
        }
    return {
        "filename": Path(filename), "readout_mode": readout,
        "exptime": float(header.get("EXPTIME", np.nan)),
        "mjd": float(header.get("MJD-OBS", np.nan)),
        "run": int(header.get("RUN", -1)), "amplifiers": amplifiers,
    }


def _file_metadata(filename):
    """Read only the metadata needed to group and order Flat files for gain characterisation."""
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
    """
    Group Flats by readout mode/exposure time and pair consecutive exposures.
    
    Files are sorted by MJD, then RUN/name; an unpaired final Flat is ignored with a warning.
    """
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
    """
    Measure photon-transfer points from one pair of amplifier Flats.
    
    The second Flat is scaled to the first to tolerate small lamp changes, pixels are binned by mean signal, 
    and robust difference variances are measured after outlier rejection.
    """
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


def _fit_gain(points, overscan_rms):
    """
    Fit amplifier gain and read noise to binned Flat-pair statistics.
    
    Uses `Var(F1-rF2) = S(1+r)/gain + RN^2(1+r^2)`, so the fitted slope gives gain
    while the intercept gives a detector-characterisation read-noise estimate.
    """
    signal = np.array([p["signal_adu"] for p in points], float)
    variance = np.array([p["variance_difference_adu2"] for p in points], float)
    ratio = np.array([p["pair_scale"] for p in points], float)
    n_pixels = np.array([p["n_pixels"] for p in points], float)
    design = np.column_stack((signal * (1 + ratio), 1 + ratio ** 2))
    uncertainty = np.maximum(variance * np.sqrt(2 / np.maximum(n_pixels - 1, 1)), np.nanmedian(variance) * 1e-4)
    read_noise2_initial = overscan_rms ** 2
    bounds = (
        [0.5, (0.1 * overscan_rms) ** 2],
        [1.5, (4.0 * overscan_rms) ** 2],
    )
    initial = [1.0, read_noise2_initial]
    residual = lambda p: (design @ p - variance) / uncertainty
    robust = least_squares(residual, initial, bounds=bounds, loss="soft_l1")
    keep = np.abs(residual(robust.x)) < 5
    if keep.sum() < 4:
        keep[:] = True
    final = least_squares(lambda p: residual(p)[keep], robust.x, bounds=bounds)
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
    """
    Characterise every amplifier represented by a list of raw Flat FITS files.
    
    Pairs consecutive Flats, fits gain/read noise separately for each 2Amp/4Amp amplifier, optionally writes an ECSV calibration table,
    and returns plotting diagnostics. This is an occasional detector-calibration task, not part of the nightly reduction.
    """
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
        overscan_rms = float(np.nanmedian(overscan[(readout, amp)]))
        fit = _fit_gain(data, overscan_rms)
        signal = np.array([p["signal_adu"] for p in data])
        ratio = np.array([p["pair_scale"] for p in data])
        rows.append({
            "ccd": np.int16(ccd),
            "readout_mode": str(readout),
            "amplifier": str(amp),
            "gain_e_per_adu": np.float32(np.round(fit["gain_e_per_adu"], 4)),
            "gain_err_e_per_adu": np.float32(np.round(fit["gain_err_e_per_adu"], 4)),
            "read_noise_adu": np.float32(np.round(fit["read_noise_adu"], 1)),
            "read_noise_err_adu": np.float32(np.round(fit["read_noise_err_adu"], 1)),
            "read_noise_e": np.float32(np.round(fit["read_noise_e"], 1)),
            "overscan_rms_adu": np.float32(np.round(overscan_rms, 1)),
            "n_pairs": np.int16(n_pairs[(readout, amp)]),
            "n_bins": np.int16(len(data)),
            "signal_min_adu": np.int32(signal.min()),
            "signal_max_adu": np.int32(signal.max()),
            "median_pair_scale": np.float32(np.round(np.nanmedian(ratio), 4)),
            "reduced_chi2": np.float32(np.round(fit["reduced_chi2"], 1)),
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


def plot_gain_characterisation(diagnostics, ccd, date, texp, files, output_directory=None):
    """Plot the photon-transfer measurements and fitted relation for each characterised amplifier, optionally saving PNG diagnostics."""
    import matplotlib.pyplot as plt
    output_directory = Path(output_directory) if output_directory else None
    if output_directory:
        output_directory.mkdir(parents=True, exist_ok=True)
    figures = []
    for (readout, amp), d in diagnostics.items():
        order = np.argsort(d["signal_adu"])
        scale = 1 + d["pair_scale"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(d["signal_adu"], d["variance_difference_adu2"] / scale, ".", label="Flat pairs (runs "+str(files[0])+'-'+str(files[-1])+")")
        ax.plot(d["signal_adu"][order], d["model_variance_difference_adu2"][order] / scale[order], label="Fit: "+rf"gain $g="+"{:.4f}".format(d["gain_e_per_adu"])+r" \pm "+f"{d['gain_err_e_per_adu']:.4f}"+r"\,\mathrm{{e^-\,ADU^{-1}}}$"+",\n"+rf"read noise $\sigma_\mathrm{{read}}="+f"{d['read_noise_adu']:.1f}"+r" \pm "+f"{d['read_noise_err_adu']:.1f}"+r"\,\mathrm{{ADU}}$")
        ax.set(xlabel="Mean signal / ADU", ylabel="Difference variance / ADU$^2$",
               title=f"CCD{ccd} {readout} Region {amp} {texp}s")
        ax.legend(loc='upper left')
        figures.append(fig)
        if output_directory:
            fig.savefig(output_directory / f"gain_ccd{ccd}_{readout}_{amp}_{texp}s.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    return figures



# ---- Detector registration relative to the reference night. ----

def phase_correlation_shift(reference, moving, upsample_factor=100):
    """
    Measure the displacement of an image relative to a reference image.

    Phase cross-correlation returns the shift required to align ``moving``
    with ``reference``. Here the sign is reversed so that ``dx`` and ``dy``
    describe the displacement of the moving detector image relative to the
    reference image, in pixels along array axes 0 and 1, respectively.

    Returns
    -------
    dx, dy : float
        Detector displacement relative to the reference image, in pixels.
    error : float
        Registration error returned by ``phase_cross_correlation``.
    """
    shift, error, _ = phase_cross_correlation(
        reference,
        moving,
        upsample_factor=upsample_factor,
        normalization="phase",
    )
    dx, dy = -shift
    return float(dx), float(dy), float(error)


def expected_detector_shifts(night):
    """
    Return historical detector shifts for epochs with calibrated offsets.

    The shifts are given as ``(dx, dy)`` in pixels for CCDs 1--3 relative to
    ``REFERENCE_NIGHT``. They are used only as a fallback when no suitable
    nightly registration exposure is available. ``None`` indicates that no
    historical fallback has been defined for the requested night.
    """
    if night == REFERENCE_NIGHT:
        return {"1": (0, 0), "2": (0, 0), "3": (0, 0)}
    date = int(night)
    if date < 231120:
        return None
    if date <= 240518:
        return {"1": (0, 0), "2": (0, 0), "3": (0.01, -0.01)}
    if date <= 241106:
        return {"1": (-0.89, 7.02), "2": (-3.86, 3.10), "3": (3.08, 1.80)}
    if date <= 250507:
        return {"1": (-0.74, 8.13), "2": (-3.58, 4.06), "3": (3.09, 2.91)}
    if date <= 250823:
        return {"1": (-6.15, 1.11), "2": (-8.75, 2.80), "3": (2.51, 0.22)}
    if date <= 260303:
        return {"1": (-6.08, 1.41), "2": (-8.76, 2.88), "3": (2.72, 0.34)}
    return None


def _registration_candidates(table, ccd):
    """Return SimTh exposures suitable for measuring the detector shift of one CCD."""
    rows = observations.select(table, "SimTh", ccd)
    return rows[~rows["lc_requested"]] if len(rows) else rows


def _reference_registration_image(paths, ccd):
    """Load the overscan-corrected registration image for one CCD on the reference night."""
    filename = observations.raw_fits_path(
        paths,
        REFERENCE_NIGHT,
        REFERENCE_RUNS[ccd],
        ccd,
    )
    if not filename.exists():
        raise FileNotFoundError(f"Reference registration image not found: {filename}")
    return preprocess_image(filename, ccd, config=None).image


def measure_detector_shifts(reduction_input, config, paths):
    """
    Measure nightly CCD shifts relative to the reference night.

    Suitable SimTh exposures are registered against the corresponding
    reference image using phase cross-correlation. For each CCD, the adopted
    shift is the median of all measurements and their scatter is stored as a
    diagnostic. If no registration exposure is available, a historical shift
    is used where defined; otherwise a zero shift is adopted.

    Existing ``detector_shifts.fits`` results are reused unless
    ``config.overwrite`` is set.

    Returns
    -------
    astropy.table.Table
        One row per CCD with ``dx``, ``dy``, their measurement scatter,
        ``n_used``, and a registration ``status``.
    """
    filename = paths.detector_shifts
    if filename.exists() and not config.overwrite:
        table = Table.read(filename)
        logger.info("Loaded cached detector shifts from %s", filename)
        if config.diagnostics != "none":
            diagnostics.plot_detector_shifts(table, paths.figures / "detector_shifts.png")
        return table

    if config.night == REFERENCE_NIGHT:
        table = Table(rows=[
            {
                "ccd": np.int16(ccd),
                "dx": np.float64(0.0),
                "dy": np.float64(0.0),
                "dx_scatter": np.float64(0.0),
                "dy_scatter": np.float64(0.0),
                "n_used": np.int16(0),
                "status": "reference",
            }
            for ccd in (1, 2, 3)
        ])
        table.write(filename, overwrite=True)
        logger.info("Reference night 001122: detector shifts are (0, 0) on all CCDs")
        if config.diagnostics != "none":
            diagnostics.plot_detector_shifts(table, paths.figures / "detector_shifts.png")
        return table

    expected, rows = expected_detector_shifts(config.night), []
    for ccd in ("1", "2", "3"):
        measurements = []
        candidates = _registration_candidates(reduction_input, ccd)
        if len(candidates):
            reference = _reference_registration_image(paths, ccd)
            for row in candidates:
                moving = preprocess_image(row[f"file_ccd{ccd}"], ccd, config).image
                measurements.append(phase_correlation_shift(reference, moving))

        if measurements:
            dxs = np.array([measurement[0] for measurement in measurements])
            dys = np.array([measurement[1] for measurement in measurements])
            dx, dy = float(np.nanmedian(dxs)), float(np.nanmedian(dys))
            dx_scatter = float(np.nanstd(dxs)) if len(dxs) > 1 else np.nan
            dy_scatter = float(np.nanstd(dys)) if len(dys) > 1 else np.nan
            status = (
                "good"
                if max(np.nan_to_num(dx_scatter), np.nan_to_num(dy_scatter)) <= 0.1
                else "large scatter"
            )
        elif expected is not None:
            dx, dy = expected[ccd]
            dx_scatter = dy_scatter = np.nan
            status = "historical fallback"
        else:
            dx = dy = 0.0
            dx_scatter = dy_scatter = np.nan
            status = "zero fallback"

        rows.append({
            "ccd": np.int16(ccd),
            "dx": np.float64(dx),
            "dy": np.float64(dy),
            "dx_scatter": np.float64(dx_scatter),
            "dy_scatter": np.float64(dy_scatter),
            "n_used": np.int16(len(measurements)),
            "status": status,
        })
        logger.info(
            "CCD%s detector shift: dx=%+.2f, dy=%+.2f px (%s; %d measurements)",
            ccd,
            dx,
            dy,
            status,
            len(measurements),
        )
        if status != "good":
            logger.warning("CCD%s detector registration status: %s", ccd, status)

    shifts = Table(rows=rows)
    shifts.write(filename, overwrite=True)
    if config.diagnostics != "none":
        diagnostics.plot_detector_shifts(shifts, paths.figures / "detector_shifts.png")
    return shifts


def detector_shift(shifts, ccd):
    """Return the measured ``(dx, dy)`` shift for one CCD, or zero if absent."""
    use = np.asarray(shifts["ccd"]) == int(ccd)
    if not np.any(use):
        return 0.0, 0.0
    return float(shifts[use][0]["dx"]), float(shifts[use][0]["dy"])
