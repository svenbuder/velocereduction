from dataclasses import dataclass
from enum import IntFlag
import numpy as np
from astropy.table import Table, vstack
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import ndtr

from .constants import SCIENCE_FIBRES
from . import utils


class PeakFlag(IntFlag):
    GOOD = 0
    LOW_SNR = 1
    SATURATED = 2
    BAD_FIT = 4
    BAD_WIDTH = 8
    BLENDED = 16


@dataclass
class PeakConfig:
    detection_snr: float = 5.0
    prominence_snr: float = 3.0
    minimum_peak_distance: int = 2
    background_window: int = 31
    noise_window: int = 15
    fit_half_width: int = 4
    minimum_sigma: float = 0.25
    maximum_sigma: float = 3.0
    minimum_fit_snr: float = 8.0
    saturation: float = 65000.0


def pixel_integrated_gaussian(y, integrated_counts, centre, sigma):
    y = np.asarray(y, float)
    return integrated_counts * (ndtr((y + 0.5 - centre) / sigma) - ndtr((y - 0.5 - centre) / sigma))


def line_model(y, integrated_counts, centre, sigma, background, slope, reference):
    return background + slope * (y - reference) + pixel_integrated_gaussian(y, integrated_counts, centre, sigma)


def detect_peaks(counts, variance=None, config=None):
    config = config or PeakConfig(); counts = np.asarray(counts, float)
    finite = np.isfinite(counts)
    if not np.any(finite):
        return np.array([], int), np.full_like(counts, np.nan), np.full_like(counts, np.nan)
    working = np.where(finite, counts, np.nanmedian(counts[finite]))
    background = median_filter(working, config.background_window, mode="nearest")
    signal = working - background
    if variance is not None:
        noise = np.sqrt(np.clip(np.asarray(variance, float), 0, None))
        fallback = max(utils.robust_sigma(signal), 1e-6)
        noise = np.where(np.isfinite(noise) & (noise > 0), noise, fallback)
    else:
        local = median_filter(signal, config.noise_window, mode="nearest")
        noise = 1.4826 * median_filter(np.abs(signal - local), config.noise_window, mode="nearest")
        noise = np.maximum(noise, max(utils.robust_sigma(signal), 1e-6))
    snr = np.divide(signal, noise, out=np.zeros_like(signal), where=np.isfinite(noise) & (noise > 0))
    peaks, _ = find_peaks(snr, height=config.detection_snr, prominence=config.prominence_snr, distance=config.minimum_peak_distance)
    return peaks.astype(int), background, snr


def fit_peak(counts, candidate, variance=None, background=None, config=None):
    config = config or PeakConfig(); counts = np.asarray(counts, float)
    left, right = max(0, candidate - config.fit_half_width), min(len(counts), candidate + config.fit_half_width + 1)
    y, observed = np.arange(left, right, dtype=float), counts[left:right]
    good = np.isfinite(observed)
    if variance is not None:
        local_var = np.asarray(variance, float)[left:right]
        good &= np.isfinite(local_var) & (local_var > 0)
    else:
        scale = max(utils.robust_sigma(observed[good]), 1.0)
        local_var = np.full_like(observed, scale ** 2)
    if good.sum() < 5:
        return None

    bg = float(background[candidate]) if background is not None and np.isfinite(background[candidate]) else float(np.nanmedian(observed))
    signal = np.maximum(observed - bg, 0)
    p0 = [max(float(np.nansum(signal)), 1), float(candidate), 0.8, bg, 0.0]
    lower = [0, candidate - 1, config.minimum_sigma, -np.inf, -np.inf]
    upper = [np.inf, candidate + 1, config.maximum_sigma, np.inf, np.inf]

    def residual(p):
        model = line_model(y, *p, reference=float(candidate))
        return (model[good] - observed[good]) / np.sqrt(local_var[good])

    robust = least_squares(residual, p0, bounds=(lower, upper), loss="soft_l1")
    fit = least_squares(residual, robust.x, bounds=(lower, upper))
    dof = max(good.sum() - len(fit.x), 1)
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * np.sum(fit.fun ** 2) / dof
    uncertainty = float(np.sqrt(max(covariance[1, 1], 0)))
    model = line_model(y, *fit.x, reference=float(candidate))
    rms = float(np.sqrt(np.nanmean((observed[good] - model[good]) ** 2)))
    noise = float(np.sqrt(np.nanmedian(local_var[good])))
    snr = fit.x[0] / max(noise * np.sqrt(2 * np.pi) * fit.x[2], 1e-12)
    flag = PeakFlag.GOOD
    if snr < config.minimum_fit_snr: flag |= PeakFlag.LOW_SNR
    if np.nanmax(observed[good]) >= config.saturation: flag |= PeakFlag.SATURATED
    if fit.x[2] <= config.minimum_sigma * 1.01 or fit.x[2] >= config.maximum_sigma * 0.99: flag |= PeakFlag.BAD_WIDTH
    if not fit.success or not np.isfinite(uncertainty): flag |= PeakFlag.BAD_FIT
    return {
        "y": float(fit.x[1]), "y_uncertainty": uncertainty, "integrated_counts": float(fit.x[0]),
        "sigma": float(fit.x[2]), "fwhm": float(2.354820045 * fit.x[2]), "background": float(fit.x[3]),
        "background_slope": float(fit.x[4]), "signal_to_noise": float(snr), "fit_rms": rms,
        "quality_flag": int(flag), "used_for_wavelength_fit": flag == PeakFlag.GOOD,
    }


def _flag_blends(table, minimum_distance=1.2):
    if len(table) < 2: return
    for order in np.unique(table["order"]):
        idx = np.where(np.asarray(table["order"]) == order)[0]
        sort = idx[np.argsort(np.asarray(table["y"])[idx])]
        for j in np.where(np.diff(np.asarray(table["y"])[sort]) < minimum_distance)[0]:
            for k in (sort[j], sort[j + 1]):
                table["quality_flag"][k] |= int(PeakFlag.BLENDED); table["used_for_wavelength_fit"][k] = False


def measure_spectrum(counts, orders, variance=None, ccd="", run="", mjd_mid=np.nan, kind="", fibre="summed", config=None):
    config = config or PeakConfig(); counts = np.asarray(counts, float)
    if counts.ndim != 2:
        raise ValueError("counts must have shape (n_orders, n_pixels)")
    variance = None if variance is None else np.asarray(variance, float)
    rows = []
    for i, order in enumerate(orders):
        var = None if variance is None else variance[i]
        candidates, background, _ = detect_peaks(counts[i], var, config)
        for candidate in candidates:
            fit = fit_peak(counts[i], int(candidate), var, background, config)
            if fit is not None:
                rows.append({
                    "ccd": int(ccd), "order": int(order), "run": str(run), "mjd_mid": float(mjd_mid),
                    "calibration_type": str(kind), "fibre": str(fibre), **fit,
                })
    table = Table(rows=rows)
    if len(table): _flag_blends(table)
    return table


def measure_extracted_exposure(exposure, peak_config=None, fibres=False):
    base = measure_spectrum(
        exposure.result.flux.T, exposure.orders, exposure.result.variance.T,
        exposure.ccd, exposure.run, exposure.mjd_mid, exposure.kind, "summed", peak_config,
    )
    if not fibres or exposure.fibre_flux is None:
        return base
    tables = [base]
    for j, fibre in enumerate(SCIENCE_FIBRES):
        tables.append(measure_spectrum(
            exposure.fibre_flux[:, :, j], exposure.orders, exposure.fibre_variance[:, :, j],
            exposure.ccd, exposure.run, exposure.mjd_mid, exposure.kind, fibre, peak_config,
        ))
    nonempty = [table for table in tables if len(table)]
    return vstack(nonempty, metadata_conflicts="silent") if nonempty else Table()
