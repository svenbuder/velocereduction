import logging
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.special import erf

from velocereduction import detector, tramlines, utils
from velocereduction.models import ExtractionResult, FibreGeometry

logger = logging.getLogger(__name__)

# Canonical Veloce slit geometry. Local utils.slit_order wins if present.
SLIT_ORDER = getattr(utils, "slit_order", [
    "ThXe", "S5", "S2", "Blank",
    7, 18, 17, 6, 16, 15, 5, 14, 13, 1, 12, 11, 4, 10, 9, 3, 8, 19, 2,
    "Blank", "S4", "S3", "S1", "LC",
])
FIBRE_COMPONENTS = tuple(
    v for v in SLIT_ORDER
    if isinstance(v, int) or (isinstance(v, str) and v.startswith("S"))
)
SCIENCE_COMPONENTS = tuple(v for v in FIBRE_COMPONENTS if isinstance(v, int))
SKY_COMPONENTS = tuple(v for v in FIBRE_COMPONENTS if isinstance(v, str) and v.startswith("S"))
CENTRAL_SLIT_INDEX = SLIT_ORDER.index(1)
FIBRE_SLOTS = np.array(
    [SLIT_ORDER.index(v) - CENTRAL_SLIT_INDEX for v in FIBRE_COMPONENTS],
    dtype=float,
)


def aperture_weights(m, begin, end):
    """Return fractional overlap of cross-dispersion pixels with an aperture.

    Pixel ``m`` spans [m-0.5, m+0.5], so edge pixels can contribute fractions
    between zero and one instead of being included/excluded as Boolean pixels.
    """
    m = np.asarray(m, dtype=float)
    return np.clip(
        np.minimum(m + 0.5, end) - np.maximum(m - 0.5, begin),
        0.0, 1.0,
    )


def extract_apertures(order, variance, m, apertures):
    """Extract one or more continuous apertures with variance propagation.

    Flux uses ``sum(w * D)`` and variance uses ``sum(w**2 * V)``.  Pixels with
    non-finite data or variance are ignored; arrays remain dispersion-first.
    """
    order, variance = np.asarray(order, float), np.asarray(variance, float)
    flux = np.full((order.shape[0], len(apertures)), np.nan)
    var = np.full_like(flux, np.nan)

    for j, (_, begin, end) in enumerate(apertures):
        w = aperture_weights(m, float(begin), float(end))
        good = np.isfinite(order) & np.isfinite(variance)
        flux[:, j] = np.nansum(np.where(good, order * w[None, :], np.nan), axis=1)
        var[:, j] = np.nansum(
            np.where(good, variance * w[None, :] ** 2, np.nan),
            axis=1,
        )

    return ExtractionResult(
        flux=flux,
        variance=var,
        components=tuple(a[0] for a in apertures),
        extraction_mode="summed",
    )


def extract_summed_order(order, variance, row, regions=("Science", "Sky_1", "Sky_2")):
    """Extract the robust summed Science/Sky apertures from one order matrix."""
    half = int(row["extraction_half_window"])
    m = np.arange(-half, half + 1, dtype=float)
    apertures = [
        (r, row[f"{r}_begin"], row[f"{r}_end"])
        for r in regions
        if np.isfinite(row[f"{r}_begin"]) and np.isfinite(row[f"{r}_end"])
    ]
    return extract_apertures(order, variance, m, apertures)


def _integrated_gaussians(m, centres, sigma, normalise=True):
    """Evaluate Gaussian fibre profiles integrated over detector-pixel areas.

    Returns a matrix with shape (cross-dispersion pixel, fibre).  Integrating
    across pixel boundaries is important because the Veloce fibre profiles are
    only sampled by a few detector pixels.
    """
    m = np.asarray(m, float)[:, None]
    centres = np.asarray(centres, float)[None, :]
    s = np.sqrt(2.0) * float(sigma)
    p = 0.5 * (
        erf((m + 0.5 - centres) / s)
        - erf((m - 0.5 - centres) / s)
    )
    if normalise:
        norm = p.sum(axis=0)
        p[:, norm > 0] /= norm[norm > 0]
    return p


def _gaussian_model(m, background, amplitudes, centres, sigma):
    """Return a constant background plus the sum of pixel-integrated fibres."""
    return background + _integrated_gaussians(m, centres, sigma) @ amplitudes


def _fit_regular_profile(
    profile,
    m,
    slots,
    separation0=2.4,
    sigma0=0.85,
    label=None,
):
    """Fit a regular fibre grid to a collapsed cross-dispersion Flat profile.

    This first-pass fit solves for background, bundle centre, common separation,
    common Gaussian sigma, and one non-negative amplitude per illuminated fibre.
    It provides the starting geometry for the more flexible fibre-centre fit.
    """
    background0 = float(np.nanmedian(np.r_[profile[:8], profile[-8:]]))
    centres0 = separation0 * slots
    profile_scale = np.nanmax(profile) - background0
    amplitudes0 = np.maximum(
        np.interp(centres0, m, profile) - background0,
        1e-3 * profile_scale,
    )
    p0 = np.r_[background0, 0.0, separation0, sigma0, amplitudes0]

    def residual(p):
        centres = p[1] + p[2] * slots
        return _gaussian_model(m, p[0], p[4:], centres, p[3]) - profile

    lower = np.r_[-np.inf, -1.5, 1.8, 0.3, np.zeros(len(slots))]
    upper = np.r_[np.inf, 1.5, 3.0, 1.8, np.full(len(slots), np.inf)]
    fit = least_squares(
        residual, p0, bounds=(lower, upper), max_nfev=30000
    )

    logger.debug(
        "%sregular fibre fit: initial d=%.3f sigma=%.3f -> "
        "d=%.3f sigma=%.3f; cost=%.3g; nfev=%d; success=%s",
        f"{label}: " if label else "",
        separation0, sigma0, fit.x[2], fit.x[3],
        fit.cost, fit.nfev, fit.success,
    )
    return fit


def fit_collapsed_fibre_profile(
    extracted_flat,
    separation0=2.4,
    sigma0=0.85,
    maximum_centre_shift=0.4,
    label=None,
):
    """Fit the mean fibre geometry from the Flat collapsed along dispersion.

    A regular-grid fit supplies the initial separation and width, after which
    individual fibre centres may move by ``maximum_centre_shift`` pixels.  The
    resulting small fibre-specific offsets are then kept fixed in local fits.
    """
    profile = np.nanmedian(np.asarray(extracted_flat, float), axis=0)
    m = np.arange(profile.size) - (profile.size - 1) / 2
    first = _fit_regular_profile(
        profile, m, FIBRE_SLOTS,
        separation0=separation0, sigma0=sigma0, label=label,
    )

    background, x0, separation, sigma = first.x[:4]
    amplitudes = first.x[4:]
    regular_centres = x0 + separation * FIBRE_SLOTS
    n = len(FIBRE_SLOTS)
    p0 = np.r_[background, sigma, amplitudes, regular_centres]

    def residual(p):
        amplitudes = p[2:2+n]
        centres = p[2+n:]
        return _gaussian_model(m, p[0], amplitudes, centres, p[1]) - profile

    lower = np.r_[
        -np.inf, 0.3, np.zeros(n),
        regular_centres - maximum_centre_shift,
    ]
    upper = np.r_[
        np.inf, 1.8, np.full(n, np.inf),
        regular_centres + maximum_centre_shift,
    ]
    fit = least_squares(
        residual, p0, bounds=(lower, upper), max_nfev=50000
    )

    centres = fit.x[2+n:]
    separation, x0 = np.polyfit(FIBRE_SLOTS, centres, 1)
    offsets = centres - (x0 + separation * FIBRE_SLOTS)

    rms = float(np.sqrt(np.nanmean(residual(fit.x) ** 2)))
    logger.debug(
        "%scollapsed fibre profile: d=%.3f px; sigma=%.3f px; "
        "centre=%.3f px; RMS=%.2f counts; max fibre offset=%.3f px",
        f"{label}: " if label else "",
        separation, fit.x[1], x0, rms,
        float(np.nanmax(np.abs(offsets))),
    )
    return {
        "profile": profile,
        "m": m,
        "background": fit.x[0],
        "sigma": fit.x[1],
        "amplitudes": fit.x[2:2+n],
        "centres": centres,
        "x0": x0,
        "separation": separation,
        "fibre_offsets": offsets,
        "rms": rms,
    }


def _fit_flat_row(profile, m, trace_offset, collapsed):
    """Fit local bundle centre, separation, and width at one dispersion sample.

    Fibre-specific offsets measured from the collapsed profile remain fixed, so
    local fits only track smooth changes of the whole fibre bundle.
    """
    slots, offsets = FIBRE_SLOTS, collapsed["fibre_offsets"]
    n = len(slots)
    background0 = float(np.nanmedian(np.r_[profile[:6], profile[-6:]]))
    x00 = trace_offset + collapsed["x0"]
    separation0, sigma0 = collapsed["separation"], collapsed["sigma"]
    centres0 = x00 + separation0 * slots + offsets
    profile_scale = np.nanmax(profile) - background0
    amplitudes0 = np.maximum(
        np.interp(centres0, m, profile) - background0,
        1e-3 * profile_scale,
    )
    p0 = np.r_[background0, x00, separation0, sigma0, amplitudes0]

    def residual(p):
        centres = p[1] + p[2] * slots + offsets
        return _gaussian_model(m, p[0], p[4:], centres, p[3]) - profile

    lower = np.r_[
        -np.inf, x00 - 0.6, separation0 - 0.3,
        max(0.3, sigma0 - 0.35), np.zeros(n),
    ]
    upper = np.r_[
        np.inf, x00 + 0.6, separation0 + 0.3,
        min(1.8, sigma0 + 0.35), np.full(n, np.inf),
    ]
    fit = least_squares(
        residual, p0, bounds=(lower, upper), max_nfev=10000
    )
    return fit.x[1], fit.x[2], fit.x[3]


def _robust_poly(x, y, x_all, degree=3, clip=4.0, iterations=4):
    """Robustly interpolate sparse geometry samples with an iteratively clipped polynomial."""
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() == 0:
        return np.full_like(x_all, np.nan, dtype=float)

    degree = min(degree, max(0, good.sum() - 1))
    if degree == 0:
        return np.full_like(x_all, np.nanmedian(y[good]), dtype=float)

    use = good.copy()
    for _ in range(iterations):
        coeff = np.polyfit(x[use], y[use], degree)
        resid = y - np.polyval(coeff, x)
        centre = np.nanmedian(resid[use])
        sigma = 1.4826 * np.nanmedian(np.abs(resid[use] - centre))
        if not np.isfinite(sigma) or sigma == 0:
            break
        new_use = good & (np.abs(resid - centre) < clip * sigma)
        if new_use.sum() <= degree:
            break
        use = new_use

    return np.polyval(np.polyfit(x[use], y[use], degree), x_all)


def fit_fibre_geometry(
    extracted_flat,
    trace_offset,
    sample_step=16,
    sample_half_width=4,
    polynomial_degree=3,
    separation0=None,
    sigma0=None,
    label=None,
):
    """Fit smoothly varying fibre separation, width, and bundle position."""

    if separation0 is None or sigma0 is None:
        if label is not None and "ccd_1_" in label:
            default_separation = 2.40
            default_sigma = 1.20
        elif label is not None and "ccd_2_" in label:
            default_separation = 2.35
            default_sigma = 0.85
        elif label is not None and "ccd_3_" in label:
            default_separation = 2.10
            default_sigma = 0.70
        else:
            default_separation = 2.20
            default_sigma = 0.8

        if separation0 is None:
            separation0 = default_separation
        if sigma0 is None:
            sigma0 = default_sigma

    logger.debug(
        "%s fibre geometry initial guess: separation=%.3f px, sigma=%.3f px",
        label or "order",
        separation0,
        sigma0,
    )

    extracted_flat = np.asarray(extracted_flat, float)
    trace_offset = np.asarray(trace_offset, float)

    collapsed = fit_collapsed_fibre_profile(
        extracted_flat,
        separation0=separation0,
        sigma0=sigma0,
        label=label,
    )

    nx, nm = extracted_flat.shape
    m = np.arange(nm) - (nm - 1) / 2
    sampled_x = np.arange(
        sample_half_width, nx - sample_half_width, sample_step, dtype=int
    )
    x0s = np.full(sampled_x.size, np.nan)
    seps = np.full_like(x0s, np.nan)
    sigmas = np.full_like(x0s, np.nan)

    low_signal = 0
    failed = 0
    bound_like = 0
    for j, x in enumerate(sampled_x):
        # Median a short section along dispersion to improve S/N without
        # allowing the geometry to vary appreciably within the sample.
        profile = np.nanmedian(
            extracted_flat[x-sample_half_width:x+sample_half_width+1],
            axis=0,
        )
        edge = np.nanmedian(np.r_[profile[:6], profile[-6:]])
        peak = np.nanmax(profile)
        contrast = peak - edge
        if (
            not np.isfinite(contrast)
            or contrast <= 0
            or contrast / max(abs(peak), 1e-12) < 0.05
        ):
            continue
        try:
            x0s[j], seps[j], sigmas[j] = _fit_flat_row(
                profile, m, trace_offset[x], collapsed
            )
            # A result very close to a local bound is useful debug information:
            # it can identify an order trapped on an incorrect solution branch.
            if (
                abs(seps[j] - (collapsed["separation"] - 0.3)) < 0.01
                or abs(seps[j] - (collapsed["separation"] + 0.3)) < 0.01
                or abs(sigmas[j] - max(0.3, collapsed["sigma"] - 0.35)) < 0.01
                or abs(sigmas[j] - min(1.8, collapsed["sigma"] + 0.35)) < 0.01
            ):
                bound_like += 1
        except (ValueError, np.linalg.LinAlgError):
            failed += 1
            continue

    bundle_samples = x0s - trace_offset[sampled_x]
    x_all = np.arange(nx, dtype=float)
    bundle = _robust_poly(
        sampled_x, bundle_samples, x_all, polynomial_degree
    )
    separation = _robust_poly(
        sampled_x, seps, x_all, polynomial_degree
    )
    sigma = _robust_poly(
        sampled_x, sigmas, x_all, polynomial_degree
    )

    centres = (
        (trace_offset + bundle)[:, None]
        + separation[:, None] * FIBRE_SLOTS[None, :]
        + collapsed["fibre_offsets"][None, :]
    )

    valid = np.isfinite(seps) & np.isfinite(sigmas)
    if valid.any():
        logger.debug(
            "%sfibre geometry: local fits %d/%d successful "
            "(low signal=%d, failed=%d, near bounds=%d); "
            "d median=%.3f [%.3f, %.3f] px; "
            "sigma median=%.3f [%.3f, %.3f] px",
            f"{label}: " if label else "",
            int(valid.sum()), len(sampled_x), low_signal, failed, bound_like,
            float(np.nanmedian(seps[valid])),
            float(np.nanpercentile(seps[valid], 16)),
            float(np.nanpercentile(seps[valid], 84)),
            float(np.nanmedian(sigmas[valid])),
            float(np.nanpercentile(sigmas[valid], 16)),
            float(np.nanpercentile(sigmas[valid], 84)),
        )
    else:
        logger.warning("%sno usable local fibre-geometry fits", f"{label}: " if label else "")

    return FibreGeometry(
        FIBRE_COMPONENTS,
        FIBRE_SLOTS.copy(),
        centres,
        sigma,
        separation,
        bundle,
        collapsed["fibre_offsets"].copy(),
        sampled_x,
        sigmas,
        seps,
        bundle_samples,
    )


def extract_fibre_order(
    order,
    variance,
    geometry,
    fit_background=True,
    return_covariance=False,
    label=None,
):
    """Simultaneously deblend all illuminated fibres in one detector order.

    At each dispersion pixel the Flat-derived geometry is held fixed and only
    fibre amplitudes (plus an optional constant background) are solved by
    weighted linear least squares.  This prevents noisy science data from
    perturbing the fibre positions or widths.  The covariance matrix can be
    retained because neighbouring deblended fibre amplitudes are correlated.
    """
    order, variance = np.asarray(order, float), np.asarray(variance, float)
    nx, nm = order.shape
    m = np.arange(nm) - (nm - 1) / 2
    nf = len(geometry.components)

    flux = np.full((nx, nf), np.nan)
    var = np.full_like(flux, np.nan)
    
    background = (
        np.full(nx, np.nan)
        if fit_background
        else None
    )

    covariance = (
        np.full((nx, nf, nf), np.nan)
        if return_covariance else None
    )

    debug = logger.isEnabledFor(logging.DEBUG)
    condition_numbers = []
    skipped = 0

    for x in range(nx):
        good = (
            np.isfinite(order[x])
            & np.isfinite(variance[x])
            & (variance[x] > 0)
        )
        if good.sum() < nf + int(fit_background):
            skipped += 1
            continue

        # Build the fixed profile matrix from the smooth Flat geometry.
        P = _integrated_gaussians(
            m, geometry.centres[x], geometry.sigma[x]
        )
        A = (
            np.column_stack((P, np.ones(nm)))
            if fit_background else P
        )

        # Weight every detector pixel by its propagated standard deviation.
        inv_sigma = 1.0 / np.sqrt(variance[x, good])
        Aw = A[good] * inv_sigma[:, None]
        dw = order[x, good] * inv_sigma

        if (
            not np.all(np.isfinite(Aw))
            or not np.all(np.isfinite(dw))
        ):
            skipped += 1
            continue

        try:
            coeff, *_ = np.linalg.lstsq(Aw, dw, rcond=None)
            cov = np.linalg.pinv(Aw.T @ Aw)[:nf, :nf]
        except np.linalg.LinAlgError:
            skipped += 1
            logger.debug(
                "%sfibre extraction row %d skipped: linear solve failed",
                f"{label}: " if label else "",
                x,
            )
            continue

        flux[x] = coeff[:nf]
        var[x] = np.diag(cov)
        if fit_background:
            background[x] = coeff[nf]
        if return_covariance:
            covariance[x] = cov

        # Condition number is an excellent debug diagnostic for strongly
        # blended/degenerate fibre solutions, but need not be computed always.
        if debug and x % 64 == 0:
            condition_numbers.append(np.linalg.cond(Aw))

    valid_rows = np.count_nonzero(np.all(np.isfinite(flux), axis=1))
    if debug:
        finite_cond = np.asarray(condition_numbers, float)
        finite_cond = finite_cond[np.isfinite(finite_cond)]
        logger.debug(
            "%sfibre extraction: valid rows %d/%d; skipped=%d; "
            "median matrix condition=%.2e; max=%.2e",
            f"{label}: " if label else "",
            valid_rows, nx, skipped,
            float(np.nanmedian(finite_cond)) if finite_cond.size else np.nan,
            float(np.nanmax(finite_cond)) if finite_cond.size else np.nan,
        )

    return ExtractionResult(
        flux=flux,
        variance=var,
        components=geometry.components,
        extraction_mode="fibre",
        covariance=covariance,
        background=background,
    )


def recombine_fibres(result, components=SCIENCE_COMPONENTS):
    """Recombine selected fibre amplitudes with correct uncertainty propagation.

    If the simultaneous fibre fit retained covariance, off-diagonal terms are
    included through ``1.T @ C @ 1`` rather than simply summing variances.
    """
    if result.extraction_mode != "fibre":
        raise ValueError("recombine_fibres requires a fibre ExtractionResult")

    idx = result.component_indices(components)
    flux = np.sum(result.flux[:, idx], axis=1)

    if result.covariance is None:
        variance = np.sum(result.variance[:, idx], axis=1)
    else:
        covariance = result.covariance[:, idx][:, :, idx]
        variance = np.sum(covariance, axis=(1, 2))

    return flux, variance


def _smooth_1d(values, sigma=50.0):
    """Gaussian-smooth a 1D sequence while normalising around missing samples."""
    values = np.asarray(values, float)
    good = np.isfinite(values)
    filled = np.where(good, values, 0.0)
    weights = gaussian_filter1d(good.astype(float), sigma, mode="nearest")
    smooth = gaussian_filter1d(filled, sigma, mode="nearest")
    result = np.full_like(values, np.nan)
    use = weights > 0.05
    result[use] = smooth[use] / weights[use]
    return result


def fibre_recombination_qa(
    summed,
    fibre,
    summed_component="Science",
    smooth_sigma=50.0,
    label=None,
):
    """Compare recombined science fibres with the direct summed extraction.

    The direct summed spectrum remains the robust spectral-shape product.  This
    diagnostic measures wavelength-dependent structure introduced by fibre
    deblending; a constant scale difference is removed with a broad smooth ratio.
    """
    if summed_component not in summed.components:
        raise KeyError(f"Summed component {summed_component!r} is unavailable")

    j = summed.components.index(summed_component)
    direct_flux = np.asarray(summed.flux[:, j], float)
    direct_variance = np.asarray(summed.variance[:, j], float)
    recombined_flux, recombined_variance = recombine_fibres(
        fibre, SCIENCE_COMPONENTS
    )

    ratio = np.full_like(direct_flux, np.nan)
    good = (
        np.isfinite(direct_flux)
        & np.isfinite(recombined_flux)
        & (direct_flux != 0)
    )
    ratio[good] = recombined_flux[good] / direct_flux[good]

    ratio_smooth = _smooth_1d(ratio, sigma=smooth_sigma)
    structure = np.full_like(ratio, np.nan)
    good_structure = good & np.isfinite(ratio_smooth) & (ratio_smooth != 0)
    structure[good_structure] = ratio[good_structure] / ratio_smooth[good_structure] - 1.0

    finite = structure[np.isfinite(structure)]
    robust_rms = (
        1.4826 * np.median(np.abs(finite - np.median(finite)))
        if finite.size else np.nan
    )
    median_ratio = float(np.nanmedian(ratio[good])) if good.any() else np.nan
    p95_structure = (
        float(np.nanpercentile(np.abs(finite), 95)) if finite.size else np.nan
    )
    logger.debug(
        "%sfibre/summed QA: median ratio=%.5f; "
        "small-scale robust RMS=%.4f%%; |structure| p95=%.4f%%",
        f"{label}: " if label else "",
        median_ratio, 100.0 * robust_rms, 100.0 * p95_structure,
    )

    return {
        "direct_flux": direct_flux,
        "direct_variance": direct_variance,
        "recombined_flux": recombined_flux,
        "recombined_variance": recombined_variance,
        "ratio": ratio,
        "ratio_smooth": ratio_smooth,
        "fractional_structure": structure,
        "robust_rms_fractional_structure": float(robust_rms),
    }


def extract_order(
    order,
    variance,
    row,
    extraction_mode,
    fibre_geometry=None,
):
    """Dispatch one order to either robust summed or fibre-resolved extraction."""
    if extraction_mode == "summed":
        return extract_summed_order(order, variance, row)
    if extraction_mode == "fibre":
        if fibre_geometry is None:
            raise ValueError(
                "fibre_geometry is required for extraction_mode='fibre'"
            )
        return extract_fibre_order(order, variance, fibre_geometry)
    raise ValueError(f"Unknown extraction mode: {extraction_mode}")


def smooth_fibre_flat(fibre_flux, sigma=50.0):
    """Smooth each extracted Flat fibre only along the dispersion direction."""
    data = np.asarray(fibre_flux, float)
    result = np.full_like(data, np.nan)
    for j in range(data.shape[1]):
        good = np.isfinite(data[:, j])
        values = np.where(good, data[:, j], 0.0)
        weights = gaussian_filter1d(
            good.astype(float), sigma, mode="nearest"
        )
        smooth = gaussian_filter1d(
            values, sigma, mode="nearest"
        )
        ok = weights > 0.05
        result[ok, j] = smooth[ok] / weights[ok]
    return result


def save_fibre_geometries(filename, geometries, config=None):
    """Write smooth fibre geometry and component metadata for all orders to FITS."""
    primary = fits.PrimaryHDU()
    primary.header["PRODUCT"] = "FIBRE_GEOMETRY"
    if config is not None:
        primary.header["NIGHT"] = config.night
    hdus = [primary]

    for order, g in geometries.items():
        columns = [
            fits.Column(
                name="X", format="J",
                array=np.arange(g.centres.shape[0]),
            ),
            fits.Column(
                name="SIGMA", format="E",
                array=g.sigma.astype(np.float32),
            ),
            fits.Column(
                name="SEPARATION", format="E",
                array=g.separation.astype(np.float32),
            ),
            fits.Column(
                name="BUNDLE", format="E",
                array=g.bundle_offset.astype(np.float32),
            ),
        ]
        for i in range(len(g.components)):
            columns.append(
                fits.Column(
                    name=f"CENTRE_{i:02d}",
                    format="E",
                    array=g.centres[:, i].astype(np.float32),
                )
            )

        hdu = fits.BinTableHDU.from_columns(
            columns, name=str(order)
        )
        hdu.header["NFIBRE"] = len(g.components)
        for i, component in enumerate(g.components):
            hdu.header[f"FIB{i:02d}"] = str(component)
            hdu.header[f"SLOT{i:02d}"] = float(g.slots[i])
            hdu.header[f"OFF{i:02d}"] = float(g.fibre_offsets[i])
        hdus.append(hdu)

    fits.HDUList(hdus).writeto(filename, overwrite=True)



def _apply_response_flat(order, variance, response_flat):
    """Apply detector-response correction and propagate variance by response squared."""
    if response_flat is None:
        return order, variance
    response = np.asarray(response_flat, float)
    valid = np.isfinite(response) & (response > 0)
    corrected = np.full_like(np.asarray(order, float), np.nan)
    corrected_var = np.full_like(np.asarray(variance, float), np.nan)
    corrected[valid] = np.asarray(order, float)[valid] / response[valid]
    corrected_var[valid] = np.asarray(variance, float)[valid] / response[valid] ** 2
    return corrected, corrected_var


def _extract_region_for_order(image, variance_image, row, region, response_flat=None):
    """Extract one named calibration aperture from a detector image and variance."""
    order, m, _ = tramlines.extract_order_matrix(
        image, row, return_trace_offset=True
    )
    var, _, _ = tramlines.extract_order_matrix(
        variance_image, row, return_trace_offset=True
    )
    order, var = _apply_response_flat(order, var, response_flat)
    name = "Science" if region == "FibTh" else region
    result = extract_apertures(
        order,
        var,
        m,
        [(name, row[f"{name}_begin"], row[f"{name}_end"])],
    )
    return result.flux[:, 0], result.variance[:, 0]


def extract_calibration_spectra(
    reduction_input,
    nightly_tramlines,
    flat_products,
    config,
    paths,
):
    """Extract summed calibration spectra and optional per-fibre FibTh spectra.

    Summed calibration products remain the absolute wavelength-calibration
    reference; fibre-resolved FibTh spectra are retained for relative
    fibre-dependent wavelength corrections.
    """
    output = {
        "SimLC": {"2": [], "3": []},
        "SimTh": {"1": [], "2": [], "3": []},
        "FibTh": {"1": [], "2": [], "3": []},
    }

    for caltype in ("SimLC", "SimTh", "FibTh"):
        for obs in reduction_input[
            reduction_input["type"] == caltype
        ]:
            if not obs["use"]:
                continue

            for ccd in ("1", "2", "3"):
                if (
                    ccd not in output[caltype]
                    or not obs[f"use_ccd{ccd}"]
                ):
                    continue

                filename = obs[f"file_ccd{ccd}"]
                if filename is None:
                    continue

                frame = detector.preprocess_image(
                    filename,
                    ccd=ccd,
                    config=config,
                )

                image = frame.image
                variance_image = frame.variance
                orders = []
                summed = []
                summed_var = []
                fibre_counts = []
                fibre_var = []

                for row in nightly_tramlines:
                    order_name = str(row["order_name"])
                    if f"ccd_{ccd}_" not in order_name:
                        continue

                    orders.append(
                        int(order_name.split("_")[-1])
                    )
                    response_flat = flat_products[order_name].get("response_flat")
                    counts, var = _extract_region_for_order(
                        image, variance_image, row, caltype, response_flat
                    )
                    summed.append(counts)
                    summed_var.append(var)

                    if (
                        caltype == "FibTh"
                        and config.extraction_mode == "fibre"
                    ):
                        matrix, _, _ = tramlines.extract_order_matrix(
                            image, row, return_trace_offset=True
                        )
                        matrix_var, _, _ = tramlines.extract_order_matrix(
                            variance_image, row, return_trace_offset=True
                        )
                        matrix, matrix_var = _apply_response_flat(
                            matrix, matrix_var, response_flat
                        )
                        geometry = flat_products[
                            order_name
                        ]["fibre_geometry"]
                        result = extract_fibre_order(
                            matrix, matrix_var, geometry
                        )
                        idx = result.component_indices(
                            SCIENCE_COMPONENTS
                        )
                        fibre_counts.append(result.flux[:, idx])
                        fibre_var.append(result.variance[:, idx])

                exposure = {
                    "run": obs["run"],
                    "mjd_mid": float(obs["mjd_mid"]),
                    "exptime": float(obs["exptime"]),
                    "orders": np.asarray(orders),
                    "counts": np.asarray(summed),
                    "variance": np.asarray(summed_var),
                    "extraction_mode": config.extraction_mode,
                }
                if fibre_counts:
                    exposure["fibre_counts"] = np.asarray(fibre_counts)
                    exposure["fibre_variance"] = np.asarray(fibre_var)
                    exposure["fibre_components"] = SCIENCE_COMPONENTS

                output[caltype][ccd].append(exposure)

    return output


def save_calibration_spectra(
    calibration_data,
    filename,
    calibration_type,
    overwrite=True,
):
    """Save summed calibration spectra plus optional fibre-resolved FibTh cubes."""
    primary = fits.PrimaryHDU()
    primary.header["CALTYPE"] = calibration_type
    hdus = [primary]

    for ccd, exposures in calibration_data.items():
        for i, exposure in enumerate(exposures):
            counts = np.asarray(exposure["counts"], np.float32)
            variance = np.asarray(
                exposure["variance"], np.float32
            )
            columns = [
                fits.Column(
                    name="ORDER",
                    format="I",
                    array=exposure["orders"],
                ),
                fits.Column(
                    name="COUNTS",
                    format=f"{counts.shape[1]}E",
                    array=counts,
                ),
                fits.Column(
                    name="VARIANCE",
                    format=f"{variance.shape[1]}E",
                    array=variance,
                ),
            ]

            hdu = fits.BinTableHDU.from_columns(
                columns, name=f"CCD{ccd}_{i:03d}"
            )
            hdu.header["CCD"] = int(ccd)
            hdu.header["RUN"] = str(exposure["run"])
            hdu.header["MJD-MID"] = exposure["mjd_mid"]
            hdu.header["EXPTIME"] = exposure["exptime"]
            hdu.header["EXTRMODE"] = exposure.get(
                "extraction_mode", "summed"
            )
            hdus.append(hdu)

            if "fibre_counts" in exposure:
                f = fits.ImageHDU(
                    exposure["fibre_counts"].astype(np.float32),
                    name=f"FIB_C{ccd}_{i:03d}",
                )
                v = fits.ImageHDU(
                    exposure["fibre_variance"].astype(np.float32),
                    name=f"FVAR_C{ccd}_{i:03d}",
                )
                f.header["AXES"] = "ORDER,DISPERSION,FIBRE"
                for j, component in enumerate(
                    exposure["fibre_components"]
                ):
                    f.header[f"FIB{j:02d}"] = str(component)
                    v.header[f"FIB{j:02d}"] = str(component)
                hdus += [f, v]

    fits.HDUList(hdus).writeto(
        filename, overwrite=overwrite
    )


def extract_science_spectra(
    reduction_input,
    nightly_tramlines,
    flat_products,
    config,
    paths=None,
):
    """Extract response-corrected Science spectra in detector-pixel space.

    The direct summed Science/Sky apertures are always retained because they are
    the least model-dependent spectral-shape products.  In fibre mode the 19
    Science and five Sky fibres are additionally deblended.  Their recombination
    is stored only as QA here; the scientifically useful fibre combination is
    performed later after applying fibre-dependent wavelength solutions.

    When ``config.diagnostics == "full"`` and ``paths`` is provided, a
    per-order fibre/summed fidelity figure is saved below ``paths.debug``.
    """
    exposures = []

    for obs in reduction_input[reduction_input["type"] == "Science"]:
        if not obs["use"]:
            continue

        logger.info(
            "Extracting science run %s (%s), mode=%s",
            obs["run"], obs["object"], config.extraction_mode,
        )
        exposure = {
            "run": obs["run"],
            "object": str(obs["object"]),
            "mjd_mid": float(obs["mjd_mid"]),
            "exptime": float(obs["exptime"]),
            "extraction_mode": config.extraction_mode,
            "ccd": {},
        }

        for ccd in ("1", "2", "3"):
            if not obs[f"use_ccd{ccd}"]:
                continue
            filename = obs[f"file_ccd{ccd}"]
            if filename is None:
                continue

            frame = detector.preprocess_image(
                filename,
                ccd=ccd,
                config=config,
            )

            image = frame.image
            variance_image = frame.variance
            ccd_orders = {}
            fidelity_values = []
            logger.debug(
                "Run %s CCD%s: detector image %s; variance finite %.2f%%",
                obs["run"], ccd, image.shape,
                100.0 * np.mean(np.isfinite(variance_image)),
            )

            for row in nightly_tramlines:
                order_name = str(row["order_name"])
                if f"ccd_{ccd}_" not in order_name:
                    continue

                matrix, m, _ = tramlines.extract_order_matrix(
                    image, row, return_trace_offset=True
                )
                matrix_var, _, _ = tramlines.extract_order_matrix(
                    variance_image, row, return_trace_offset=True
                )
                matrix, matrix_var = _apply_response_flat(
                    matrix,
                    matrix_var,
                    flat_products[order_name].get("response_flat"),
                )

                summed = extract_summed_order(matrix, matrix_var, row)
                product = {
                    "components": summed.components,
                    "summed_flux": summed.flux,
                    "summed_variance": summed.variance,
                }

                if config.extraction_mode == "fibre":
                    geometry = flat_products[order_name]["fibre_geometry"]
                    label = f"run {obs['run']} {order_name}"
                    fibre = extract_fibre_order(
                        matrix,
                        matrix_var,
                        geometry,
                        return_covariance=True,
                        label=label,
                    )
                    qa = fibre_recombination_qa(summed, fibre, label=label)
                    fidelity_values.append(qa["robust_rms_fractional_structure"])
                    product.update(
                        fibre_components=fibre.components,
                        fibre_flux=fibre.flux,
                        fibre_variance=fibre.variance,
                        fibre_covariance=fibre.covariance,
                        fibre_recombined_flux=qa["recombined_flux"],
                        fibre_recombined_variance=qa["recombined_variance"],
                        fibre_to_summed_ratio=qa["ratio"],
                        fibre_fractional_structure=qa["fractional_structure"],
                        fibre_fidelity_rms=qa["robust_rms_fractional_structure"],
                    )

                    if (
                        getattr(config, "diagnostics", "none") == "full"
                        and paths is not None
                    ):
                        # Import lazily to keep plotting dependencies out of the
                        # normal extraction path.
                        from velocereduction import diagnostics
                        diagnostics.plot_fibre_extraction_qa(
                            qa,
                            geometry,
                            paths.debug / "fibre_extraction"
                            / f"run{int(obs['run']):04d}_{order_name}.png",
                            title=label,
                        )

                ccd_orders[order_name] = product

            if fidelity_values:
                values = np.asarray(fidelity_values, float)
                finite = values[np.isfinite(values)]
                logger.info(
                    "Run %s CCD%s fibre fidelity: %d orders; "
                    "median small-scale RMS %.4f%%; max %.4f%%",
                    obs["run"], ccd, len(fidelity_values),
                    100.0 * float(np.nanmedian(finite)) if finite.size else np.nan,
                    100.0 * float(np.nanmax(finite)) if finite.size else np.nan,
                )
            exposure["ccd"][ccd] = ccd_orders

        exposures.append(exposure)

    return exposures


def save_science_extractions(exposures, output_dir, overwrite=True):
    """Save detector-space summed spectra, fibres, and fibre-fidelity QA arrays.

    ``FR_*``, ``FRV_*``, ``FQR_*``, and ``FQS_*`` are explicitly QA-only:
    final fibre recombination occurs later on a common wavelength grid.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    for exposure in exposures:
        filename = output_dir / f'science_{int(exposure["run"]):04d}_extracted.fits'
        if filename.exists() and not overwrite:
            filenames.append(filename)
            continue

        primary = fits.PrimaryHDU()
        primary.header["RUN"] = str(exposure["run"])
        primary.header["OBJECT"] = exposure["object"]
        primary.header["MJD-MID"] = exposure["mjd_mid"]
        primary.header["EXPTIME"] = exposure["exptime"]
        primary.header["EXTRMODE"] = exposure["extraction_mode"]
        hdus = [primary]

        for ccd, orders in exposure["ccd"].items():
            for order_name, product in orders.items():
                summed = fits.ImageHDU(
                    np.asarray(product["summed_flux"], np.float32),
                    name=f"S_{order_name}",
                )
                svar = fits.ImageHDU(
                    np.asarray(product["summed_variance"], np.float32),
                    name=f"SV_{order_name}",
                )
                for j, component in enumerate(product["components"]):
                    summed.header[f"COMP{j:02d}"] = str(component)
                    svar.header[f"COMP{j:02d}"] = str(component)
                hdus += [summed, svar]

                if "fibre_flux" in product:
                    fibre = fits.ImageHDU(
                        np.asarray(product["fibre_flux"], np.float32),
                        name=f"F_{order_name}",
                    )
                    fvar = fits.ImageHDU(
                        np.asarray(product["fibre_variance"], np.float32),
                        name=f"FV_{order_name}",
                    )
                    for j, component in enumerate(product["fibre_components"]):
                        fibre.header[f"FIB{j:02d}"] = str(component)
                        fvar.header[f"FIB{j:02d}"] = str(component)
                    fibre.header["FIDRMS"] = float(product["fibre_fidelity_rms"])
                    hdus += [fibre, fvar]

                    frecombined = fits.ImageHDU(
                        np.asarray(product["fibre_recombined_flux"], np.float32),
                        name=f"FR_{order_name}",
                    )
                    frvar = fits.ImageHDU(
                        np.asarray(product["fibre_recombined_variance"], np.float32),
                        name=f"FRV_{order_name}",
                    )
                    fratio = fits.ImageHDU(
                        np.asarray(product["fibre_to_summed_ratio"], np.float32),
                        name=f"FQR_{order_name}",
                    )
                    fstructure = fits.ImageHDU(
                        np.asarray(product["fibre_fractional_structure"], np.float32),
                        name=f"FQS_{order_name}",
                    )
                    frecombined.header["PURPOSE"] = "QA_ONLY"
                    frvar.header["PURPOSE"] = "QA_ONLY"
                    fratio.header["PURPOSE"] = "QA_ONLY"
                    fstructure.header["PURPOSE"] = "QA_ONLY"
                    fstructure.header["FIDRMS"] = float(product["fibre_fidelity_rms"])
                    hdus += [frecombined, frvar, fratio, fstructure]

        fits.HDUList(hdus).writeto(filename, overwrite=True)
        filenames.append(filename)

    return filenames
