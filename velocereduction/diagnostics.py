"""Diagnostic plots for inspecting each reduction stage."""
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from .constants import N_DISPERSION

logger = logging.getLogger(__name__)


REGION_COLOURS = {
    "SimTh": "C1",
    "Sky_1": "C0",
    "Science": "C4",
    "Sky_2": "C0",
    "SimLC": "C3",
}

REGION_LABELS = {
    "SimTh": "SimTh",
    "Sky_1": "Sky",
    "Science": "Science",
    "Sky_2": None,     # avoid duplicate legend entry
    "SimLC": "SimLC",
}


def _save(fig, filename, dpi=160):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filename


def _representative_name(names, ccd):
    use = [name for name in names if str(name).split("_")[1] == str(ccd)]
    if not use:
        return None
    return sorted(use, key=lambda name: int(str(name).split("_")[-1]))[len(use) // 2]


def _binned_percentiles(x, y, n_bins=20, minimum=5):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < minimum:
        return np.array([]), np.array([]), np.array([]), np.array([])
    edges = np.linspace(np.nanmin(x[good]), np.nanmax(x[good]), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    p16 = np.full(n_bins, np.nan); p50 = p16.copy(); p84 = p16.copy()
    for i in range(n_bins):
        use = good & (x >= edges[i]) & (x < edges[i + 1] if i < n_bins - 1 else x <= edges[i + 1])
        if use.sum() >= minimum:
            p16[i], p50[i], p84[i] = np.nanpercentile(y[use], [16, 50, 84])
    return centres, p16, p50, p84


def _calibration_quality_summary(peak_table):
    """Local wrapper to avoid importing calibration.py at module import time."""
    from .calibration import calibration_quality_summary
    return calibration_quality_summary(peak_table)


def plot_calibration_order_diagnostic(
    counts, background, detection_snr, candidate_pixels, order_table, *,
    config, calibration_type, ccd, exposure_index, order, fibre=-1,
):
    """Create the v0.7-style full diagnostic page for one echelle order."""
    y = np.arange(len(counts), dtype=float)
    good = np.asarray(order_table["used_for_wavelength_fit"], dtype=bool)
    fibre_text = "" if int(fibre) == -1 else f" | fibre {int(fibre):+d}"

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2, 1.2, 1.2]},
    )

    ax = axes[0]
    ax.plot(y, counts, lw=0.7, label="Extracted spectrum")
    ax.plot(y, background, lw=0.8, label="Detection background")
    if str(calibration_type).lower() in {"simth", "fibth"}:
        ax.set_yscale("log")

    # Candidates that never produced a fitted-table row are especially useful
    # for diagnosing contiguous regions where the local peak fit failed.
    fitted_candidates = (
        np.asarray(order_table["candidate_pixel"], dtype=int)
        if "candidate_pixel" in order_table.colnames and len(order_table)
        else np.array([], dtype=int)
    )
    failed_candidates = np.setdiff1d(
        np.asarray(candidate_pixels, dtype=int), fitted_candidates, assume_unique=False
    )
    if len(failed_candidates):
        ax.scatter(
            failed_candidates, np.asarray(counts, float)[failed_candidates],
            marker="v", s=24, label="Detected, no fitted row", zorder=4,
        )
    if np.any(good):
        ax.scatter(order_table["y"][good], np.interp(order_table["y"][good], y, counts),
                   s=12, label="Accepted", zorder=5)
    if np.any(~good):
        ax.scatter(order_table["y"][~good], np.interp(order_table["y"][~good], y, counts),
                   marker="x", s=28, label="Rejected", zorder=6)
    if config.maximum_signal is not None:
        ax.axhline(config.maximum_signal, ls=":", lw=1, label="Maximum signal")
    ax.set_ylabel("Counts")
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    ax.set_title(
        f"{calibration_type} | CCD {ccd} | exposure {exposure_index}{fibre_text} | order {order}"
    )

    ax = axes[1]
    ax.plot(y, detection_snr, lw=0.7)
    if len(candidate_pixels):
        ax.scatter(candidate_pixels, detection_snr[candidate_pixels], marker="x", s=15)
    if str(calibration_type).lower() in {"simth", "fibth"}:
        ax.set_yscale("log")
    ax.axhline(config.detection_snr, ls="--", lw=1)
    ax.set_ylabel("Detection S/N")

    ax = axes[2]
    # For SimLC the final ``fwhm`` is the effective width of one fixed LSF per
    # order and is therefore identical for every refitted mode by construction.
    # Preserve/show the initial Gaussian widths when available so dispersion-
    # direction changes in the actual line profile remain visible.
    fwhm_name = "fwhm_initial" if "fwhm_initial" in order_table.colnames else "fwhm"
    fwhm_label = "Initial Gaussian FWHM [pixel]" if fwhm_name == "fwhm_initial" else "FWHM [pixel]"
    if np.any(good):
        ax.scatter(order_table["y"][good], np.clip(order_table[fwhm_name][good], 0, 4),
                   s=10, label="Accepted")
    if np.any(~good):
        ax.scatter(order_table["y"][~good], np.clip(order_table[fwhm_name][~good], 0, 4),
                   marker="x", s=20, label="Rejected")
    width_reference = np.asarray(order_table[fwhm_name][good], dtype=float)
    if np.any(np.isfinite(width_reference)):
        ax.axhline(np.nanmedian(width_reference), ls="--", lw=1)
    if fwhm_name == "fwhm_initial" and "fwhm" in order_table.colnames:
        final_width = np.asarray(order_table["fwhm"], dtype=float)
        if np.any(np.isfinite(final_width)):
            ax.axhline(np.nanmedian(final_width), ls=":", lw=1, label="Adopted order LSF FWHM")
            ax.legend(fontsize=7, loc="best")
    ax.set_ylabel(fwhm_label)

    ax = axes[3]
    if np.any(good):
        ax.scatter(order_table["y"][good], np.clip(order_table["y_uncertainty"][good], 0, 0.2),
                   s=10, label="Accepted")
    if np.any(~good):
        ax.scatter(order_table["y"][~good], np.clip(order_table["y_uncertainty"][~good], 0, 0.2),
                   marker="x", s=20, label="Rejected")
    ax.axhline(config.maximum_y_uncertainty, ls="--", lw=1)
    ax.set_ylabel(r"$\sigma_y$ [pixel]")
    ax.set_xlabel(r"Dispersion pixel $y$")

    summary = _calibration_quality_summary(order_table)
    lsf_text = ""
    if "lsf_model" in order_table.colnames and len(order_table):
        models = sorted(set(str(v) for v in order_table["lsf_model"] if str(v)))
        sources = (
            sorted(set(int(v) for v in order_table["lsf_source_order"] if int(v) >= 0))
            if "lsf_source_order" in order_table.colnames else []
        )
        if models:
            lsf_text = f" | LSF={','.join(models)}"
            if sources:
                lsf_text += f" src={','.join(map(str, sources))}"
    fig.text(
        0.99, 0.01,
        f"candidates={len(candidate_pixels)} | fitted={len(order_table)} | "
        f"no-fit={len(failed_candidates)} | accepted={summary['accepted']} | "
        f"unmatched={summary.get('unmatched', 0)} | badfit={summary.get('bad_profile_fit', 0)} | "
        f"sat={summary.get('saturated', 0)} | width={summary.get('width_outlier', 0)} | "
        f"blend={summary.get('blend_candidate', 0)} | lowS/N={summary.get('low_snr', 0)} | "
        f"sigma_y={summary.get('large_centroid_error', 0)}{lsf_text}",
        ha="right", va="bottom", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    return fig


def plot_calibration_summary(
    peak_table, *, calibration_type, ccd, exposure_index, filename, fibre=-1, show=False,
):
    """Save a compact CCD/exposure-level calibration-line QA figure."""
    if len(peak_table) == 0:
        return None
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    good = np.asarray(peak_table["used_for_wavelength_fit"], dtype=bool)
    identified = (
        np.isfinite(np.asarray(peak_table["wavelength_nm"], float))
        if "wavelength_nm" in peak_table.colnames else np.zeros(len(peak_table), bool)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    if np.any(good):
        sc = ax.scatter(peak_table["y"][good], peak_table["order"][good],
                        c=peak_table["fwhm"][good], s=6)
        fig.colorbar(sc, ax=ax, label="FWHM [pixel]")
    ax.set(xlabel=r"Dispersion pixel $y$", ylabel=r"Echelle order $m$", title="Line width")

    ax = axes[0, 1]
    if np.any(good):
        ax.scatter(peak_table["y"][good], peak_table["y_uncertainty"][good], s=6)
    ax.set(xlabel=r"Dispersion pixel $y$", ylabel=r"$\sigma_y$ [pixel]", title="Centroid precision")

    ax = axes[1, 0]
    if np.any(good):
        ax.scatter(peak_table["y"][good], peak_table["signal_to_noise"][good], s=6)
    ax.set(xlabel=r"Dispersion pixel $y$", ylabel="Fitted line S/N", title="Line signal-to-noise")

    ax = axes[1, 1]
    if np.any(good):
        ax.hist(peak_table["pixel_phase"][good], bins=30)
    ax.set(xlabel="Pixel phase", ylabel="Number of accepted lines", title="Sub-pixel centroid sampling")

    summary = _calibration_quality_summary(peak_table)
    fibre_text = "" if int(fibre) == -1 else f" | fibre {int(fibre):+d}"
    fig.suptitle(
        f"{calibration_type} | CCD {ccd} | exposure {exposure_index}{fibre_text} | "
        f"{summary['accepted']}/{summary['total']} retained; {np.count_nonzero(identified)} matched"
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return filename


def plot_worst_peak_fits(
    counts, order_table, *, config, calibration_type, ccd, exposure_index,
    order, fibre=-1, max_peaks=8,
):
    """Plot rejected or otherwise worst local calibration-peak fits."""
    from .calibration import CalibrationPeakFlag, calibration_line_model
    if len(order_table) == 0:
        return None
    rejected = np.where(~np.asarray(order_table["used_for_wavelength_fit"], bool))[0]
    fit_rms = np.asarray(order_table["fit_rms"], float)
    ranking = np.argsort(np.nan_to_num(fit_rms, nan=-np.inf))[::-1]
    selected = list(rejected[:max_peaks])
    for index in ranking:
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) >= max_peaks:
            break
    selected = selected[:max_peaks]
    if not selected:
        return None

    n_columns = 2
    n_rows = int(np.ceil(len(selected) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12, 3.0 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, table_index in zip(axes.ravel(), selected):
        ax.set_visible(True)
        candidate_pixel = int(order_table["candidate_pixel"][table_index])
        left = max(0, candidate_pixel - config.fit_half_width)
        right = min(len(counts), candidate_pixel + config.fit_half_width + 1)
        fit_y = np.arange(left, right, dtype=float)
        row = order_table[table_index]
        sigma = float(row["fwhm"]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        fit_model = calibration_line_model(
            fit_y, float(row["integrated_counts"]), float(row["y"]), sigma,
            float(row["background"]), float(row["background_slope"]),
            y_reference=float(candidate_pixel),
        )
        ax.scatter(fit_y, np.asarray(counts, float)[left:right], s=20, label="data")
        ax.plot(fit_y, fit_model, lw=1.2, label="fit")
        ax.axvline(float(row["y"]), ls="--", lw=0.8)
        flag_value = int(order_table["quality_flag"][table_index])
        flag_names = [flag.name for flag in CalibrationPeakFlag
                      if flag != CalibrationPeakFlag.GOOD and (flag_value & int(flag)) != 0]
        ax.set_title(
            f"y={float(row['y']):.3f}, FWHM={float(row['fwhm']):.2f}, S/N={float(row['signal_to_noise']):.1f}\n"
            f"{', '.join(flag_names) if flag_names else 'GOOD'}", fontsize=9,
        )
        ax.set(xlabel=r"Dispersion pixel $y$", ylabel="Counts")
    axes.ravel()[0].legend(fontsize=8)
    fibre_text = "" if int(fibre) == -1 else f" | fibre {int(fibre):+d}"
    fig.suptitle(
        f"{calibration_type} | CCD {ccd} | exposure {exposure_index}{fibre_text} | "
        f"order {order}: rejected / worst local fits"
    )
    fig.tight_layout()
    return fig


def save_calibration_order_diagnostics(
    order_diagnostics, peak_table, *, config, calibration_type, ccd,
    exposure_index, diagnostic_dir, fibre=-1, include_worst_fits=True,
):
    """Write the multi-page v0.7-style per-order calibration QA PDF."""
    diagnostic_dir = Path(diagnostic_dir)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    fibre_suffix = "" if int(fibre) == -1 else f"_fibre{int(fibre):+03d}"
    filename = diagnostic_dir / (
        f"{str(calibration_type).lower()}_ccd{ccd}_exposure{int(exposure_index):03d}"
        f"{fibre_suffix}_peaks.pdf"
    )
    table_orders = (
        np.unique(np.asarray(peak_table["order"], int))
        if len(peak_table) and "order" in peak_table.colnames else np.array([], dtype=int)
    )
    orders = np.unique(np.r_[table_orders, np.asarray(list(order_diagnostics), dtype=int)])
    with PdfPages(filename) as pdf:
        for order in orders:
            subset = (
                peak_table[np.asarray(peak_table["order"], int) == int(order)]
                if "order" in peak_table.colnames else peak_table
            )
            diag = order_diagnostics.get(int(order))
            if diag is None:
                continue
            fig = plot_calibration_order_diagnostic(
                diag["counts"], diag["background"], diag["detection_snr"],
                diag["candidate_pixels"], subset, config=config,
                calibration_type=calibration_type, ccd=ccd,
                exposure_index=exposure_index, order=int(order), fibre=fibre,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            if include_worst_fits:
                worst = plot_worst_peak_fits(
                    diag["counts"], subset, config=config,
                    calibration_type=calibration_type, ccd=ccd,
                    exposure_index=exposure_index, order=int(order), fibre=fibre,
                )
                if worst is not None:
                    pdf.savefig(worst, bbox_inches="tight")
                    plt.close(worst)
    return filename


def save_calibration_order_diagnostics_from_counts(
    counts, orders, peak_table, *, config, calibration_type, ccd,
    exposure_index, diagnostic_dir, fibre=-1,
):
    """Create per-order peak QA from raw extracted counts and an existing line table.

    This path is deliberately measurement-free: it reruns only candidate detection
    to reconstruct the background/SNR curves and candidate locations, then combines
    those with the already-saved line table.  It is therefore safe to use when
    cached calibration-line FITS products are being reused.
    """
    from .calibration import detect_calibration_peaks

    counts = np.asarray(counts, dtype=float)
    orders = np.asarray(orders, dtype=int)
    if counts.ndim != 2 or counts.shape[0] != len(orders):
        raise ValueError("counts must have shape (n_orders, n_pixels)")

    order_diagnostics = {}
    for order_index, order in enumerate(orders):
        candidates, background, _, detection_snr, _ = detect_calibration_peaks(
            counts[order_index], config=config
        )
        order_diagnostics[int(order)] = dict(
            counts=counts[order_index],
            background=background,
            detection_snr=detection_snr,
            candidate_pixels=candidates,
        )

    return save_calibration_order_diagnostics(
        order_diagnostics, peak_table, config=config,
        calibration_type=calibration_type, ccd=ccd,
        exposure_index=exposure_index, diagnostic_dir=diagnostic_dir,
        fibre=fibre, include_worst_fits=False,
    )


def print_wavelength_fit_summary(fit, peak_table, *, calibration_type="FibTh", ccd="", max_outliers=8):
    """Print final surface-fit usage, robust statistics, and worst clipped lines."""
    used = np.asarray(peak_table["used_for_wavelength_fit"], bool)
    finite_residual = np.isfinite(np.asarray(peak_table["velocity_residual_mps"], float))
    considered = finite_residual
    v = np.asarray(peak_table["velocity_residual_mps"], float)
    p = np.asarray(peak_table["pixel_residual"], float)
    used_v = v[used & finite_residual]
    used_p = p[used & finite_residual]
    flags = np.asarray(peak_table["quality_flag"], np.int64)
    from .calibration import CalibrationPeakFlag
    clipped = (flags & int(CalibrationPeakFlag.WAVELENGTH_OUTLIER)) != 0

    def _rms(x):
        return float(np.sqrt(np.nanmean(x*x))) if len(x) else np.nan
    def _pct(x, q):
        return float(np.nanpercentile(np.abs(x), q)) if len(x) else np.nan

    degrees = (fit.solution.coefficients.shape[0]-1, fit.solution.coefficients.shape[1]-1)
    print("\n" + "=" * 78)
    print(f"Wavelength solution {calibration_type} CCD{ccd}")
    print(
        f"  Legendre degrees: y={degrees[0]}, m={degrees[1]} | "
        f"iterations={fit.n_iterations}"
    )
    print(
        f"  surface input={np.count_nonzero(considered)} | used={np.count_nonzero(used & considered)} | "
        f"hard-clipped={np.count_nonzero(clipped)} | "
        f"Huber-downweighted={np.count_nonzero(fit.used & (fit.robust_weight < 0.999))}"
    )
    print(
        f"  RMS={_rms(used_p):.4f} pix / {_rms(used_v):.1f} m/s | "
        f"median |dv|={np.nanmedian(np.abs(used_v)):.1f} m/s | "
        f"P95 |dv|={_pct(used_v,95):.1f} m/s | max |dv|={_pct(used_v,100):.1f} m/s"
    )
    if np.any(clipped & finite_residual):
        idx = np.where(clipped & finite_residual)[0]
        idx = idx[np.argsort(np.abs(v[idx]))[::-1]][:max_outliers]
        print("  worst hard-clipped lines:")
        for i in idx:
            print(
                f"    order {int(peak_table['order'][i]):3d}  y={float(peak_table['y'][i]):8.3f}  "
                f"res={p[i]:+7.3f} pix = {v[i]/1000:+7.3f} km/s"
            )



def print_wavelength_degree_summary(validation_table, chosen, *, top=8, ccd=""):
    """Print diagnostics for choosing Legendre surface degree.

    The mean held-out RMS and its standard error remain the quantities used by
    the one-standard-error selector.  Robust fold-level summaries are printed
    alongside them so a single catastrophic blocked fold is obvious rather
    than hidden inside the mean.
    """
    if len(validation_table) == 0:
        print(f"CCD{ccd}: no viable wavelength-surface CV models")
        return

    rms = np.asarray(validation_table["validation_rms_pixel"], float)
    order = np.argsort(rms)
    best = validation_table[int(order[0])]
    threshold = float(best["validation_rms_pixel"] + best["validation_rms_pixel_se"])

    print(
        f"CCD{ccd} wavelength-surface CV: best raw mean RMS "
        f"(y={int(best['y_degree'])}, m={int(best['order_degree'])}) = "
        f"{float(best['validation_rms_pixel']):.4f} +/- "
        f"{float(best['validation_rms_pixel_se']):.4f} pix; "
        f"one-SE threshold={threshold:.4f} pix"
    )
    print(
        f"  best fold statistics: median={float(best['validation_median_fold_rms_pixel']):.4f}, "
        f"MAD={float(best['validation_mad_fold_rms_pixel']):.4f}, "
        f"P90={float(best['validation_p90_fold_rms_pixel']):.4f}, "
        f"max={float(best['validation_max_fold_rms_pixel']):.4f} pix, "
        f"max/median={float(best['validation_fold_rms_ratio']):.2f}"
    )
    print(
        f"  chosen simplest model within one SE: "
        f"(y={int(chosen['y_degree'])}, m={int(chosen['order_degree'])}), "
        f"CV mean RMS={float(chosen['validation_rms_pixel']):.4f} pix / "
        f"{float(chosen['validation_rms_velocity_mps']):.1f} m/s, "
        f"median-fold={float(chosen['validation_median_fold_rms_pixel']):.4f} pix, "
        f"max/median={float(chosen['validation_fold_rms_ratio']):.2f}, "
        f"P95(all)={float(chosen['validation_p95_abs_pixel']):.4f} pix, "
        f"gap={float(chosen['generalisation_gap_pixel']):+.4f} pix, "
        f"cond_max={float(chosen['max_condition_number']):.2e}"
    )
    print("  lowest held-out mean-RMS models:")
    for i in order[: min(int(top), len(order))]:
        row = validation_table[int(i)]
        marker = "*" if (
            int(row["y_degree"]) == int(chosen["y_degree"])
            and int(row["order_degree"]) == int(chosen["order_degree"])
        ) else " "
        print(
            f"   {marker} y={int(row['y_degree']):2d}, m={int(row['order_degree']):2d}: "
            f"mean={float(row['validation_rms_pixel']):.4f}, "
            f"median={float(row['validation_median_fold_rms_pixel']):.4f}, "
            f"max/med={float(row['validation_fold_rms_ratio']):.2f}, "
            f"train={float(row['train_rms_pixel']):.4f}, "
            f"P95={float(row['validation_p95_abs_pixel']):.4f}, "
            f"Npar={int(row['n_parameters']):3d}, "
            f"cond={float(row['max_condition_number']):.2e}"
        )


def save_wavelength_degree_diagnostics(
    validation_table,
    chosen,
    filename,
    *,
    calibration_type="FibTh",
    ccd="",
    diagnostics="basic",
):
    """Save a robust six-panel diagnostic for Legendre degree selection.

    The colour range is determined robustly so isolated catastrophic CV cells
    do not wash out the structure among otherwise sensible degree pairs.  The
    actual values remain in the FITS table and the mean-RMS panel is annotated.
    """
    if str(diagnostics).lower() == "none" or len(validation_table) == 0:
        return None

    y_values = np.sort(np.unique(np.asarray(validation_table["y_degree"], int)))
    m_values = np.sort(np.unique(np.asarray(validation_table["order_degree"], int)))

    def grid(name):
        out = np.full((len(y_values), len(m_values)), np.nan)
        for row in validation_table:
            iy = np.flatnonzero(y_values == int(row["y_degree"]))
            im = np.flatnonzero(m_values == int(row["order_degree"]))
            if len(iy) and len(im):
                out[iy[0], im[0]] = float(row[name])
        return out

    mean_rms = grid("validation_rms_pixel")
    median_rms = grid("validation_median_fold_rms_pixel")
    fold_ratio = grid("validation_fold_rms_ratio")
    p95 = grid("validation_p95_abs_pixel")
    gap = grid("generalisation_gap_pixel")
    condition = np.log10(grid("max_condition_number"))

    best_i = int(np.nanargmin(np.asarray(validation_table["validation_rms_pixel"], float)))
    best = validation_table[best_i]
    best_xy = (
        int(np.flatnonzero(m_values == int(best["order_degree"]))[0]),
        int(np.flatnonzero(y_values == int(best["y_degree"]))[0]),
    )
    chosen_xy = (
        int(np.flatnonzero(m_values == int(chosen["order_degree"]))[0]),
        int(np.flatnonzero(y_values == int(chosen["y_degree"]))[0]),
    )

    panels = (
        (mean_rms, "Held-out mean RMS", "pixel", True),
        (median_rms, "Held-out median fold RMS", "pixel", False),
        (fold_ratio, "Maximum / median fold RMS", "", False),
        (p95, "Held-out P95 |residual|", "pixel", False),
        (gap, "Validation - training RMS", "pixel", False),
        (condition, r"Maximum $\log_{10}$ condition number", "", False),
    )

    def robust_limits(values):
        finite = np.asarray(values, float)
        finite = finite[np.isfinite(finite)]
        if len(finite) < 4:
            return None, None
        lo = float(np.nanpercentile(finite, 2))
        hi = float(np.nanpercentile(finite, 90))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None, None
        return lo, hi

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (values, title, label, annotate) in zip(axes.ravel(), panels):
        vmin, vmax = robust_limits(values)
        image = ax.imshow(
            values, origin="lower", aspect="auto", cmap="RdYlBu",
            vmin=vmin, vmax=vmax,
        )
        cb = fig.colorbar(image, ax=ax)
        if label:
            cb.set_label(label)
        ax.set(
            xticks=np.arange(len(m_values)),
            xticklabels=m_values,
            yticks=np.arange(len(y_values)),
            yticklabels=y_values,
            xlabel=r"Legendre degree $d_m$",
            ylabel=r"Legendre degree $d_y$",
            title=title,
        )
        ax.plot(
            best_xy[0], best_xy[1], marker="*", ms=13, linestyle="none",
            label="minimum mean CV RMS",
        )
        ax.plot(
            chosen_xy[0], chosen_xy[1], marker="o", ms=12, mfc="none",
            mew=2, linestyle="none", label="one-SE choice",
        )
        if annotate:
            for iy in range(values.shape[0]):
                for im in range(values.shape[1]):
                    if np.isfinite(values[iy, im]):
                        value = values[iy, im]
                        text = f"{value:.2f}" if abs(value) >= 10 else f"{value:.3f}"
                        ax.text(im, iy, text, ha="center", va="center", fontsize=6)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"{calibration_type} CCD{ccd}: wavelength-surface complexity | "
        f"chosen (d_y,d_m)=({int(chosen['y_degree'])},{int(chosen['order_degree'])}), "
        f"CV mean RMS={float(chosen['validation_rms_pixel']):.4f} pix / "
        f"{float(chosen['validation_rms_velocity_mps']):.1f} m s$^{{-1}}$"
    )

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    if str(diagnostics).lower() == "full":
        plt.show()
    plt.close(fig)
    return filename


def save_wavelength_diagnostics(
    fit, peak_table, filename, *, calibration_type="FibTh", ccd="", diagnostics="basic",
):
    """Save the v0.7-style presentation-quality static wavelength QA figure.

    Only final surface-fit lines are shown in the residual panels.  Rejected
    wavelength outliers are reported by ``print_wavelength_fit_summary`` rather
    than mixed into the fitted residual distribution.
    """
    diagnostics = str(diagnostics).lower()
    if diagnostics == "none":
        return None
    used = np.asarray(peak_table["used_for_wavelength_fit"], bool)
    used &= np.isfinite(np.asarray(peak_table["pixel_residual"], float))
    if not np.any(used):
        return None

    y = np.asarray(peak_table["y"], float)[used]
    m = np.asarray(peak_table["order"], float)[used]
    phase = np.asarray(peak_table["pixel_phase"], float)[used]
    pixel_residual = np.asarray(peak_table["pixel_residual"], float)[used]
    wavelength_residual_angstrom = 10.0 * np.asarray(peak_table["wavelength_residual_nm"], float)[used]
    velocity_residual_mps = np.asarray(peak_table["velocity_residual_mps"], float)[used]
    degree_y = fit.solution.coefficients.shape[0] - 1
    degree_m = fit.solution.coefficients.shape[1] - 1

    rms_pixel = float(np.sqrt(np.nanmean(pixel_residual**2)))
    rms_angstrom = float(np.sqrt(np.nanmean(wavelength_residual_angstrom**2)))
    rms_velocity = float(np.sqrt(np.nanmean(velocity_residual_mps**2)))
    residual_colour_limit = max(float(np.nanpercentile(np.abs(pixel_residual), 99)), 1e-6)
    velocity_limit = max(1.1 * float(np.nanpercentile(np.abs(velocity_residual_mps), 99)), 1.0)
    colour_norm = TwoSlopeNorm(vmin=-residual_colour_limit, vcenter=0.0, vmax=residual_colour_limit)
    unique_orders = np.unique(m)
    order_limits = (np.nanmin(unique_orders)-0.5, np.nanmax(unique_orders)+0.5)
    dispersion_limits = (0, 4111)

    with plt.rc_context({"font.size":14,"axes.labelsize":15,"xtick.labelsize":13,
                         "ytick.labelsize":13,"legend.fontsize":13}):
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(
            5, 3, height_ratios=[0.55,1.0,1.0,0.18,0.78], width_ratios=[1,1,1],
            left=0.07, right=0.965, bottom=0.085, top=0.965, wspace=0.28, hspace=0.16,
        )
        ax_info=fig.add_subplot(gs[0,0]); ax_cb=fig.add_subplot(gs[0,1:3])
        ax_order=fig.add_subplot(gs[1:3,0]); ax_map=fig.add_subplot(gs[1:3,1:3],sharey=ax_order)
        ax_phase=fig.add_subplot(gs[4,0]); ax_y=fig.add_subplot(gs[4,1:3],sharex=ax_map)

        ax_info.axis("off")
        ax_info.text(0.5,0.8,f"Wavelength solution {calibration_type} CCD{ccd}",
                     transform=ax_info.transAxes,ha="center",va="top",fontsize=18,fontweight="bold")
        ax_info.text(0.5,0.4,f"2-dim. Legendre fit (deg_y={degree_y}, deg_m={degree_m}) to {len(y):,} lines",
                     transform=ax_info.transAxes,ha="center",va="top",fontsize=14)
        ax_info.text(0.5,0.15,
                     rf"$\mathbf{{RMS}} = {rms_pixel:.4f}$ px; {rms_angstrom:.5f} $\AA$; {rms_velocity:.1f} m s$^{{-1}}$",
                     transform=ax_info.transAxes,ha="center",va="top",fontsize=14)

        order_median=[]; order_lower=[]; order_upper=[]
        for order in unique_orders:
            r=pixel_residual[m==order]; p16,p50,p84=np.nanpercentile(r,[16,50,84])
            order_median.append(p50); order_lower.append(p50-p16); order_upper.append(p84-p50)
        ax_order.errorbar(order_median,unique_orders,xerr=np.vstack([order_lower,order_upper]),
                          fmt="o",markersize=5,capsize=2,linewidth=1.2)
        ax_order.axvline(0,ls="--",lw=1); ax_order.set_xlim(-residual_colour_limit,residual_colour_limit)
        ax_order.set_ylim(order_limits); ax_order.set_xlabel(r"Residual$~/~\mathrm{px}$",labelpad=7)
        ax_order.set_ylabel(r"Echelle order $m$")

        scatter=ax_map.scatter(y,m,c=pixel_residual,s=16,alpha=0.90,cmap="RdBu_r",norm=colour_norm,
                               linewidths=0,rasterized=True)
        ax_map.set_xlim(dispersion_limits); ax_map.set_ylim(order_limits)
        ax_map.set_xlabel(r"Dispersion pixel $y$",labelpad=7); ax_map.set_ylabel(r"Echelle order $m$")
        ax_cb.axis("off"); cax=ax_cb.inset_axes([0.0,0.40,1.0,0.24])
        cb=fig.colorbar(scatter,cax=cax,orientation="horizontal")
        cb.set_label(r"Wavelength-model residual$~/~\mathrm{px}$",fontsize=14,labelpad=4)
        cb.ax.tick_params(labelsize=12); cb.ax.xaxis.set_ticks_position("bottom"); cb.ax.xaxis.set_label_position("bottom")

        centres,p16,p50,p84=_binned_percentiles(phase,velocity_residual_mps,15)
        ax_phase.scatter(phase,velocity_residual_mps,s=12,alpha=0.40,linewidths=0,rasterized=True)
        ax_phase.fill_between(centres,p16,p84,alpha=0.25,linewidth=0); ax_phase.plot(centres,p50,lw=2.5)
        ax_phase.axhline(0,ls="--",lw=1); ax_phase.set_xlim(-0.5,0.5); ax_phase.set_ylim(-velocity_limit,velocity_limit)
        ax_phase.set_xlabel("Pixel phase"); ax_phase.set_ylabel(r"Residual$~/~\mathrm{m\,s^{-1}}$")

        centres,p16,p50,p84=_binned_percentiles(y,velocity_residual_mps,20)
        ax_y.scatter(y,velocity_residual_mps,s=12,alpha=0.40,linewidths=0,rasterized=True)
        ax_y.fill_between(centres,p16,p84,alpha=0.25,linewidth=0); ax_y.plot(centres,p50,lw=2.5)
        ax_y.axhline(0,ls="--",lw=1); ax_y.set_xlim(dispersion_limits); ax_y.set_ylim(-velocity_limit,velocity_limit)
        ax_y.set_xlabel(r"Dispersion pixel $y$"); ax_y.set_ylabel(r"Residual$~/~\mathrm{m\,s^{-1}}$")

        filename=Path(filename); filename.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(filename,dpi=200,bbox_inches="tight")
        if diagnostics == "full":
            plt.show()
        plt.close(fig)
    return filename


def plot_detector_shifts(detector_shifts, filename):
    ccd = np.asarray(detector_shifts["ccd"], int)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for ax, key, scatter_key, label in (
        (axes[0], "dx", "dx_scatter", r"$\Delta x$ / pixel"),
        (axes[1], "dy", "dy_scatter", r"$\Delta y$ / pixel"),
    ):
        values = np.asarray(detector_shifts[key], float)
        error = np.asarray(detector_shifts[scatter_key], float)
        ax.errorbar(ccd, values, yerr=np.where(np.isfinite(error), error, 0), fmt="o", capsize=3)
        ax.axhline(0, ls="--", lw=1)
        ax.set(xticks=ccd, xlabel="CCD", ylabel=label)
    fig.suptitle("Detector registration relative to reference night")
    return _save(fig, filename)


def plot_read_noise(read_noise_table, filename):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True, constrained_layout=True)
    for ccd, ax in zip((1, 2, 3), axes):
        subset = read_noise_table[np.asarray(read_noise_table["ccd"], int) == ccd]
        for amp in sorted(set(np.asarray(subset["amplifier"]).astype(str))):
            rows = subset[np.asarray(subset["amplifier"]).astype(str) == amp]
            order = np.argsort(np.asarray(rows["run"], int))
            ax.plot(np.asarray(rows["run"], int)[order], np.asarray(rows["rms_adu"], float)[order], ".-", label=amp)
        ax.set(title=f"CCD{ccd}", xlabel="Flat run")
        if len(subset): ax.legend(fontsize=8)
    axes[0].set_ylabel("Overscan RMS / ADU")
    fig.suptitle("Read-noise monitoring from Flat overscans")
    return _save(fig, filename)


def plot_order_geometry_summary(geometries, filename):
    """Trace quality and extraction regions versus physical echelle order."""
    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 6),
        constrained_layout=True,
    )

    for column, ccd in enumerate(("1", "2", "3")):
        subset = sorted([g for g in geometries if g.ccd == ccd],key=lambda g: g.order,)

        order = np.array([g.order for g in subset])
        rms = np.array([g.trace_rms for g in subset])

        axes[0, column].plot(order, rms, "o-", lw=1)
        axes[0, column].set_title(f"CCD{ccd}")

        if column == 0:
            axes[0, column].set_ylabel("Trace RMS / pixel")

        for region in ("SimTh", "Sky_1", "Science", "Sky_2", "SimLC"):
            begin = np.array([ g.regions.get(region, (np.nan, np.nan))[0] for g in subset])
            end = np.array([ g.regions.get(region, (np.nan, np.nan))[1] for g in subset])

            colour = REGION_COLOURS[region]
            label = REGION_LABELS[region]

            axes[1, column].plot(order,begin,color=colour,lw=1.0,label=label,)
            axes[1, column].plot(order,end,color=colour,lw=1.0,)
            axes[1, column].fill_between(order,begin,end,color=colour,alpha=0.08,)

        axes[1, column].set_xlabel("Echelle order")

        if column == 0:
            axes[1, column].set_ylabel("Relative cross-dispersion pixel")

    axes[1, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Order geometry")
    return _save(fig, filename)


def plot_order_matrix_examples(combined_flats, geometries, filename):
    """Representative 81-pixel OrderMatrix and extraction regions."""
    from . import orders

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True,)

    for column, ccd in enumerate(("1", "2", "3")):
        subset = sorted([g for g in geometries if g.ccd == ccd], key=lambda g: g.order,)
        if not subset:
            continue

        geometry = subset[len(subset) // 2]
        matrix = orders.extract_order_matrix(combined_flats[ccd],geometry,)

        finite = matrix.flux[np.isfinite(matrix.flux)]
        vmin, vmax = (np.nanpercentile(finite, [5, 99]) if finite.size else (0, 1))

        axes[0, column].imshow(matrix.flux, origin="lower", aspect="auto", cmap="Greys", vmin=vmin, vmax=vmax, interpolation="none",)

        axes[0, column].set_title(f"CCD{ccd} order {geometry.order}")

        profile = np.nanmedian(matrix.flux, axis=0)
        axes[1, column].plot(matrix.relative_x, profile, color="0.2", lw=1,)

        for region in (
            "SimTh", "Sky_1", "Science", "Sky_2", "SimLC"
        ):
            if region not in geometry.regions:
                continue

            begin, end = geometry.regions[region]
            axes[1, column].axvspan(begin, end, color=REGION_COLOURS[region], alpha=0.18, lw=0, label=REGION_LABELS[region],)

        axes[1, column].set_xlabel("Relative cross-dispersion pixel")

        if column == 0:
            axes[0, column].set_ylabel("Dispersion pixel")
            axes[1, column].set_ylabel("Median Flat counts")

    axes[1, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Representative OrderMatrix products")
    return _save(fig, filename)


def plot_fibre_geometry_summary(geometries, filename):
    """Central fibre width/separation versus order, with within-order ranges."""
    values = list(geometries.values()) if isinstance(geometries, dict) else list(geometries)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for column, ccd in enumerate(("1", "2", "3")):
        subset = sorted([g for g in values if g.ccd == ccd], key=lambda g: g.order)
        order = np.array([g.order for g in subset])
        y = np.arange(4112)
        for ax, method, ylabel in (
            (axes[0, column], "sigma", r"$\sigma$ / pixel"),
            (axes[1, column], "separation", "Separation / pixel"),
        ):
            curves = [getattr(g, method)(y) for g in subset]
            centre = np.array([getattr(g, method)(g.y_reference) for g in subset])
            lower = np.array([np.nanmin(v) for v in curves])
            upper = np.array([np.nanmax(v) for v in curves])
            ax.errorbar(order, centre, yerr=np.vstack((centre - lower, upper - centre)), fmt="o", capsize=2)
            if column == 0: ax.set_ylabel(ylabel)
        axes[0, column].set_title(f"CCD{ccd}")
        axes[1, column].set_xlabel("Echelle order")
    fig.suptitle("Compact Flat-derived fibre geometry")
    return _save(fig, filename)


def plot_fibre_profile_summary(geometries, flat_order_matrices, filename):
    """Observed collapsed Flat profile, fibre model and residual."""
    from . import fibres

    names = list(flat_order_matrices)
    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 5),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    for column, ccd in enumerate(("1", "2", "3")):
        name = _representative_name(names, ccd)
        if name is None:
            continue

        matrix = flat_order_matrices[name]
        collapsed = fibres.fit_collapsed_fibre_profile(matrix)

        axes[0, column].plot(
            collapsed["x"],
            collapsed["profile"],
            lw=1.0,
            label="Flat",
        )
        axes[0, column].plot(
            collapsed["x"],
            collapsed["model"],
            color="C3",
            lw=1.0,
            label="fibre model",
        )

        axes[1, column].plot(
            collapsed["x"],
            collapsed["profile"] - collapsed["model"],
            lw=0.8,
        )
        axes[1, column].axhline(0, color="0.5", ls="--", lw=0.8)

        axes[0, column].set_title(
            name.replace("ccd_", "CCD ").replace("_order_", " order ")
        )
        axes[0, column].legend(fontsize=8)
        axes[1, column].set_xlabel("Relative cross-dispersion pixel")

        axes[0, column].set_xlim(-1,81)
        axes[0, column].set_xticks([0, 20, 40, 60, 80],[-40, -20, 0, 20, 40])
        axes[1, column].set_xlim(-41,41)
        axes[1, column].set_xticks([-40, -20, 0, 20, 40])

        if column == 0:
            axes[0, column].set_ylabel("Counts")
            axes[1, column].set_ylabel("Residual")

    fig.suptitle("Representative fibre-profile fits")
    return _save(fig, filename)


def plot_fibre_geometry_order(geometry, order_matrix, filename):
    """Full per-order QA: sparse measurements and compact polynomial model."""
    y = np.arange(order_matrix.flux.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    for ax, sampled, model, label in (
        (axes[0], geometry.sampled_bundle, geometry.bundle(y), "Bundle offset / pixel"),
        (axes[1], geometry.sampled_separation, geometry.separation(y), "Separation / pixel"),
        (axes[2], geometry.sampled_sigma, geometry.sigma(y), r"$\sigma$ / pixel"),
    ):
        good = np.isfinite(sampled)
        ax.plot(geometry.sampled_y[good], sampled[good], ".", label="local fits")
        ax.plot(y, model, "-", label="compact model")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Dispersion pixel")
    fig.suptitle(geometry.name.replace("ccd_", "CCD ").replace("_order_", " order "))
    return _save(fig, filename)




def _native_order_coordinates(order_matrix):
    """Return native detector x coordinates for an OrderMatrix."""
    n = order_matrix.flux.shape[0]
    trace = order_matrix.geometry.trace(n)
    centre_pixel = np.rint(trace - order_matrix.trace_offset).astype(int)
    x_native = centre_pixel[:, None] + order_matrix.relative_x[None, :]
    return trace, centre_pixel, x_native


def plot_fibre_extraction_detector(order_matrix, fibre_geometry, filename, rows=None, half_height=150):
    """Show fitted fibre centres over native detector pixels at three order locations."""
    n = order_matrix.flux.shape[0]
    if rows is None:
        # rows = [int(f * (n - 1)) for f in (0.1, 0.5, 0.9)]
        rows = [500, 2055, 3500]

    trace, centre_pixel, x_native = _native_order_coordinates(order_matrix)
    centres, sigma, _, _ = fibre_geometry.evaluate(n, order_matrix.trace_offset)
    centres_native = centre_pixel[:, None] + centres

    fig, axes = plt.subplots(1, len(rows), figsize=(10, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, ycentre in zip(axes, rows):
        y0, y1 = max(0, ycentre - half_height), min(n, ycentre + half_height)
        use = slice(y0, y1)

        xmin = int(np.nanmin(x_native[use]))
        xmax = int(np.nanmax(x_native[use]))
        image = np.full((y1 - y0, xmax - xmin + 1), np.nan)

        for j, y in enumerate(range(y0, y1)):
            x = x_native[y].astype(int) - xmin
            image[j, x] = order_matrix.flux[y]

        finite = image[np.isfinite(image)]
        vmin, vmax = np.nanpercentile(finite, [5, 99.5])

        ax.imshow(
            image, origin="lower", cmap="Greys", aspect="auto", interpolation="none",
            extent=(xmin - 0.5, xmax + 0.5, y0 - 0.5, y1 - 0.5),
            vmin=vmin, vmax=1.2*vmax,
        )

        science_label, sky_label = True, True
        for i, component in enumerate(fibre_geometry.components):
            science = isinstance(component, (int, np.integer))
            colour = "C4" if science else "C0"
            label = "Science fibres" if science and science_label else "Sky fibres" if not science and sky_label else None
            ax.plot(centres_native[use, i], np.arange(y0, y1), color=colour, lw=1.0, label=label)
            science_label &= not science
            sky_label &= science

        ax.plot(trace[use], np.arange(y0, y1), color="0.15", lw=1.0, label="Order trace")
        ax.plot(centre_pixel[use], np.arange(y0, y1), color="0.5", lw=1.0,
                drawstyle="steps-mid", label="OrderMatrix centre")
        ax.set(xlabel="Native cross-dispersion pixel", title=f"y = {ycentre}")

    axes[0].set_ylabel("Dispersion pixel")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"CCD{order_matrix.ccd} order {order_matrix.order}: native fibre extraction geometry")
    return _save(fig, filename)


def plot_fibre_extraction_rows(order_matrix, fibre_geometry, filename, rows=None):
    """Show observed profiles and the actual fibre model used for extraction."""
    from . import extraction

    n = order_matrix.flux.shape[0]
    if rows is None:
        # rows = [int(f * (n - 1)) for f in (0.1, 0.5, 0.9)]
        rows = [500, 2055, 3500]

    result = extraction.extract_fibre_order(order_matrix, fibre_geometry)
    _, centre_pixel, _ = _native_order_coordinates(order_matrix)
    centres, sigma, _, _ = fibre_geometry.evaluate(n, order_matrix.trace_offset)

    fig, axes = plt.subplots(
        2, len(rows), figsize=(13, 5), sharex="col", constrained_layout=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    for column, y in enumerate(rows):
        x = centre_pixel[y] + order_matrix.relative_x
        profiles = extraction.integrated_gaussian_cube(
            order_matrix.relative_x, centres[y:y + 1], sigma[y:y + 1]
        )[0]

        background = 0.0 if result.background is None else result.background[y]
        components = profiles * result.flux[y][None, :]
        model = background + np.nansum(components, axis=1)
        data = order_matrix.flux[y]

        axes[0, column].step(x, data, where="mid", color="0.2", lw=1.0, label="Flat")
        for i, component in enumerate(fibre_geometry.components):
            colour = "C4" if isinstance(component, (int, np.integer)) else "C0"
            axes[0, column].plot(x, components[:, i], color=colour, lw=0.5, alpha=0.45)

        axes[0, column].plot(x, model, color="C3", lw=1.1, label="extraction model")
        axes[0, column].set_title(f"y = {y}, σ = {sigma[y]:.2f} px")

        axes[1, column].step(x, data - model, where="mid", lw=0.8)
        axes[1, column].axhline(0, color="0.5", ls="--", lw=0.7)
        axes[1, column].set_xlabel("Native cross-dispersion pixel")

        if column == 0:
            axes[0, column].set_ylabel("Counts")
            axes[1, column].set_ylabel("Residual")

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"CCD{order_matrix.ccd} order {order_matrix.order}: fibre extraction profiles")
    return _save(fig, filename)


def _representative_product(products, ccd):
    names = list(products)
    name = _representative_name(names, ccd)
    return None if name is None else products[name]


def plot_flat_summed_response(products, filename):
    """Summed Flat, smooth illumination model and resulting 1D response."""
    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 5),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    for column, ccd in enumerate(("1", "2", "3")):
        product = _representative_product(products, ccd)
        if product is None:
            continue

        axes[0, column].plot(
            product.summed_flat,
            lw=0.8,
            label="summed Flat",
        )
        axes[0, column].plot(
            product.summed_smooth,
            color="C3",
            lw=1.0,
            label="smooth Flat",
        )

        axes[1, column].plot(
            product.summed_response,
            lw=0.8,
        )
        axes[1, column].axhline(
            1, color="0.5", ls="--", lw=0.8
        )
        axes[1, column].set_ylim(0.5, 1.5)

        axes[0, column].set_title(
            f"CCD{ccd} order {product.order}"
        )
        axes[0, column].legend(fontsize=8)
        axes[1, column].set_xlabel("Dispersion pixel")

        if column == 0:
            axes[0, column].set_ylabel("Flat counts")
            axes[1, column].set_ylabel("Response")

    fig.suptitle("Summed Flat response")
    return _save(fig, filename)


def plot_flat_fibre_response(products, filename):
    """Smooth individual-fibre Flats and their 1D responses."""
    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 5),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    for column, ccd in enumerate(("1", "2", "3")):
        product = _representative_product(products, ccd)
        if product is None or product.fibre_response is None:
            continue

        axes[0, column].plot(
            product.fibre_smooth,
            lw=0.7,
        )

        axes[1, column].plot(
            product.fibre_response,
            lw=0.6,
        )
        axes[1, column].axhline(
            1, color="0.5", ls="--", lw=0.8
        )
        axes[1, column].set_ylim(0.5, 1.5)

        axes[0, column].set_title(
            f"CCD{ccd} order {product.order}"
        )
        axes[1, column].set_xlabel("Dispersion pixel")

        if column == 0:
            axes[0, column].set_ylabel("Smooth fibre Flat")
            axes[1, column].set_ylabel("Response")

    fig.suptitle("Fibre-resolved Flat response")
    return _save(fig, filename)


def plot_flat_recombination_qa(products, fibre_geometries, filename):
    """Compare direct summed Flat with recombined science fibres."""
    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 5),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    for column, ccd in enumerate(("1", "2", "3")):
        product = _representative_product(products, ccd)
        if product is None or product.fibre_flat is None:
            continue

        geometry = fibre_geometries[product.name]
        science = [
            geometry.components.index(fibre)
            for fibre in geometry.components
            if isinstance(fibre, (int, np.integer))
        ]

        recombined = np.nansum(
            product.fibre_flat[:, science],
            axis=1,
        )
        summed = np.asarray(product.summed_flat, float)

        scale = np.nanmedian(summed / recombined)
        recombined *= scale

        ratio = np.divide(
            recombined,
            summed,
            out=np.full_like(summed, np.nan),
            where=np.isfinite(summed) & (summed != 0),
        )

        axes[0, column].plot(
            summed,
            lw=0.8,
            label="direct sum",
        )
        axes[0, column].plot(
            recombined,
            color="C3",
            lw=1.0,
            label="recombined fibres",
        )

        axes[1, column].plot(
            ratio - 1,
            lw=0.8,
        )
        axes[1, column].axhline(
            0, color="0.5", ls="--", lw=0.8
        )

        axes[0, column].set_title(
            f"CCD{ccd} order {product.order}"
        )
        axes[1, column].set_xlabel("Dispersion pixel")

        if column == 0:
            axes[0, column].set_ylabel("Flat counts")
            axes[1, column].set_ylabel(
                "Recombined / summed - 1"
            )

    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Flat fibre-recombination QA")
    return _save(fig, filename)


# The wavelength/science diagnostics below are retained so this file remains a
# drop-in replacement when the later pipeline stages are reconnected.
def plot_wavelength_fit(data, node, filename, calibration_type=None):
    used = np.asarray(data["used_surface"], bool) if "used_surface" in data.colnames else np.ones(len(data), bool)
    y = np.asarray(data["y"], float)[used]
    order = np.asarray(data["order"], float)[used]
    pixel = np.asarray(data["pixel_residual"], float)[used]
    velocity = np.asarray(data["velocity_residual_mps"], float)[used]
    phase = y - np.floor(y + 0.5)
    family = calibration_type or str(node.source).split(":")[0]
    rms_pixel = float(np.sqrt(np.nanmean(pixel ** 2)))
    rms_velocity = float(np.sqrt(np.nanmean(velocity ** 2)))
    limit = np.nanpercentile(np.abs(pixel), 99) if len(pixel) else 1.0
    limit = max(float(limit), 1e-6)
    velocity_limit = max(float(np.nanpercentile(np.abs(velocity), 99)) * 1.1, 1.0) if len(velocity) else 1.0

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    axes[0, 0].axis("off")
    axes[0, 0].text(0.5, 0.78, f"Wavelength solution {family} CCD{node.ccd}", ha="center", va="center", fontsize=16, fontweight="bold")
    axes[0, 0].text(0.5, 0.50, f"{node.coefficients.shape[0]-1} x {node.coefficients.shape[1]-1} Legendre degrees; {len(y):,} lines", ha="center")
    axes[0, 0].text(0.5, 0.25, f"RMS = {rms_pixel:.4f} px = {rms_velocity:.1f} m/s", ha="center", fontsize=14)

    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    scatter = axes[0, 1].scatter(y, order, c=pixel, s=14, cmap="RdBu_r", norm=norm, linewidths=0, rasterized=True)
    axes[0, 1].set(xlabel="Dispersion pixel", ylabel="Echelle order", title="2-D residual map")
    fig.colorbar(scatter, ax=axes[0, 1], label="Residual / pixel")

    unique_orders = np.unique(order)
    stats = []
    for m in unique_orders:
        values = pixel[order == m]
        stats.append((m, *np.nanpercentile(values, [16, 50, 84])))
    stats = np.asarray(stats, float)
    axes[1, 0].errorbar(stats[:, 2], stats[:, 0], xerr=np.vstack((stats[:, 2]-stats[:, 1], stats[:, 3]-stats[:, 2])), fmt="o", ms=4, capsize=2)
    axes[1, 0].axvline(0, ls="--", lw=1)
    axes[1, 0].set(xlabel="Residual / pixel", ylabel="Echelle order", title="Residual by order")

    axes[1, 1].scatter(y, velocity, s=10, alpha=0.35, linewidths=0, rasterized=True)
    centres, p16, p50, p84 = _binned_percentiles(y, velocity, 20)
    axes[1, 1].fill_between(centres, p16, p84, alpha=0.25); axes[1, 1].plot(centres, p50, lw=2)
    axes[1, 1].axhline(0, ls="--", lw=1); axes[1, 1].set_ylim(-velocity_limit, velocity_limit)
    axes[1, 1].set(xlabel="Dispersion pixel", ylabel="Residual / m/s", title="Residual vs dispersion")

    axes[2, 0].scatter(phase, velocity, s=10, alpha=0.35, linewidths=0, rasterized=True)
    centres, p16, p50, p84 = _binned_percentiles(phase, velocity, 15)
    axes[2, 0].fill_between(centres, p16, p84, alpha=0.25); axes[2, 0].plot(centres, p50, lw=2)
    axes[2, 0].axhline(0, ls="--", lw=1); axes[2, 0].set_xlim(-0.5, 0.5); axes[2, 0].set_ylim(-velocity_limit, velocity_limit)
    axes[2, 0].set(xlabel="Pixel phase", ylabel="Residual / m/s", title="Sub-pixel residual")

    axes[2, 1].hist(pixel[np.isfinite(pixel)], bins=35, histtype="step")
    axes[2, 1].axvline(0, ls="--", lw=1); axes[2, 1].axvline(-rms_pixel, ls=":", lw=1); axes[2, 1].axvline(rms_pixel, ls=":", lw=1)
    axes[2, 1].set(xlabel="Residual / pixel", ylabel="Lines", title=f"Residual distribution; RMS={rms_pixel:.4f} px")
    return _save(fig, filename, dpi=200)


def plot_wavelength_surface_validation(validation, *, filename=None):
    """Plot blocked-CV RMS and overfitting gap versus Legendre complexity."""
    y_degree = np.asarray(validation["y_degree"], int)
    m_degree = np.asarray(validation["order_degree"], int)
    value = np.asarray(validation["validation_rms_pixel"], float)
    train = np.asarray(validation["train_rms_pixel"], float)
    ys = np.unique(y_degree); ms = np.unique(m_degree)
    image = np.full((len(ms), len(ys)), np.nan)
    gap = np.full_like(image, np.nan)
    for i, md in enumerate(ms):
        for j, yd in enumerate(ys):
            q = (y_degree == yd) & (m_degree == md)
            if np.any(q):
                image[i, j] = value[q][0]
                gap[i, j] = value[q][0] - train[q][0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, z, title in [
        (axes[0], image, "Held-out RMS"),
        (axes[1], gap, "Held-out minus training RMS"),
    ]:
        im = ax.imshow(z, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax, label="pixel")
        ax.set_xticks(np.arange(len(ys)), ys)
        ax.set_yticks(np.arange(len(ms)), ms)
        ax.set_xlabel("Legendre degree in y")
        ax.set_ylabel("Legendre degree in order m")
        ax.set_title(title)
        for i in range(len(ms)):
            for j in range(len(ys)):
                if np.isfinite(z[i, j]):
                    ax.text(j, i, f"{z[i,j]:.3f}", ha="center", va="center", fontsize=7)
    if filename is not None:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    return fig


def plot_lc_drift(drift_by_ccd, filename):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for ccd, (mjd, velocity) in drift_by_ccd.items():
        if len(mjd): ax.plot(mjd, velocity, "o-", label=f"CCD{ccd}")
    ax.axhline(0, ls="--", lw=1); ax.set(xlabel="MJD", ylabel="SimLC drift / m/s", title="Laser-comb drift through the night")
    if drift_by_ccd: ax.legend()
    return _save(fig, filename)


def plot_fibre_displacement_qa(
    matched_lines,
    *,
    ccd,
    quantity="displacement",
    filename,
    minimum_snr=5.0,
    maximum_snr=100.0,
    dpi=300,
):
    """Plot measured fibre displacements or post-model residuals for one CCD."""
    ccd = str(ccd)

    if quantity == "displacement":
        value_name = "delta_y"
        colourbar_label = r"$\Delta y = y_{\rm fibre}-y_{\rm reference}$ / px"
        maximum = 1.0
    elif quantity == "residual":
        value_name = "residual_y"
        colourbar_label = r"$\Delta y-\Delta y_{\rm model}$ / px"
        maximum = 0.2
    else:
        raise ValueError("quantity must be 'displacement' or 'residual'")

    data = matched_lines[np.asarray(matched_lines["ccd"], str) == ccd]
    if len(data) == 0:
        raise ValueError(f"No matched fibre lines are available for CCD{ccd}")

    norm = TwoSlopeNorm(
        vmin=-maximum,
        vcenter=0.0,
        vmax=maximum,
    )

    with plt.rc_context({
        "font.size": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
    }):
        fig, axes = plt.subplots(
            5,
            4,
            figsize=(9 / 2.54, 12 / 2.54),
            sharex=True,
            sharey=True,
        )
        axes = axes.ravel()

        fig.subplots_adjust(
            left=0.12,
            right=0.98,
            bottom=0.08,
            top=0.88,
            wspace=0.08,
            hspace=0.08,
        )

        # --------------------------------------------------------------
        # Top-left: summed-spectrum S/N as black points (size only)
        # --------------------------------------------------------------

        peak_id = np.asarray(data["summed_peak_id"], int)
        _, first = np.unique(peak_id, return_index=True)
        summed = data[np.sort(first)]

        summed_snr = np.asarray(summed["summed_signal_to_noise"], float)
        good_summed = np.isfinite(summed_snr) & (summed_snr >= minimum_snr)

        ax = axes[0]
        ax.scatter(
            np.asarray(summed["y_reference"], float)[good_summed],
            np.asarray(summed["order"], float)[good_summed],
            s=0.01 * np.clip(
                summed_snr[good_summed],
                minimum_snr,
                maximum_snr,
            ),
            color="black",
            alpha=0.8,
            lw = 0,
            rasterized=True,
        )

        ax.text(
            0.5,
            0.95,
            "Summed",
            fontsize=6,
            transform=ax.transAxes,
            va="top",
            ha="center",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.5, pad=0.0)
        )

        # --------------------------------------------------------------
        # Remaining 19 panels: fibres
        # --------------------------------------------------------------

        scatter = None

        for ax, fibre in zip(axes[1:], range(1, 20)):
            fibre_data = data[np.asarray(data["fibre"], int) == fibre]

            fibre_values = np.asarray(fibre_data[value_name], float)
            fibre_snr = np.asarray(fibre_data["signal_to_noise"], float)

            scatter = ax.scatter(
                np.asarray(fibre_data["y_reference"], float),
                np.asarray(fibre_data["order"], float),
                c=fibre_values,
                s=0.1 * np.clip(
                    fibre_snr,
                    minimum_snr,
                    maximum_snr,
                ),
                lw = 0,
                cmap="coolwarm",
                norm=norm,
                rasterized=True,
            )

            rms = (
                float(np.sqrt(np.nanmean(fibre_values**2)))
                if np.any(np.isfinite(fibre_values))
                else np.nan
            )

            ax.text(
                0.5,
                0.95,
                f"Fibre {fibre}\nRMS = {rms:.3f} px",
                fontsize=6,
                transform=ax.transAxes,
                va="top",
                ha="center"
            )

            ax.set_xlim(-5, N_DISPERSION - 1 + 5)
            ax.set_xticks([0,2055.5,4111])

        # --------------------------------------------------------------
        # Shared colourbar across the top of the full figure
        # --------------------------------------------------------------

        if scatter is not None:
            cax = fig.add_axes([0.24, 0.945, 0.56, 0.018])
            cbar = fig.colorbar(
                scatter,
                cax=cax,
                orientation="horizontal",
            )
            cbar.set_label(colourbar_label, fontsize=6, labelpad=2)
            cbar.ax.tick_params(labelsize=6, pad=1)

        # --------------------------------------------------------------
        # Axis labels only at outer edges
        # --------------------------------------------------------------

        for row in range(5):
            axes[4 * row].set_ylabel("Order", fontsize=7)

        for ax in axes[-4:]:
            ax.set_xlabel(r"Dispersion $y$", fontsize=7)

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

    return filename


def plot_fibre_wavelength_offsets(offsets_by_ccd, filename):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True, constrained_layout=True)
    for ccd, ax in zip(("1", "2", "3"), axes):
        fibre, slot, velocity, rms = offsets_by_ccd.get(ccd, (np.array([]),) * 4)
        if len(fibre):
            ax.plot(slot, velocity, "o-")
            ax.set_xticks(slot); ax.set_xticklabels([str(x) for x in fibre], rotation=90, fontsize=7)
            ax.text(0.03, 0.97, f"median fit RMS {np.nanmedian(rms):.1f} m/s", transform=ax.transAxes, va="top", fontsize=8)
        ax.axhline(0, ls="--", lw=1); ax.set(title=f"CCD{ccd}", xlabel="Science fibre (slit order)")
    axes[0].set_ylabel("Fibre wavelength offset / m/s")
    fig.suptitle("Per-fibre FibTh wavelength corrections")
    return _save(fig, filename)


def plot_science_summary(exposures, filename):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    for column, ccd in enumerate(("1", "2", "3")):
        matches = [e for e in exposures if str(e.ccd) == ccd]
        if not matches:
            for ax in axes[:, column]: ax.axis("off")
            continue
        exposure = matches[0]
        orders = np.array([o.order for o in exposure.orders])
        snr = np.array([
            np.nanmedian(np.divide(o.flux, np.sqrt(o.variance), out=np.full_like(o.flux, np.nan, float), where=np.asarray(o.variance) > 0))
            for o in exposure.orders
        ])
        axes[0, column].plot(orders, snr, "o-"); axes[0, column].set(title=f"CCD{ccd} run {exposure.run}", xlabel="Echelle order")
        if column == 0: axes[0, column].set_ylabel("Median S/N per pixel")
        representative = exposure.orders[len(exposure.orders) // 2]
        axes[1, column].plot(representative.wavelength_nm, representative.flux, label="science")
        if representative.sky is not None: axes[1, column].plot(representative.wavelength_nm, representative.sky, label="sky")
        axes[1, column].set(xlabel="Wavelength / nm", title=f"Order {representative.order}")
        if column == 0: axes[1, column].set_ylabel("Counts")
        axes[1, column].legend(fontsize=8)
    fig.suptitle("Representative science-extraction QA")
    return _save(fig, filename)


def plot_fibre_profile_order(product, filename):
    observed = np.nanmedian(product.matrix, axis=0)
    model = np.nanmedian(product.smooth, axis=0)
    m = np.arange(len(observed)) - (len(observed) - 1) / 2
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, constrained_layout=True, height_ratios=[3, 1])
    axes[0].plot(m, observed, label="Flat"); axes[0].plot(m, model, label="24-fibre model")
    axes[0].set(ylabel="Counts", title=product.order_name.replace("ccd_", "CCD ").replace("_order_", " order "))
    axes[0].legend()
    axes[1].plot(m, observed - model); axes[1].axhline(0, ls="--", lw=1)
    axes[1].set(xlabel="Cross-dispersion pixel", ylabel="Residual")
    return _save(fig, filename)


# -----------------------------------------------------------------------------
# SimLC two-pass LSF diagnostics
# -----------------------------------------------------------------------------


def plot_simlc_lsf_diagnostics(peak_table, lsf_model, *, filename=None, show=False):
    """Compare first-pass shape measurements with the smooth optimized SimLC LSF."""
    if len(peak_table) == 0:
        return None

    y = np.asarray(peak_table["y_initial"], float)
    order = np.asarray(peak_table["order"], int)
    used_shape = np.asarray(peak_table["used_for_lsf_fit"], bool)
    fwhm_initial = np.asarray(peak_table["fwhm_initial"], float)
    fwhm_model = np.asarray(peak_table["fwhm_model"], float)
    fwhm_unc = np.asarray(peak_table["fwhm_initial_uncertainty"], float)

    finite_width = np.isfinite(fwhm_initial) & np.isfinite(fwhm_model)
    if np.any(finite_width):
        lo, hi = np.nanpercentile(
            np.r_[fwhm_initial[finite_width], fwhm_model[finite_width]], [2, 98]
        )
    else:
        lo, hi = 0.5, 2.0

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))

    ax = axes[0, 0]
    sc = ax.scatter(y[finite_width], order[finite_width], c=fwhm_initial[finite_width],
                    s=np.where(used_shape[finite_width], 9, 3), vmin=lo, vmax=hi)
    fig.colorbar(sc, ax=ax, label="Initial FWHM [pixel]")
    ax.set(xlabel=r"Dispersion pixel $y$", ylabel=r"Echelle order $m$",
           title="First-pass local width measurements")

    ax = axes[0, 1]
    sc = ax.scatter(y[finite_width], order[finite_width], c=fwhm_model[finite_width],
                    s=6, vmin=lo, vmax=hi)
    fig.colorbar(sc, ax=ax, label="Fitted FWHM [pixel]")
    ax.set(xlabel=r"Dispersion pixel $y$", ylabel=r"Echelle order $m$",
           title=r"Smooth fitted FWHM$(y,m)$")

    ax = axes[1, 0]
    normalized = np.divide(
        fwhm_initial-fwhm_model, fwhm_unc,
        out=np.full(len(peak_table), np.nan),
        where=np.isfinite(fwhm_unc) & (fwhm_unc > 0),
    )
    q = used_shape & np.isfinite(normalized)
    ax.scatter(y[q], normalized[q], s=7)
    ax.axhline(0, lw=0.8, ls="--")
    ax.set(xlabel=r"Dispersion pixel $y$", ylabel=r"$(\mathrm{FWHM}_{i}-\mathrm{FWHM}_{\rm model})/\sigma_i$",
           title="Width-model residuals used in the smooth fit")

    ax = axes[1, 1]
    orders = np.array(sorted(lsf_model.order_snr_thresholds), int)
    thresholds = np.array([lsf_model.order_snr_thresholds[o] for o in orders], float)
    nlines = np.array([lsf_model.order_n_shape_lines.get(o, 0) for o in orders], int)
    ax.plot(orders, thresholds, marker="o", lw=1, label="adaptive S/N threshold")
    ax.set(xlabel=r"Echelle order $m$", ylabel="Minimum S/N", title="LSF-training selection by order")
    ax2 = ax.twinx()
    ax2.plot(orders, nlines, marker=".", lw=0.8, alpha=0.65, label="shape lines")
    ax2.set_ylabel("Number of LSF-training lines")

    ax = axes[2, 0]
    phase = np.asarray(peak_table["y_initial"], float) - np.round(np.asarray(peak_table["y_initial"], float))
    shift = np.asarray(peak_table["lsf_y_shift"], float)
    q = np.isfinite(phase) & np.isfinite(shift)
    ax.scatter(phase[q], shift[q], s=5, alpha=0.45)
    centres, p16, p50, p84 = _binned_percentiles(phase[q], shift[q], n_bins=16, minimum=5)
    good = np.isfinite(p50)
    if np.any(good):
        ax.plot(centres[good], p50[good], lw=1.5)
        ax.fill_between(centres[good], p16[good], p84[good], alpha=0.15)
    ax.axhline(0, lw=0.8, ls="--")
    ax.set(xlabel="Initial pixel phase", ylabel=r"Optimized $y-y_{\rm initial}$ [pixel]",
           title="Centroid correction")

    ax = axes[2, 1]
    initial_unc = np.asarray(peak_table["y_uncertainty_initial"], float)
    final_unc = np.asarray(peak_table["y_uncertainty"], float)
    snr = np.asarray(peak_table["signal_to_noise"], float)
    q = np.isfinite(initial_unc) & np.isfinite(final_unc) & np.isfinite(snr)
    ax.scatter(snr[q], initial_unc[q], s=5, alpha=0.35, label="initial Gaussian")
    ax.scatter(snr[q], final_unc[q], s=5, alpha=0.35, label="optimized LSF")
    ax.set(xscale="log", yscale="log", xlabel="Line S/N", ylabel=r"$\sigma_y$ [pixel]",
           title="Centroid uncertainty")
    ax.legend(fontsize=8)

    # Add the order-only shape parameter(s) as compact text rather than adding
    # another panel; the fitted values are also stored line-by-line in FITS.
    if lsf_model.shape == "moffat":
        mgrid = np.linspace(np.nanmin(order), np.nanmax(order), 8)
        qbeta = lsf_model.parameter("one_over_beta", mgrid)
        shape_text = r"Moffat: fitted $1/\beta(m)$ = " + f"{np.nanmin(qbeta):.3f}--{np.nanmax(qbeta):.3f}"
    elif lsf_model.shape == "core_wing_gaussians":
        mgrid = np.linspace(np.nanmin(order), np.nanmax(order), 8)
        qwing = lsf_model.parameter("wing_fraction", mgrid)
        ratio = lsf_model.parameter("wing_sigma_ratio", mgrid)
        shape_text = (
            f"Core+wing: q(m)={np.nanmin(qwing):.2f}--{np.nanmax(qwing):.2f}; "
            f"sigma_wing/sigma_core={np.nanmin(ratio):.2f}--{np.nanmax(ratio):.2f}"
        )
    else:
        shape_text = "Integrated Gaussian: FWHM(y,m) only"

    fig.suptitle(f"SimLC LSF optimization: {lsf_model.shape}\n{shape_text}")
    fig.tight_layout()
    if filename is not None:
        _save(fig, filename)
    elif show:
        plt.show()
    else:
        plt.close(fig)
    return filename if filename is not None else fig


def save_simlc_fit_examples(
    counts, orders, peak_table, lsf_model, *, filename, variance=None,
    fit_half_width=3, maximum_examples=16,
):
    """Save representative first-pass and optimized SimLC line fits."""
    from .utils import pixel_integrated_lsf

    counts = np.asarray(counts, float)
    orders = np.asarray(orders, int)
    lookup = {int(order): i for i, order in enumerate(orders)}
    available = (
        np.asarray(peak_table["lsf_initial_fit_success"], bool)
        & np.isfinite(np.asarray(peak_table["y"], float))
        & (np.asarray(peak_table["comb_mode"], np.int64) >= 0)
    )
    idx = np.flatnonzero(available)
    if len(idx) == 0:
        return None
    # Sort in detector/order space and choose evenly so examples span the CCD.
    sort = np.lexsort((np.asarray(peak_table["y_initial"], float)[idx], np.asarray(peak_table["order"], int)[idx]))
    idx = idx[sort]
    if len(idx) > maximum_examples:
        idx = idx[np.unique(np.linspace(0, len(idx)-1, maximum_examples).round().astype(int))]

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(filename) as pdf:
        ncol = 2
        nrow = 4
        per_page = ncol*nrow
        for start in range(0, len(idx), per_page):
            selected = idx[start:start+per_page]
            fig, axes = plt.subplots(nrow, ncol, figsize=(12, 11), squeeze=False)
            for ax in axes.ravel():
                ax.set_visible(False)
            for ax, i in zip(axes.ravel(), selected):
                ax.set_visible(True)
                row = peak_table[i]
                order = int(row["order"])
                center_pixel = int(round(float(row["y_initial"])))
                pix = np.arange(center_pixel-fit_half_width, center_pixel+fit_half_width+1)
                if pix[0] < 0 or pix[-1] >= counts.shape[1] or order not in lookup:
                    continue
                signal = counts[lookup[order], pix]
                ax.scatter(pix, signal, s=18, label="data")

                # Free first-pass LSF fit.
                initial_kwargs = dict(fwhm=float(row["fwhm_initial"]))
                if lsf_model.shape == "moffat":
                    initial_kwargs["one_over_beta"] = float(row["one_over_beta_initial"])
                elif lsf_model.shape == "core_wing_gaussians":
                    initial_kwargs["wing_fraction"] = float(row["wing_fraction_initial"])
                    initial_kwargs["wing_sigma_ratio"] = float(row["wing_sigma_ratio_initial"])
                p0 = pixel_integrated_lsf(
                    pix-float(row["y_lsf_initial"]), lsf_model.shape, **initial_kwargs
                )
                yref0 = float(round(float(row["y_lsf_initial"])))
                model0 = (
                    float(row["lsf_background_initial"])
                    + float(row["lsf_background_slope_initial"])*(pix-yref0)
                    + float(row["lsf_integrated_counts_initial"])*p0
                )
                ax.plot(pix, model0, ls="--", lw=1.1, label="first LSF fit")

                # Final centroid fit with smooth shape field.
                final_kwargs = dict(fwhm=float(row["fwhm_model"]))
                if lsf_model.shape == "moffat":
                    final_kwargs["one_over_beta"] = float(row["one_over_beta_model"])
                elif lsf_model.shape == "core_wing_gaussians":
                    final_kwargs["wing_fraction"] = float(row["wing_fraction_model"])
                    final_kwargs["wing_sigma_ratio"] = float(row["wing_sigma_ratio_model"])
                pf = pixel_integrated_lsf(pix-float(row["y"]), lsf_model.shape, **final_kwargs)
                yreff = float(round(float(row["y"])))
                modelf = (
                    float(row["background"])
                    + float(row["background_slope"])*(pix-yreff)
                    + float(row["integrated_counts"])*pf
                )
                ax.plot(pix, modelf, lw=1.2, label="optimized centroid")
                ax.axvline(float(row["y_initial"]), ls=":", lw=0.8)
                ax.axvline(float(row["y"]), ls="-.", lw=0.8)
                ax.set_title(
                    f"m={order}, y={float(row['y']):.3f}, S/N={float(row['signal_to_noise']):.1f}\n"
                    f"FWHM={float(row['fwhm_model']):.3f} pix",
                    fontsize=9,
                )
                ax.set(xlabel=r"Dispersion pixel $y$", ylabel="Counts")
            axes.ravel()[0].legend(fontsize=7)
            fig.suptitle(f"SimLC first-pass vs optimized fits: {lsf_model.shape}")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return filename
