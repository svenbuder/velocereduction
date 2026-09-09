from pathlib import Path
import logging

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

logger = logging.getLogger(__name__)

REGION_COLOURS = {
    "SimTh": "C1", "Sky_1": "C0", "Science": "C4",
    "Sky_2": "C0", "SimLC": "C3",
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


def plot_detector_shifts(detector_shifts, filename):
    ccd = np.asarray(detector_shifts["ccd"], int)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for ax, key, scatter_key, label in (
        (axes[0], "dx", "dx_scatter", r"$\Delta x$ / pixel"),
        (axes[1], "dy", "dy_scatter", r"$\Delta y$ / pixel"),
    ):
        values = np.asarray(detector_shifts[key], float)
        error = np.asarray(detector_shifts[scatter_key], float)
        error = np.where(np.isfinite(error), error, 0.0)
        ax.errorbar(ccd, values, yerr=error, fmt="o", capsize=3)
        ax.axhline(0, ls="--", lw=1)
        ax.set(xticks=ccd, xlabel="CCD", ylabel=label)
    fig.suptitle("Detector registration relative to 001122")
    return _save(fig, filename)


def plot_read_noise(read_noise_table, filename):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True, constrained_layout=True)
    for ccd, ax in zip((1, 2, 3), axes):
        use = np.asarray(read_noise_table["ccd"], int) == ccd
        subset = read_noise_table[use]
        for amp in sorted(set(np.asarray(subset["amplifier"]).astype(str))):
            rows = subset[np.asarray(subset["amplifier"]).astype(str) == amp]
            order = np.argsort(np.asarray(rows["run"], int))
            ax.plot(np.asarray(rows["run"], int)[order], np.asarray(rows["rms_adu"], float)[order], ".-", label=amp)
        ax.set(title=f"CCD{ccd}", xlabel="Flat run")
        if len(subset):
            ax.legend(fontsize=8)
    axes[0].set_ylabel("Overscan RMS / ADU")
    fig.suptitle("Read-noise monitoring from Flat overscans")
    return _save(fig, filename)


def plot_tramline_diagnostic(flat_image, simth_image, simlc_image, row, filename):
    """3-column x 2-row diagnostic retained from the original pipeline design."""
    from . import tramlines

    half = int(row["extraction_half_window"])
    name = tramlines.order_name(row)
    ccd, order = tramlines.ccd_from_order_name(name), tramlines.physical_order(name)
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7), sharex="col", constrained_layout=True,
        height_ratios=[3, 1], width_ratios=[1, 4, 1],
    )
    fig.suptitle(f"Tramline Fits to CCD {ccd} Order {order}")

    for column, (source, image) in enumerate((('SimTh', simth_image), ('Flat', flat_image), ('SimLC', simlc_image))):
        ax_image, ax_profile = axes[:, column]
        ax_image.set_title(source)
        if source == "Flat":
            x_min, x_max = 0, 2 * half
        else:
            begin, end = float(row[f"{source}_begin"]), float(row[f"{source}_end"])
            centre = half + 0.5 * (begin + end)
            x_min, x_max = centre - 6, centre + 6
        ax_profile.set_xlim(x_min, x_max)

        if image is None:
            for ax in (ax_image, ax_profile):
                ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=ax.transAxes)
            continue

        matrix, _ = tramlines.extract_order_matrix(image, row)
        profile = tramlines.collapsed_profile(matrix)
        region = matrix[:, max(0, int(np.floor(x_min))):min(matrix.shape[1], int(np.ceil(x_max)) + 1)]
        values = region[np.isfinite(region)]
        if values.size:
            vmin = np.nanpercentile(values, 5 if source == "Flat" else 20)
            vmax = np.nanpercentile(values, 95 if source == "Flat" else 99.5)
        else:
            vmin, vmax = 0.0, 1.0
        ax_image.imshow(matrix, origin="lower", aspect="auto", cmap="Greys_r", vmin=vmin, vmax=vmax)
        ax_image.set_xlim(x_min, x_max)
        ax_profile.plot(profile / 1e3, color="k", lw=1)

        regions = ("Sky_1", "Science", "Sky_2") if source == "Flat" else (source,)
        for region_name in regions:
            begin, end = float(row[f"{region_name}_begin"]), float(row[f"{region_name}_end"])
            if not (np.isfinite(begin) and np.isfinite(end)):
                continue
            plot_begin, plot_end = begin + half, end + half
            if source == "Flat":
                plot_begin += 0.15; plot_end -= 0.15
            colour = REGION_COLOURS[region_name]
            ax_image.axvline(plot_begin, color=colour, lw=1, ls="--")
            ax_image.axvline(plot_end, color=colour, lw=1, ls="--")
            ax_profile.axvspan(plot_begin, plot_end, color=colour, alpha=0.25, lw=0, label=region_name)
        handles, _ = ax_profile.get_legend_handles_labels()
        if handles:
            ax_profile.legend(fontsize=8, loc="lower center", ncol=3)

    axes[0, 0].set_ylabel("Dispersion pixel")
    axes[1, 0].set_ylabel(r"Counts / $10^3$")
    for column in range(3):
        ax = axes[1, column]
        xmin, xmax = ax.get_xlim()
        first = int(np.ceil((xmin - half) / 4)) * 4
        last = int(np.floor((xmax - half) / 4)) * 4
        ticks = np.arange(first, last + 1, 4)
        ax.set_xticks(ticks + half)
        ax.set_xticklabels([f"{value:+d}" for value in ticks])
        ax.set_xlabel("Cross-dispersion pixel")
    return _save(fig, filename)


def save_tramline_diagnostics(master_flat, calibration_images, nightly_tramlines, config, paths):
    if config.diagnostics == "none":
        return []
    names = [str(row["order_name"].decode() if isinstance(row["order_name"], bytes) else row["order_name"]) for row in nightly_tramlines]
    selected = set()
    if config.diagnostics == "basic":
        selected.update(filter(None, (_representative_name(names, ccd) for ccd in ("1", "2", "3"))))
    else:
        selected.update(names)
    files = []
    for row in nightly_tramlines:
        name = row["order_name"].decode() if isinstance(row["order_name"], bytes) else str(row["order_name"])
        if name not in selected:
            continue
        ccd, order = name.split("_")[1], name.split("_")[-1]
        directory = paths.figures if config.diagnostics == "basic" else paths.debug / "tramlines"
        files.append(plot_tramline_diagnostic(
            master_flat[f"ccd_{ccd}"], calibration_images.get(("SimTh", ccd)), calibration_images.get(("SimLC", ccd)),
            row, directory / f"tramline_ccd{ccd}_order{order}.png",
        ))
    return files


def plot_flat_response_summary(flat_products, filename):
    names = list(flat_products)
    fig, axes = plt.subplots(3, 3, figsize=(13, 7), sharex="col", constrained_layout=True)
    for column, ccd in enumerate(("1", "2", "3")):
        name = _representative_name(names, ccd)
        if name is None:
            for ax in axes[:, column]: ax.axis("off")
            continue
        product = flat_products[name]
        for row, (label, data) in enumerate((("Extracted Flat", product.matrix), ("Smooth model", product.smooth), ("Response", product.response))):
            ax = axes[row, column]
            values = np.asarray(data, float)
            finite = values[np.isfinite(values)]
            if row < 2:
                vmin, vmax = np.nanpercentile(finite, [5, 95]) if finite.size else (0, 1)
            else:
                scatter = 1.4826 * np.nanmedian(np.abs(finite - np.nanmedian(finite))) if finite.size else 0.01
                span = min(max(4 * scatter, 0.01), 0.10)
                vmin, vmax = 1 - span, 1 + span
            ax.imshow(values.T, origin="lower", aspect="auto", cmap="Greys_r", vmin=vmin, vmax=vmax)
            if column == 0: ax.set_ylabel(f"{label}\nCross-dispersion pixel")
            if row == 0: ax.set_title(name.replace("ccd_", "CCD ").replace("_order_", " order "))
            if row == 2: ax.set_xlabel("Dispersion pixel")
    fig.suptitle("Flat response diagnostics")
    return _save(fig, filename)


def plot_fibre_geometry_summary(flat_products, filename):
    names = list(flat_products)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex="col", constrained_layout=True)
    for column, ccd in enumerate(("1", "2", "3")):
        rows = []
        for name in names:
            product = flat_products[name]
            if str(name).split("_")[1] != ccd or product.geometry is None:
                continue
            g = product.geometry
            rows.append((int(str(name).split("_")[-1]), np.nanmedian(g.sigma), np.nanpercentile(g.sigma, 16), np.nanpercentile(g.sigma, 84),
                         np.nanmedian(g.separation), np.nanpercentile(g.separation, 16), np.nanpercentile(g.separation, 84)))
        if not rows:
            for ax in axes[:, column]: ax.axis("off")
            continue
        rows = np.asarray(sorted(rows), float)
        for ax, middle, lower, upper, ylabel in (
            (axes[0, column], 1, 2, 3, r"$\sigma$ / pixel"),
            (axes[1, column], 4, 5, 6, "Separation / pixel"),
        ):
            ax.errorbar(rows[:, 0], rows[:, middle], yerr=np.vstack((rows[:, middle] - rows[:, lower], rows[:, upper] - rows[:, middle])), fmt="o", ms=3, capsize=1)
            ax.set_ylabel(ylabel if column == 0 else "")
        axes[0, column].set_title(f"CCD{ccd}")
        axes[1, column].set_xlabel("Echelle order")
    fig.suptitle("Flat-derived fibre geometry")
    return _save(fig, filename)


def plot_fibre_profile_summary(flat_products, filename):
    names = list(flat_products)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex="col", constrained_layout=True)
    for column, ccd in enumerate(("1", "2", "3")):
        name = _representative_name(names, ccd)
        product = None if name is None else flat_products[name]
        if product is None or product.geometry is None:
            for ax in axes[:, column]: ax.axis("off")
            continue
        observed = np.nanmedian(product.matrix, axis=0)
        model = np.nanmedian(product.smooth, axis=0)
        m = np.arange(len(observed)) - (len(observed) - 1) / 2
        axes[0, column].plot(m, observed, label="Flat")
        axes[0, column].plot(m, model, label="model")
        axes[0, column].set_title(name.replace("ccd_", "CCD ").replace("_order_", " order "))
        axes[0, column].legend(fontsize=8)
        axes[1, column].plot(m, observed - model)
        axes[1, column].axhline(0, ls="--", lw=1)
        axes[1, column].set_xlabel("Cross-dispersion pixel")
        if column == 0:
            axes[0, column].set_ylabel("Counts")
            axes[1, column].set_ylabel("Flat - model")
    fig.suptitle("Representative 24-fibre Flat profile fits")
    return _save(fig, filename)


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


def plot_lc_drift(drift_by_ccd, filename):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for ccd, (mjd, velocity) in drift_by_ccd.items():
        if len(mjd): ax.plot(mjd, velocity, "o-", label=f"CCD{ccd}")
    ax.axhline(0, ls="--", lw=1); ax.set(xlabel="MJD", ylabel="SimLC drift / m/s", title="Laser-comb drift through the night")
    if drift_by_ccd: ax.legend()
    return _save(fig, filename)


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


def plot_fibre_extraction_qa(qa, geometry, filename, title=None):
    """Plot per-order fibre/summed fidelity and the adopted fibre geometry.

    The top panel compares the raw recombined/summed ratio with its broad smooth
    trend.  The middle panel isolates small-scale structure introduced by fibre
    deblending.  The bottom panel shows the Flat-derived sigma and separation
    used for the science extraction.  Intended primarily for full/debug QA.
    """
    ratio = np.asarray(qa["ratio"], float)
    smooth = np.asarray(qa["ratio_smooth"], float)
    structure = np.asarray(qa["fractional_structure"], float)
    x = np.arange(ratio.size)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    axes[0].plot(x, ratio, lw=0.8, label="recombined / summed")
    axes[0].plot(x, smooth, lw=1.5, label="broad trend")
    axes[0].set_ylabel("Flux ratio")
    axes[0].legend(fontsize=8)

    rms = float(qa["robust_rms_fractional_structure"])
    axes[1].plot(x, 100.0 * structure, lw=0.8)
    axes[1].axhline(0, ls="--", lw=1)
    if np.isfinite(rms):
        axes[1].axhline(100.0 * rms, ls=":", lw=1)
        axes[1].axhline(-100.0 * rms, ls=":", lw=1)
    axes[1].set_ylabel("Small-scale\nstructure / %")
    axes[1].set_title(f"robust RMS = {100.0 * rms:.4f}%")

    sigma = np.asarray(geometry.sigma, float)
    separation = np.asarray(geometry.separation, float)
    axes[2].plot(np.arange(sigma.size), sigma, label=r"$\sigma$")
    axes[2].plot(np.arange(separation.size), separation, label="separation")
    axes[2].set(xlabel="Dispersion pixel", ylabel="Geometry / pixel")
    axes[2].legend(fontsize=8)

    fig.suptitle(title or "Fibre-extraction fidelity")
    return _save(fig, filename)
