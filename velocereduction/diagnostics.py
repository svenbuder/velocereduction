"""Diagnostic plots for inspecting each reduction stage."""
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
import numpy as np

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
    fig.suptitle("Detector registration relative to 001122")
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
        subset = sorted(
            [g for g in geometries if g.ccd == ccd],
            key=lambda g: g.order,
        )

        order = np.array([g.order for g in subset])
        rms = np.array([g.trace_rms for g in subset])

        axes[0, column].plot(order, rms, "o-", lw=1)
        axes[0, column].set_title(f"CCD{ccd}")

        if column == 0:
            axes[0, column].set_ylabel("Trace RMS / pixel")

        for region in (
            "SimTh", "Sky_1", "Science", "Sky_2", "SimLC"
        ):
            begin = np.array([
                g.regions.get(region, (np.nan, np.nan))[0]
                for g in subset
            ])
            end = np.array([
                g.regions.get(region, (np.nan, np.nan))[1]
                for g in subset
            ])

            colour = REGION_COLOURS[region]
            label = REGION_LABELS[region]

            axes[1, column].plot(
                order,
                begin,
                color=colour,
                lw=1.0,
                label=label,
            )
            axes[1, column].plot(
                order,
                end,
                color=colour,
                lw=1.0,
            )
            axes[1, column].fill_between(
                order,
                begin,
                end,
                color=colour,
                alpha=0.08,
            )

        axes[1, column].set_xlabel("Echelle order")

        if column == 0:
            axes[1, column].set_ylabel(
                "Relative cross-dispersion pixel"
            )

    axes[1, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Order geometry")
    return _save(fig, filename)


def plot_order_matrix_examples(combined_flats, geometries, filename):
    """Representative 81-pixel OrderMatrix and extraction regions."""
    from . import orders

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 6),
        constrained_layout=True,
    )

    for column, ccd in enumerate(("1", "2", "3")):
        subset = sorted(
            [g for g in geometries if g.ccd == ccd],
            key=lambda g: g.order,
        )
        if not subset:
            continue

        geometry = subset[len(subset) // 2]
        matrix = orders.extract_order_matrix(
            combined_flats[ccd],
            geometry,
        )

        finite = matrix.flux[np.isfinite(matrix.flux)]
        vmin, vmax = (
            np.nanpercentile(finite, [5, 99])
            if finite.size else (0, 1)
        )

        axes[0, column].imshow(
            matrix.flux,
            origin="lower",
            aspect="auto",
            cmap="Greys",
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
        )

        axes[0, column].set_title(
            f"CCD{ccd} order {geometry.order}"
        )

        profile = np.nanmedian(matrix.flux, axis=0)
        axes[1, column].plot(
            matrix.relative_x,
            profile,
            color="0.2",
            lw=1,
        )

        for region in (
            "SimTh", "Sky_1", "Science", "Sky_2", "SimLC"
        ):
            if region not in geometry.regions:
                continue

            begin, end = geometry.regions[region]
            axes[1, column].axvspan(
                begin,
                end,
                color=REGION_COLOURS[region],
                alpha=0.18,
                lw=0,
                label=REGION_LABELS[region],
            )

        axes[1, column].set_xlabel(
            "Relative cross-dispersion pixel"
        )

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


#
# LEGACY CODE
#


REGION_COLOURS = {
    "SimTh": "C1", "Sky_1": "C0", "Science": "C4",
    "Sky_2": "C0", "SimLC": "C3",
}


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


def plot_summed_response_image(products, filename):
    """Display summed Flat response versus dispersion pixel and order."""
    fig, axes = plt.subplots(
        1, 3,
        figsize=(13, 4),
        constrained_layout=True,
    )

    image = None

    for column, ccd in enumerate(("1", "2", "3")):
        subset = sorted(
            [p for p in products.values() if p.ccd == ccd],
            key=lambda p: p.order,
        )

        if not subset:
            axes[column].axis("off")
            continue

        response = np.stack([
            p.summed_response
            for p in subset
        ])

        physical_orders = np.array([
            p.order for p in subset
        ])

        image = axes[column].imshow(
            response,
            origin="lower",
            cmap="Greys",
            aspect="auto",
            interpolation="none",
            vmin=0.9,
            vmax=1.1,
            extent=(
                -0.5,
                response.shape[1] - 0.5,
                physical_orders[0] - 0.5,
                physical_orders[-1] + 0.5,
            ),
        )

        axes[column].set(
            title=f"CCD{ccd}",
            xlabel="Dispersion pixel",
        )

        if column == 0:
            axes[column].set_ylabel("Echelle order")

    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=axes,
            pad=0.02,
            fraction=0.025,
        )
        cbar.set_label(
            r"$F_{\rm flat}/\widetilde{F}_{\rm flat}$"
        )

    fig.suptitle("Summed Flat response")
    return _save(fig, filename)
    
    
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
    if np.any(good):
        ax.scatter(order_table["y"][good], np.clip(order_table["fwhm"][good], 0, 4),
                   s=10, label="Accepted")
    if np.any(~good):
        ax.scatter(order_table["y"][~good], np.clip(order_table["fwhm"][~good], 0, 4),
                   marker="x", s=20, label="Rejected")
    width_reference = np.asarray(order_table["fwhm"][good], dtype=float)
    if np.any(np.isfinite(width_reference)):
        ax.axhline(np.nanmedian(width_reference), ls="--", lw=1)
    ax.set_ylabel("FWHM [pixel]")

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
    fig.text(
        0.99, 0.01,
        f"candidates={len(candidate_pixels)} | fitted={len(order_table)} | "
        f"accepted={summary['accepted']} | sat={summary.get('saturated', 0)} | "
        f"width={summary.get('width_outlier', 0)} | "
        f"blend={summary.get('blend_candidate', 0)} | lowS/N={summary.get('low_snr', 0)}",
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
    exposure_index, diagnostic_dir, fibre=-1,
):
    """Write the multi-page v0.7-style per-order calibration QA PDF."""
    diagnostic_dir = Path(diagnostic_dir)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    fibre_suffix = "" if int(fibre) == -1 else f"_fibre{int(fibre):+03d}"
    filename = diagnostic_dir / (
        f"{str(calibration_type).lower()}_ccd{ccd}_exposure{int(exposure_index):03d}"
        f"{fibre_suffix}_peaks.pdf"
    )
    orders = np.unique(np.asarray(peak_table["order"], int))
    with PdfPages(filename) as pdf:
        for order in orders:
            subset = peak_table[np.asarray(peak_table["order"], int) == int(order)]
            diag = order_diagnostics.get(int(order))
            if diag is None or len(subset) == 0:
                continue
            fig = plot_calibration_order_diagnostic(
                diag["counts"], diag["background"], diag["detection_snr"],
                diag["candidate_pixels"], subset, config=config,
                calibration_type=calibration_type, ccd=ccd,
                exposure_index=exposure_index, order=int(order), fibre=fibre,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            worst = plot_worst_peak_fits(
                diag["counts"], subset, config=config,
                calibration_type=calibration_type, ccd=ccd,
                exposure_index=exposure_index, order=int(order), fibre=fibre,
            )
            if worst is not None:
                pdf.savefig(worst, bbox_inches="tight")
                plt.close(worst)
    return filename


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
