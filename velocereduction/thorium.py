"""Thorium-line measurement and identification for Veloce FibTh/SimTh.

This module owns the source-specific steps before wavelength fitting: reading
and curating the Th-only Murphy atlas, measuring emission peaks, identifying
reference lines, and returning the common calibration-line table consumed by
``wavelength.py``.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table

from .calibration import (
    CalibrationLineSet,
    CalibrationPeakConfig,
    CalibrationPeakFlag,
    _clear_identification,
    _match_peak_table_to_predicted_lines,
    _predict_reference_lines_for_order,
    _refresh_used_for_wavelength_fit,
    calibration_quality_summary,
    measure_calibration_peaks,
)

def load_murphy_thorium_atlas(filename: str | Path) -> Table:
    """Read the Murphy UVES ThAr atlas and retain only thorium lines.

    The attached Murphy file contains laboratory wavenumber, air wavelength,
    intensity, species/ion, and source.  Vacuum wavelength is calculated
    directly from the wavenumber:

        wavelength_vacuum_nm = 1e7 / wavenumber_cm^-1

    The original Murphy file should remain unchanged.  A Veloce-specific
    ``use_veloce`` flag can be saved separately with
    ``write_veloce_thorium_atlas``.
    """

    rows = []

    with open(filename, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            values = line.split()
            if len(values) < 6:
                continue

            try:
                wavenumber_cm = float(values[0])
                air_wavelength_angstrom = float(values[1])
                reference_intensity = float(values[2])
            except ValueError:
                continue

            element = values[3]
            ion = values[4]
            source = values[5]

            if element != "Th":
                continue

            wavelength_nm = 1e7 / wavenumber_cm

            rows.append(
                dict(
                    reference_id=f"MM090311_{line_number:05d}",
                    wavelength_nm=float(wavelength_nm),
                    wavelength_uncertainty_nm=np.nan,
                    wavenumber_cm=float(wavenumber_cm),
                    air_wavelength_angstrom=float(air_wavelength_angstrom),
                    reference_intensity=float(reference_intensity),
                    species=f"{element} {ion}",
                    reference_source=str(source),
                    use_veloce=True,
                    manual_flag="",
                    note="",
                )
            )

    return Table(rows=rows)

def write_veloce_thorium_atlas(
    atlas: Table,
    filename: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save the editable Veloce-specific thorium reference table as FITS."""

    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "velocereduction"
    primary.header["CONTENT"] = "Veloce thorium reference atlas"
    primary.header["SOURCE"] = "Murphy MM090311-derived thorium lines"

    atlas_hdu = fits.table_to_hdu(atlas)
    atlas_hdu.name = "THORIUM"

    fits.HDUList([primary, atlas_hdu]).writeto(
        filename,
        overwrite=overwrite,
    )

def read_veloce_thorium_atlas(filename: str | Path) -> Table:
    """Read a Veloce-specific thorium reference FITS table."""

    with fits.open(filename) as hdul:
        return Table(hdul["THORIUM"].data)

def identify_thorium_lines(
    peak_table: Table,
    thorium_atlas: Table,
    *,
    reference_wavelength_function,
    detector_shift_y: float,
    y_bounds: tuple[float, float],
    config: CalibrationPeakConfig | None = None,
    minimum_reference_intensity: float | None = None,
    initial_matching_radius_pixel: float = 3.0,
    final_matching_radius_pixel: float = 1.5,
) -> tuple[Table, float]:
    """Match SimTh/FibTh peaks to selected thorium laboratory lines."""

    if config is None:
        config = CalibrationPeakConfig()

    peak_table = peak_table.copy(copy_data=True)
    _clear_identification(peak_table)

    atlas = thorium_atlas.copy(copy_data=True)
    if "use_veloce" in atlas.colnames:
        atlas = atlas[np.asarray(atlas["use_veloce"], dtype=bool)]
    if minimum_reference_intensity is not None:
        atlas = atlas[
            np.asarray(atlas["reference_intensity"], dtype=float)
            >= float(minimum_reference_intensity)
        ]

    predicted_by_order: dict[int, Table] = {}

    for order in np.unique(np.asarray(peak_table["order"], dtype=int)):
        predicted_by_order[int(order)] = _predict_reference_lines_for_order(
            atlas,
            int(order),
            reference_wavelength_function=reference_wavelength_function,
            detector_shift_y=detector_shift_y,
            y_bounds=y_bounds,
        )

    calibration_shift_y = _match_peak_table_to_predicted_lines(
        peak_table,
        predicted_by_order,
        initial_matching_radius_pixel=initial_matching_radius_pixel,
        final_matching_radius_pixel=final_matching_radius_pixel,
    )

    # Reference-atlas blend rejection.  A thorium reference line is rejected
    # if the nearest other atlas line is closer than a configurable multiple
    # of the typical measured FWHM in that order.
    orders = np.asarray(peak_table["order"], dtype=int)

    for order, predicted in predicted_by_order.items():
        if len(predicted) < 2:
            continue

        predicted_y = (
            np.asarray(predicted["y_expected"], dtype=float)
            + calibration_shift_y
        )

        sort_index = np.argsort(predicted_y)
        sorted_y = predicted_y[sort_index]
        nearest_sorted = np.full(len(sorted_y), np.inf)
        separation = np.diff(sorted_y)
        nearest_sorted[:-1] = np.minimum(nearest_sorted[:-1], separation)
        nearest_sorted[1:] = np.minimum(nearest_sorted[1:], separation)

        reference_nearest = np.full(len(predicted_y), np.inf)
        reference_nearest[sort_index] = nearest_sorted

        measured_indices = np.where(orders == int(order))[0]
        if len(measured_indices) == 0:
            continue

        good_width = np.asarray(
            peak_table["fwhm"][measured_indices],
            dtype=float,
        )
        typical_fwhm = float(np.nanmedian(good_width))
        if not np.isfinite(typical_fwhm):
            continue

        for peak_i in measured_indices:
            reference_id = str(peak_table["reference_id"][peak_i])
            if reference_id == "":
                continue

            reference_matches = np.where(
                np.asarray(predicted["reference_id"], dtype=str)
                == reference_id
            )[0]
            if len(reference_matches) != 1:
                continue

            reference_i = int(reference_matches[0])
            nearest_distance = float(reference_nearest[reference_i])
            peak_table["atlas_neighbour_distance_pixel"][peak_i] = nearest_distance

            if nearest_distance < config.atlas_blend_fwhm_factor * typical_fwhm:
                peak_table["quality_flag"][peak_i] = int(
                    int(peak_table["quality_flag"][peak_i])
                    | int(CalibrationPeakFlag.ATLAS_BLEND)
                )

    _refresh_used_for_wavelength_fit(peak_table)
    return peak_table, calibration_shift_y

def measure_thorium_lines(
    counts,
    orders,
    thorium_atlas,
    *,
    reference_wavelength_function,
    detector_shift_y=0.0,
    y_bounds=(0.0, 4111.0),
    variance=None,
    source="FibTh",
    ccd=None,
    exposure_index=-1,
    mjd_mid=np.nan,
    fibre=-1,
    trace_x_function=None,
    config=None,
    minimum_reference_intensity=None,
    initial_matching_radius_pixel=3.0,
    final_matching_radius_pixel=1.5,
    diagnostics="none",
    diagnostic_dir=None,
    log_level=None,
):
    """Measure and identify one FibTh or SimTh extracted spectrum.

    The final table is source-independent: it contains physical order, measured
    ``y`` and uncertainty, laboratory vacuum wavelength and uncertainty,
    quality/usage flags, and the fitted detector-space FWHM.  Thorium peaks are
    currently measured with the pixel-integrated Gaussian seed profile; the
    FWHM is retained explicitly for later resolution-profile work.
    """
    if source not in {"FibTh", "SimTh"}:
        raise ValueError("source must be 'FibTh' or 'SimTh'")
    if config is None:
        config = CalibrationPeakConfig()

    lines = measure_calibration_peaks(
        counts,
        orders,
        variance=variance,
        calibration_type=source,
        ccd=ccd,
        exposure_index=exposure_index,
        mjd_mid=mjd_mid,
        fibre=fibre,
        trace_x_function=trace_x_function,
        config=config,
        diagnostics=diagnostics,
        diagnostic_dir=diagnostic_dir,
        log_level=log_level,
    )
    if len(lines) == 0:
        return CalibrationLineSet(lines, source, np.nan)

    lines, calibration_shift_y = identify_thorium_lines(
        lines,
        thorium_atlas,
        reference_wavelength_function=reference_wavelength_function,
        detector_shift_y=detector_shift_y,
        y_bounds=y_bounds,
        config=config,
        minimum_reference_intensity=minimum_reference_intensity,
        initial_matching_radius_pixel=initial_matching_radius_pixel,
        final_matching_radius_pixel=final_matching_radius_pixel,
    )
    debug = (isinstance(log_level, str) and log_level.upper() == "DEBUG") or (
        not isinstance(log_level, str) and log_level is not None and int(log_level) <= 10
    )
    summary = calibration_quality_summary(lines)
    if debug:
        fibre_text = "" if int(fibre) == -1 else f" fibre {int(fibre):+d}"
        print(
            f"{source} CCD{ccd} exposure {exposure_index}{fibre_text}: "
            f"{summary['identified']}/{summary['total']} matched; "
            f"{summary['accepted']} retained | "
            f"unmatched={summary.get('unmatched', 0)}, "
            f"atlas_blend={summary.get('atlas_blend', 0)}, "
            f"measured_blend={summary.get('blend_candidate', 0)}, "
            f"width={summary.get('width_outlier', 0)}, "
            f"lowS/N={summary.get('low_snr', 0)}, "
            f"sigma_y={summary.get('large_centroid_error', 0)}"
        )
        for order in np.unique(np.asarray(lines["order"], dtype=int)):
            subset = lines[np.asarray(lines["order"], dtype=int) == int(order)]
            order_summary = calibration_quality_summary(subset)
            print(
                f"    order {int(order)}: matched={order_summary['identified']}/{order_summary['total']}; "
                f"retained={order_summary['accepted']}; "
                f"unmatched={order_summary.get('unmatched', 0)}, "
                f"atlas_blend={order_summary.get('atlas_blend', 0)}, "
                f"blend={order_summary.get('blend_candidate', 0)}, "
                f"width={order_summary.get('width_outlier', 0)}"
            )

    if str(diagnostics).lower() != "none" and diagnostic_dir is not None:
        from . import diagnostics as diagnostic_plots
        fibre_suffix = "" if int(fibre) == -1 else f"_fibre{int(fibre):+03d}"
        filename = (
            Path(diagnostic_dir)
            / f"{source.lower()}_ccd{ccd}_exposure{int(exposure_index):03d}{fibre_suffix}_summary.png"
        )
        diagnostic_plots.plot_calibration_summary(
            lines, calibration_type=source, ccd=str(ccd),
            exposure_index=int(exposure_index), fibre=int(fibre), filename=filename,
            show=str(diagnostics).lower() == "full",
        )
        if debug:
            print(f"  calibration summary -> {filename}")

    if "lsf_model" not in lines.colnames:
        lines["lsf_model"] = np.full(len(lines), "integrated_gaussian", dtype="U24")
    if "fwhm_pixel" not in lines.colnames:
        lines["fwhm_pixel"] = np.asarray(lines["fwhm"], dtype=float)
    if "fwhm_uncertainty_pixel" not in lines.colnames:
        lines["fwhm_uncertainty_pixel"] = np.asarray(lines["fwhm_uncertainty"], dtype=float)
    return CalibrationLineSet(lines, source, calibration_shift_y)


# Backwards-compatible name while the notebooks are migrated.
identify_thorium_peaks = identify_thorium_lines
