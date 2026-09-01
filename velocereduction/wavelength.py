from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from astropy.io import fits
from astropy.table import Table

from numpy.polynomial.legendre import (
    legvander2d,
    legval2d,
    legder,
)

from . import utils


logger = logging.getLogger(__name__)


SPEED_OF_LIGHT_MPS = 299_792_458.0

VELOCE_CCD_ORDERS = {
    "1": np.arange(167, 138 - 1, -1),
    "2": np.arange(140, 103 - 1, -1),
    "3": np.arange(104, 65 - 1, -1),
}


# ============================================================
# Legacy/reference wavelength solutions
# ============================================================

def load_initial_wavelength_solutions(
    repository_path,
    calibration_types=("SimTh", "SimLC", "FibTh"),
):
    """
    Read the old per-order wavelength solutions.

    The legacy coefficients are assumed to describe wavelength in nm as

        lambda(y) = sum_k c_k (y - 2048)^k.

    Returns
    -------
    dict
        initial_wavelength_solution[calibration_type][order_name]
    """

    repository_path = Path(repository_path)

    solutions = {}

    for calibration_type in calibration_types:

        solutions[calibration_type] = {}

        for ccd, orders in VELOCE_CCD_ORDERS.items():

            for order in orders:

                order_name = f"ccd_{ccd}_order_{order}"

                lc_filename = (
                    repository_path
                    / "velocereduction"
                    / "wavelength_coefficients"
                    / f"wavelength_coefficients_ccd_{ccd}_order_{order}_lc.txt"
                )

                thxe_filename = (
                    repository_path
                    / "velocereduction"
                    / "wavelength_coefficients"
                    / f"wavelength_coefficients_ccd_{ccd}_order_{order}_thxe.txt"
                )

                # Prefer the relevant old calibration where possible.
                if calibration_type == "SimLC":

                    preferred = [
                        lc_filename,
                        thxe_filename,
                    ]

                else:

                    preferred = [
                        thxe_filename,
                        lc_filename,
                    ]

                coefficients = None

                for filename in preferred:

                    if filename.exists():

                        coefficients = np.loadtxt(
                            filename
                        )

                        break

                if coefficients is not None:

                    solutions[
                        calibration_type
                    ][order_name] = coefficients

    return solutions


# ============================================================
# SimLC identification
# ============================================================

def identify_simlc_peaks(
    peak_table,
    initial_wavelength_solution,
    detector_shifts,
    *,
    reference_pixel=2048.0,
):
    """
    Assign the nearest laser-comb mode to each measured SimLC peak.

    Notes
    -----
    ``closest_lc_wavelength`` is stored in Angstrom because this is the
    convention used by the existing laser-comb utility functions.

    The old wavelength coefficients are assumed to be in nm.
    """

    if "closest_lc_number" not in peak_table.colnames:
        peak_table["closest_lc_number"] = np.full(
            len(peak_table),
            -1,
            dtype=np.int64,
        )

    if "closest_lc_wavelength" not in peak_table.colnames:
        peak_table["closest_lc_wavelength"] = np.full(
            len(peak_table),
            np.nan,
            dtype=float,
        )

    if "expected_wavelength" not in peak_table.colnames:
        peak_table["expected_wavelength"] = np.full(
            len(peak_table),
            np.nan,
            dtype=float,
        )

    for order_name, coefficients in (
        initial_wavelength_solution.items()
    ):

        ccd = int(
            order_name.split("_")[1]
        )

        order_number = int(
            order_name.split("_")[-1]
        )

        detector_selection = (
            detector_shifts["ccd"] == ccd
        )

        if np.count_nonzero(
            detector_selection
        ) == 0:

            logger.warning(
                "No detector shift found for CCD %d",
                ccd,
            )

            continue

        detector_shift_dy = float(
            detector_shifts["dy"][
                detector_selection
            ][0]
        )

        relevant_peaks = (
            (peak_table["ccd"] == ccd)
            & (
                peak_table["order"]
                == order_number
            )
        )

        if np.count_nonzero(
            relevant_peaks
        ) == 0:

            continue

        y_measured = np.asarray(
            peak_table["y"][
                relevant_peaks
            ],
            dtype=float,
        )

        # Preserve the sign convention currently used by the
        # pipeline. Once detector_shift_dy is formally defined,
        # this is the only place that needs changing if required.
        y_reference = (
            y_measured
            - reference_pixel
            + detector_shift_dy
        )

        # Legacy polynomial returns nm.
        expected_wavelength_nm = (
            np.polynomial.polynomial.polyval(
                y_reference,
                coefficients,
            )
        )

        # Existing LC utility convention: Angstrom.
        expected_wavelength_angstrom = (
            10.0
            * expected_wavelength_nm
        )

        closest_lc_number = np.asarray(
            np.round(
                utils.lasercomb_numbers_from_wavelength(
                    expected_wavelength_angstrom
                )
            ),
            dtype=np.int64,
        )

        closest_lc_wavelength = np.asarray(
            utils.lasercomb_wavelength_from_numbers(
                closest_lc_number
            ),
            dtype=float,
        )

        peak_table[
            "expected_wavelength"
        ][relevant_peaks] = (
            expected_wavelength_angstrom
        )

        peak_table[
            "closest_lc_number"
        ][relevant_peaks] = (
            closest_lc_number
        )

        peak_table[
            "closest_lc_wavelength"
        ][relevant_peaks] = (
            closest_lc_wavelength
        )

    return peak_table


# ============================================================
# Surface helpers
# ============================================================

def _normalisation_for_ccd(ccd):
    """
    Return fixed physical normalization for one detector.
    """

    ccd = str(ccd)

    y_min = 0.0
    y_max = 4111.0

    orders = VELOCE_CCD_ORDERS[ccd]

    m_min = float(
        np.min(orders)
    )

    m_max = float(
        np.max(orders)
    )

    y_mid = 0.5 * (
        y_min + y_max
    )

    y_scale = 0.5 * (
        y_max - y_min
    )

    m_mid = 0.5 * (
        m_min + m_max
    )

    m_scale = 0.5 * (
        m_max - m_min
    )

    return {
        "y_min": y_min,
        "y_max": y_max,
        "m_min": m_min,
        "m_max": m_max,
        "y_mid": y_mid,
        "y_scale": y_scale,
        "m_mid": m_mid,
        "m_scale": m_scale,
    }


def evaluate_wavelength_surface(
    y,
    order,
    coefficients,
    normalisation,
):
    """
    Evaluate lambda(y,m) in nm from the fitted m*lambda surface.
    """

    y = np.asarray(
        y,
        dtype=float,
    )

    order = np.asarray(
        order,
        dtype=float,
    )

    y_norm = (
        (y - normalisation["y_mid"])
        / normalisation["y_scale"]
    )

    m_norm = (
        (order - normalisation["m_mid"])
        / normalisation["m_scale"]
    )

    m_lambda = legval2d(
        y_norm,
        m_norm,
        coefficients,
    )

    return m_lambda / order


def _surface_derivative_y(
    y,
    order,
    coefficients,
    normalisation,
):
    """
    Return d(m*lambda)/dy in nm per pixel.
    """

    y_norm = (
        (y - normalisation["y_mid"])
        / normalisation["y_scale"]
    )

    m_norm = (
        (order - normalisation["m_mid"])
        / normalisation["m_scale"]
    )

    coefficients_dy = legder(
        coefficients,
        axis=0,
    )

    return (
        legval2d(
            y_norm,
            m_norm,
            coefficients_dy,
        )
        / normalisation["y_scale"]
    )


# ============================================================
# Per-order representation
# ============================================================

def calculate_per_order_coefficients(
    coefficients,
    ccd,
    normalisation,
    *,
    reference_pixel=2048.0,
    degree_y=None,
):
    """
    Convert the 2D surface into legacy-style 1D wavelength polynomials.

    Returned coefficients are in nm and satisfy approximately

        lambda(y) =
            c0
            + c1*(y-reference_pixel)
            + ...

    for each echelle order.
    """

    if degree_y is None:
        degree_y = (
            coefficients.shape[0] - 1
        )

    y_grid = np.arange(
        4112,
        dtype=float,
    )

    order_rows = []

    for order in VELOCE_CCD_ORDERS[
        str(ccd)
    ]:

        order_array = np.full(
            len(y_grid),
            float(order),
        )

        wavelength_nm = (
            evaluate_wavelength_surface(
                y_grid,
                order_array,
                coefficients,
                normalisation,
            )
        )

        coefficients_order = (
            np.polynomial.polynomial.polyfit(
                y_grid - reference_pixel,
                wavelength_nm,
                degree_y,
            )
        )

        row = {
            "order": int(order),
        }

        for index, value in enumerate(
            coefficients_order
        ):

            row[
                f"coefficient_{index}"
            ] = float(value)

        order_rows.append(
            row
        )

    return Table(
        rows=order_rows
    )


# ============================================================
# Main wavelength-surface fit
# ============================================================

def fit_wavelength_surface(
    peak_table,
    *,
    calibration_type="SimLC",
    ccd=3,
    degree_y=7,
    degree_m=5,
    minimum_y=200,
    maximum_y=3900,
    minimum_signal_to_noise=20,
    maximum_y_uncertainty=0.005,
    maximum_iterations=10,
    sigma_clip=5.0,
    minimum_clip_pixel=0.05,
):
    """
    Fit m*lambda(y,m) with a 2D Legendre surface.

    Returns
    -------
    result : dict
        Surface coefficients, normalization, per-line residuals,
        fit mask, statistics and per-order coefficients.
    """

    if calibration_type == 'SimLC':
        closet_wavelength_column = 'closest_lc_wavelength'
    elif calibration_type == 'FibTh':
        closet_wavelength_column = 'closest_th_wavelength'
    elif calibration_type == 'SimTh':
        closet_wavelength_column = 'closest_th_wavelength'
    else:
        raise ValueError(
            f"Unknown calibration_type {calibration_type}"
        )

    input_peaks = (
        (peak_table["ccd"] == ccd)
        & peak_table[
            "used_for_wavelength_fit"
        ]
        & (
            peak_table["y"]
            > minimum_y
        )
        & (
            peak_table["y"]
            < maximum_y
        )
        & (
            peak_table[
                "signal_to_noise"
            ]
            > minimum_signal_to_noise
        )
        & (
            peak_table[
                "y_uncertainty"
            ]
            < maximum_y_uncertainty
        )
        & np.isfinite(
            peak_table[
                closet_wavelength_column
            ]
        )
    )

    peak_data = peak_table[
        input_peaks
    ].copy()

    m = np.asarray(
        peak_data["order"],
        dtype=float,
    )

    y = np.asarray(
        peak_data["y"],
        dtype=float,
    )

    y_sigma = np.asarray(
        peak_data[
            "y_uncertainty"
        ],
        dtype=float,
    )

    # Existing comb utility output is Angstrom.
    wavelength_nm = (
        np.asarray(
            peak_data[
                closet_wavelength_column
            ],
            dtype=float,
        )
        / 10.0
    )

    m_lambda = (
        m * wavelength_nm
    )

    normalisation = (
        _normalisation_for_ccd(
            ccd
        )
    )

    y_norm = (
        (y - normalisation["y_mid"])
        / normalisation["y_scale"]
    )

    m_norm = (
        (m - normalisation["m_mid"])
        / normalisation["m_scale"]
    )

    design_matrix = (
        legvander2d(
            y_norm,
            m_norm,
            [
                degree_y,
                degree_m,
            ],
        )
        .reshape(
            len(y),
            -1,
        )
    )

    finite = (
        np.isfinite(y)
        & np.isfinite(y_sigma)
        & (y_sigma > 0)
        & np.isfinite(wavelength_nm)
    )

    used = finite.copy()

    # --------------------------------------------------------
    # Initial unweighted fit
    # --------------------------------------------------------

    coefficients_flat, _, _, _ = (
        np.linalg.lstsq(
            design_matrix[used],
            m_lambda[used],
            rcond=None,
        )
    )

    coefficients = (
        coefficients_flat.reshape(
            degree_y + 1,
            degree_m + 1,
        )
    )

    # --------------------------------------------------------
    # Iteratively weighted + robust fit
    # --------------------------------------------------------

    for iteration in range(
        maximum_iterations
    ):

        old_used = used.copy()

        dm_lambda_dy = (
            _surface_derivative_y(
                y,
                m,
                coefficients,
                normalisation,
            )
        )

        m_lambda_sigma = (
            np.abs(dm_lambda_dy)
            * y_sigma
        )

        valid_sigma = (
            np.isfinite(
                m_lambda_sigma
            )
            & (m_lambda_sigma > 0)
        )

        used &= valid_sigma

        weighted_design_matrix = (
            design_matrix[used]
            / m_lambda_sigma[
                used,
                None,
            ]
        )

        weighted_m_lambda = (
            m_lambda[used]
            / m_lambda_sigma[used]
        )

        coefficients_flat, _, _, _ = (
            np.linalg.lstsq(
                weighted_design_matrix,
                weighted_m_lambda,
                rcond=None,
            )
        )

        coefficients = (
            coefficients_flat.reshape(
                degree_y + 1,
                degree_m + 1,
            )
        )

        m_lambda_model = (
            legval2d(
                y_norm,
                m_norm,
                coefficients,
            )
        )

        dm_lambda_dy = (
            _surface_derivative_y(
                y,
                m,
                coefficients,
                normalisation,
            )
        )

        pixel_residual = (
            m_lambda
            - m_lambda_model
        ) / dm_lambda_dy

        residual_median = (
            np.nanmedian(
                pixel_residual[used]
            )
        )

        residual_mad = (
            np.nanmedian(
                np.abs(
                    pixel_residual[used]
                    - residual_median
                )
            )
        )

        residual_sigma = (
            1.4826
            * residual_mad
        )

        clipping_limit = max(
            sigma_clip
            * residual_sigma,
            minimum_clip_pixel,
        )

        used = (
            finite
            & valid_sigma
            & (
                np.abs(
                    pixel_residual
                    - residual_median
                )
                < clipping_limit
            )
        )

        logger.debug(
            "%s CCD%d wavelength fit iteration %d: "
            "%d/%d lines, robust scatter %.4f pix",
            calibration_type,
            ccd,
            iteration + 1,
            np.count_nonzero(used),
            len(used),
            residual_sigma,
        )

        if np.array_equal(
            used,
            old_used,
        ):

            break

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    m_lambda_model = (
        legval2d(
            y_norm,
            m_norm,
            coefficients,
        )
    )

    wavelength_model_nm = (
        m_lambda_model / m
    )

    wavelength_residual_nm = (
        wavelength_nm
        - wavelength_model_nm
    )

    wavelength_residual_angstrom = (
        10.0
        * wavelength_residual_nm
    )

    dm_lambda_dy = (
        _surface_derivative_y(
            y,
            m,
            coefficients,
            normalisation,
        )
    )

    pixel_residual = (
        m_lambda
        - m_lambda_model
    ) / dm_lambda_dy

    velocity_residual_mps = (
        SPEED_OF_LIGHT_MPS
        * wavelength_residual_nm
        / wavelength_nm
    )

    peak_data[
        "wavelength_model_nm"
    ] = wavelength_model_nm

    peak_data[
        "wavelength_residual_angstrom"
    ] = wavelength_residual_angstrom

    peak_data[
        "pixel_residual"
    ] = pixel_residual

    peak_data[
        "velocity_residual_mps"
    ] = velocity_residual_mps

    peak_data[
        "used_for_surface_fit"
    ] = used

    rms_pixel = np.sqrt(
        np.mean(
            pixel_residual[used] ** 2
        )
    )

    rms_angstrom = np.sqrt(
        np.mean(
            wavelength_residual_angstrom[
                used
            ] ** 2
        )
    )

    rms_velocity = np.sqrt(
        np.mean(
            velocity_residual_mps[
                used
            ] ** 2
        )
    )

    median_abs_pixel = (
        np.median(
            np.abs(
                pixel_residual[used]
            )
        )
    )

    per_order_coefficients = (
        calculate_per_order_coefficients(
            coefficients,
            ccd,
            normalisation,
            reference_pixel=2048.0,
            degree_y=degree_y,
        )
    )

    result = {
        "calibration_type":
            calibration_type,

        "ccd":
            int(ccd),

        "degree_y":
            int(degree_y),

        "degree_m":
            int(degree_m),

        "coefficients":
            coefficients,

        "normalisation":
            normalisation,

        "peak_data":
            peak_data,

        "per_order_coefficients":
            per_order_coefficients,

        "rms_pixel":
            float(rms_pixel),

        "median_abs_pixel":
            float(median_abs_pixel),

        "rms_angstrom":
            float(rms_angstrom),

        "rms_velocity_mps":
            float(rms_velocity),

        "number_lines":
            int(
                np.count_nonzero(
                    used
                )
            ),
    }

    return result

def save_wavelength_solution(
    result,
    filename,
    *,
    overwrite=False,
):
    """
    Save a complete wavelength solution to one FITS file.
    """

    filename = Path(
        filename
    )

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalisation = result[
        "normalisation"
    ]

    primary = fits.PrimaryHDU()

    header = primary.header

    header["ORIGIN"] = (
        "velocereduction"
    )

    header["CALTYPE"] = result[
        "calibration_type"
    ]

    header["CCD"] = result[
        "ccd"
    ]

    header["DEGY"] = result[
        "degree_y"
    ]

    header["DEGM"] = result[
        "degree_m"
    ]

    header["WUNIT"] = "nm"

    header["YMIN"] = (
        normalisation["y_min"]
    )

    header["YMAX"] = (
        normalisation["y_max"]
    )

    header["YMID"] = (
        normalisation["y_mid"]
    )

    header["YSCALE"] = (
        normalisation["y_scale"]
    )

    header["MMIN"] = (
        normalisation["m_min"]
    )

    header["MMAX"] = (
        normalisation["m_max"]
    )

    header["MMID"] = (
        normalisation["m_mid"]
    )

    header["MSCALE"] = (
        normalisation["m_scale"]
    )

    header["REFPIX"] = 2048.0

    header["NLINES"] = result[
        "number_lines"
    ]

    header["RMSPIX"] = result[
        "rms_pixel"
    ]

    header["RMSANG"] = result[
        "rms_angstrom"
    ]

    header["RMSMPS"] = result[
        "rms_velocity_mps"
    ]

    # 2D Legendre coefficients.
    surface_hdu = fits.ImageHDU(
        np.asarray(
            result["coefficients"],
            dtype=float,
        ),
        name="SURFACE",
    )

    surface_hdu.header[
        "CONTENT"
    ] = "2D Legendre coefficients for m*lambda"

    # Legacy-style 1D coefficients for each order.
    order_hdu = fits.table_to_hdu(
        result[
            "per_order_coefficients"
        ]
    )

    order_hdu.name = (
        "ORDER_COEFFICIENTS"
    )

    order_hdu.header[
        "WUNIT"
    ] = "nm"

    order_hdu.header[
        "REFPIX"
    ] = 2048.0

    # Individual fitted calibration lines.
    lines_hdu = fits.table_to_hdu(
        result["peak_data"]
    )

    lines_hdu.name = "LINES"

    fits.HDUList(
        [
            primary,
            surface_hdu,
            order_hdu,
            lines_hdu,
        ]
    ).writeto(
        filename,
        overwrite=overwrite,
    )

    logger.info(
        "Saved %s CCD%d wavelength solution to %s",
        result["calibration_type"],
        result["ccd"],
        filename,
    )

def save_wavelength_diagnostics(
    result,
    diagnostic_dir,
    *,
    diagnostics="basic",
):
    """
    Save a presentation/paper-quality wavelength-solution QA figure.

    Layout
    ------
    Top left:
        Wavelength-solution description and RMS statistics.

    Top right:
        Horizontal colour bar for the main residual map.

    Middle left:
        Median pixel residual per echelle order, with 16th--84th
        percentile ranges.

    Middle right:
        Dispersion pixel versus echelle order, coloured by pixel residual.

    Bottom left:
        Velocity residual versus pixel phase, with binned median and
        16th--84th percentile range.

    Bottom right:
        Velocity residual versus dispersion pixel, with binned median and
        16th--84th percentile range.
    """

    if diagnostics == "none":
        return

    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.colors import TwoSlopeNorm

    # ------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------

    diagnostic_dir = Path(
        diagnostic_dir
    )

    diagnostic_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Input data
    # ------------------------------------------------------------

    peak_data = result[
        "peak_data"
    ]

    used = np.asarray(
        peak_data[
            "used_for_surface_fit"
        ],
        dtype=bool,
    )

    y = np.asarray(
        peak_data["y"],
        dtype=float,
    )[used]

    m = np.asarray(
        peak_data["order"],
        dtype=float,
    )[used]

    pixel_phase = np.asarray(
        peak_data["pixel_phase"],
        dtype=float,
    )[used]

    pixel_residual = np.asarray(
        peak_data["pixel_residual"],
        dtype=float,
    )[used]

    wavelength_residual_angstrom = np.asarray(
        peak_data[
            "wavelength_residual_angstrom"
        ],
        dtype=float,
    )[used]

    velocity_residual_mps = np.asarray(
        peak_data[
            "velocity_residual_mps"
        ],
        dtype=float,
    )[used]

    calibration_type = result[
        "calibration_type"
    ]

    ccd = result[
        "ccd"
    ]

    degree_y = result[
        "degree_y"
    ]

    degree_m = result[
        "degree_m"
    ]

    # ------------------------------------------------------------
    # RMS statistics
    # ------------------------------------------------------------

    rms_pixel = np.sqrt(
        np.mean(
            pixel_residual**2
        )
    )

    rms_angstrom = np.sqrt(
        np.mean(
            wavelength_residual_angstrom**2
        )
    )

    rms_velocity = np.sqrt(
        np.mean(
            velocity_residual_mps**2
        )
    )

    # ------------------------------------------------------------
    # Symmetric residual colour scale
    # ------------------------------------------------------------

    residual_colour_limit = np.nanpercentile(
        np.abs(
            pixel_residual
        ),
        99,
    )

    colour_norm = TwoSlopeNorm(
        vmin=-residual_colour_limit,
        vcenter=0.0,
        vmax=residual_colour_limit,
    )

    # ------------------------------------------------------------
    # Common plotting limits
    # ------------------------------------------------------------

    unique_orders = np.unique(
        m
    )

    order_limits = (
        np.nanmin(
            unique_orders
        ) - 0.5,
        np.nanmax(
            unique_orders
        ) + 0.5,
    )

    dispersion_limits = (
        0,
        4111,
    )

    velocity_limit = (
        1.1
        * np.nanpercentile(
            np.abs(
                velocity_residual_mps
            ),
            99,
        )
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    with plt.rc_context(
        {
            "font.size": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    ):

        fig = plt.figure(
            figsize=(16, 9),
        )

        # Five rows:
        #
        # 0     compact information / colourbar
        # 1-2   main panels
        # 3     explicit spacer for x-axis labels
        # 4     bottom panels
        #
        gs = fig.add_gridspec(
            5,
            3,

            height_ratios=[
                0.55,
                1.0,
                1.0,
                0.18,
                0.78,
            ],

            width_ratios=[
                1.0,
                1.0,
                1.0,
            ],

            left=0.07,
            right=0.965,
            bottom=0.085,
            top=0.965,

            wspace=0.28,
            hspace=0.16,
        )

        # --------------------------------------------------------
        # Top row
        # --------------------------------------------------------

        ax_info = fig.add_subplot(
            gs[0, 0]
        )

        ax_colourbar_panel = fig.add_subplot(
            gs[0, 1:3]
        )

        # --------------------------------------------------------
        # Main panels
        # --------------------------------------------------------

        ax_order = fig.add_subplot(
            gs[1:3, 0]
        )

        ax_map = fig.add_subplot(
            gs[1:3, 1:3],
            sharey=ax_order,
        )

        # --------------------------------------------------------
        # Bottom panels
        # --------------------------------------------------------

        ax_phase = fig.add_subplot(
            gs[4, 0]
        )

        ax_y = fig.add_subplot(
            gs[4, 1:3],
            sharex=ax_map,
        )

        # ========================================================
        # TOP LEFT
        # Title + RMS statistics
        # ========================================================

        ax_info.axis(
            "off"
        )

        ax_info.text(
            0.5,
            0.8,
            (
                f"Wavelength solution "
                f"{calibration_type} CCD{ccd}"
            ),
            transform=ax_info.transAxes,
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
        )

        ax_info.text(
            0.5,
            0.4,
            (
                f"2-dim. Legendre fit "
                f"(deg_y={degree_y}, deg_m={degree_m}) "
                f"to {len(y):,} lines"
            ),
            transform=ax_info.transAxes,
            ha="center",
            va="top",
            fontsize=14,
        )

        ax_info.text(
            0.5,
            0.15,
            (
                rf"$\mathbf{{RMS}} = {rms_pixel:.4f}$ px; {rms_angstrom:.5f} $\text{{\AA}}$; {rms_velocity:.1f} m s$^{-1}$"
            ),
            transform=ax_info.transAxes,
            ha="center",
            va="top",
            fontsize=14,
        )

        # ========================================================
        # MIDDLE LEFT
        # Residual statistics by order
        # ========================================================

        order_median = []
        order_lower = []
        order_upper = []

        for order in unique_orders:

            selection = (
                m == order
            )

            residuals = (
                pixel_residual[
                    selection
                ]
            )

            p16, p50, p84 = np.nanpercentile(
                residuals,
                [
                    16,
                    50,
                    84,
                ],
            )

            order_median.append(
                p50
            )

            order_lower.append(
                p50 - p16
            )

            order_upper.append(
                p84 - p50
            )

        order_median = np.asarray(
            order_median
        )

        order_lower = np.asarray(
            order_lower
        )

        order_upper = np.asarray(
            order_upper
        )

        ax_order.errorbar(
            order_median,
            unique_orders,

            xerr=np.vstack(
                [
                    order_lower,
                    order_upper,
                ]
            ),

            fmt="o",
            markersize=5,
            capsize=2,
            linewidth=1.2,
        )

        ax_order.axvline(
            0,
            ls="--",
            lw=1,
        )

        ax_order.set_xlim(
            -residual_colour_limit,
            residual_colour_limit,
        )

        ax_order.set_ylim(
            order_limits
        )

        ax_order.set_xlabel(
            r"Residual$~/~\mathrm{px}$",
            labelpad=7,
        )

        ax_order.set_ylabel(
            r"Echelle order $m$"
        )

        # ========================================================
        # MIDDLE RIGHT
        # Detector/order residual map
        # ========================================================

        scatter = ax_map.scatter(
            y,
            m,

            c=pixel_residual,

            s=16,
            alpha=0.90,

            cmap="RdBu_r",
            norm=colour_norm,

            linewidths=0,
            rasterized=True,
        )

        ax_map.set_xlim(
            dispersion_limits
        )

        ax_map.set_ylim(
            order_limits
        )

        ax_map.set_xlabel(
            r"Dispersion pixel $y$",
            labelpad=7,
        )

        ax_map.set_ylabel(
            r"Echelle order $m$"
        )

        # ========================================================
        # TOP RIGHT
        # Dedicated colourbar
        # ========================================================

        ax_colourbar_panel.axis(
            "off"
        )

        colourbar_axis = (
            ax_colourbar_panel.inset_axes(
                [
                    0.0,
                    0.40,
                    1.0,
                    0.24,
                ]
            )
        )

        colourbar = fig.colorbar(
            scatter,
            cax=colourbar_axis,
            orientation="horizontal",
        )

        colourbar.set_label(
            r"Wavelength-model residual$~/~\mathrm{px}$",
            fontsize=14,
            labelpad=4,
        )

        colourbar.ax.tick_params(
            labelsize=12,
        )

        colourbar.ax.xaxis.set_ticks_position(
            "bottom"
        )

        colourbar.ax.xaxis.set_label_position(
            "bottom"
        )

        # ========================================================
        # BOTTOM LEFT
        # Velocity residual versus pixel phase
        # ========================================================

        ax_phase.scatter(
            pixel_phase,
            velocity_residual_mps,

            s=12,
            alpha=0.40,

            linewidths=0,
            rasterized=True,
        )

        number_phase_bins = 15

        phase_edges = np.linspace(
            -0.5,
            0.5,
            number_phase_bins + 1,
        )

        phase_centres = (
            0.5
            * (
                phase_edges[:-1]
                + phase_edges[1:]
            )
        )

        phase_median = np.full(
            number_phase_bins,
            np.nan,
        )

        phase_p16 = np.full(
            number_phase_bins,
            np.nan,
        )

        phase_p84 = np.full(
            number_phase_bins,
            np.nan,
        )

        for i in range(
            number_phase_bins
        ):

            selection = (
                (
                    pixel_phase
                    >= phase_edges[i]
                )
                & (
                    pixel_phase
                    < phase_edges[i + 1]
                )
            )

            if np.count_nonzero(
                selection
            ) >= 5:

                (
                    phase_p16[i],
                    phase_median[i],
                    phase_p84[i],
                ) = np.nanpercentile(
                    velocity_residual_mps[
                        selection
                    ],
                    [
                        16,
                        50,
                        84,
                    ],
                )

        ax_phase.fill_between(
            phase_centres,
            phase_p16,
            phase_p84,
            alpha=0.25,
            linewidth=0,
        )

        ax_phase.plot(
            phase_centres,
            phase_median,
            lw=2.5,
        )

        ax_phase.axhline(
            0,
            ls="--",
            lw=1,
        )

        ax_phase.set_xlim(
            -0.5,
            0.5,
        )

        ax_phase.set_ylim(
            -velocity_limit,
            velocity_limit,
        )

        ax_phase.set_xlabel(
            r"Pixel phase"
        )

        ax_phase.set_ylabel(
            r"Residual$~/~\mathrm{m\,s^{-1}}$"
        )

        # ========================================================
        # BOTTOM RIGHT
        # Velocity residual versus dispersion pixel
        # ========================================================

        ax_y.scatter(
            y,
            velocity_residual_mps,

            s=12,
            alpha=0.40,

            linewidths=0,
            rasterized=True,
        )

        number_y_bins = 20

        y_edges = np.linspace(
            dispersion_limits[0],
            dispersion_limits[1],
            number_y_bins + 1,
        )

        y_centres = (
            0.5
            * (
                y_edges[:-1]
                + y_edges[1:]
            )
        )

        y_median = np.full(
            number_y_bins,
            np.nan,
        )

        y_p16 = np.full(
            number_y_bins,
            np.nan,
        )

        y_p84 = np.full(
            number_y_bins,
            np.nan,
        )

        for i in range(
            number_y_bins
        ):

            selection = (
                (
                    y
                    >= y_edges[i]
                )
                & (
                    y
                    < y_edges[i + 1]
                )
            )

            if np.count_nonzero(
                selection
            ) >= 5:

                (
                    y_p16[i],
                    y_median[i],
                    y_p84[i],
                ) = np.nanpercentile(
                    velocity_residual_mps[
                        selection
                    ],
                    [
                        16,
                        50,
                        84,
                    ],
                )

        ax_y.fill_between(
            y_centres,
            y_p16,
            y_p84,
            alpha=0.25,
            linewidth=0,
        )

        ax_y.plot(
            y_centres,
            y_median,
            lw=2.5,
        )

        ax_y.axhline(
            0,
            ls="--",
            lw=1,
        )

        ax_y.set_xlim(
            dispersion_limits
        )

        ax_y.set_ylim(
            -velocity_limit,
            velocity_limit,
        )

        ax_y.set_xlabel(
            r"Dispersion pixel $y$"
        )

        ax_y.set_ylabel(
            r"Residual$~/~\mathrm{m\,s^{-1}}$"
        )

        # --------------------------------------------------------
        # Save
        # --------------------------------------------------------

        filename = (
            diagnostic_dir
            / (
                f"{calibration_type.lower()}_"
                f"ccd{ccd}_"
                f"wavelength_solution.png"
            )
        )

        fig.savefig(
            filename,
            dpi=200,
            bbox_inches="tight",
        )

        if diagnostics == "full":
            plt.show()

        plt.close(
            fig
        )


def fit_and_save_wavelength_surface(
    peak_table,
    *,
    calibration_type,
    ccd,
    output_dir,
    diagnostic_dir,
    diagnostics="basic",
    overwrite=False,
    **fit_kwargs,
):
    """
    Fit, save, diagnose and return one CCD wavelength solution.
    """

    result = fit_wavelength_surface(
        peak_table,
        calibration_type=calibration_type,
        ccd=ccd,
        **fit_kwargs,
    )

    filename = (
        Path(output_dir)
        / (
            f"{calibration_type.lower()}_"
            f"ccd{ccd}_"
            f"wavelength_solution.fits"
        )
    )

    save_wavelength_solution(
        result,
        filename,
        overwrite=overwrite,
    )

    save_wavelength_diagnostics(
        result,
        diagnostic_dir,
        diagnostics=diagnostics,
    )

    logger.info(
        "%s CCD%d: %d lines, "
        "RMS %.4f pix / %.5f A / %.1f m/s",
        calibration_type,
        ccd,
        result["number_lines"],
        result["rms_pixel"],
        result["rms_angstrom"],
        result["rms_velocity_mps"],
    )

    return result

def load_th_reference_lines(paths, th_reference = 'murphy'):

    if th_reference == 'murphy':
        # ThAr atlas from Murphy et al. (2007) with update from 090311
        dtype = [
            ("wavenumber", float),
            ("wave_air", float),
            ("log10_intensity", float),
            ("element", "U10"),
            ("ion", "U10"),
            ("source", "U2"),
        ]
        thar_lines = Table(np.genfromtxt(paths.repository / 'velocereduction/veloce_reference_data/thar_UVES_MM090311.dat',
            dtype=dtype,
            comments="#",
            autostrip=True
        ))
        th_lines = thar_lines[thar_lines['element'] == 'Th']
        th_lines['wave_vac'] = utils.wavelength_air_to_vac(th_lines['wave_air'])
    else:
        raise ValueError(f"Currently, only th_reference='murphy' is supported. Got {th_reference} instead.")

    return(th_lines)