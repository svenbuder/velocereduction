import logging

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table

from numpy.polynomial import Polynomial

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from skimage.registration import phase_cross_correlation

from velocereduction import utils

logger = logging.getLogger(__name__)


_REGISTRATION_SOURCES = {
    '1': ['SimTh'],
    '2': ['SimTh'],
    '3': ['SimTh'],
}

_REFERENCE_REGISTRATION_RUNS = {
    '1': {'SimTh': '0001'},
    '2': {'SimTh': '0002'},
    '3': {'SimTh': '0003'},
}

def phase_correlation_shift(
    reference_image,
    moving_image,
    upsample_factor=100,
):
    """
    Measure the shift from the reference detector position to the
    current detector position.
    """

    shift, error, _ = phase_cross_correlation(
        reference_image,
        moving_image,
        upsample_factor=upsample_factor,
        normalization='phase',
    )

    # report negative shift for matching the reference image to the current image
    dx, dy = -shift

    return float(dx), float(dy), float(error)


def _expected_detector_shifts(night):
    """
    Return the historically expected detector position for this night.

    Values are relative to reference night 001122.
    """

    if night == '001122':
        return {
            'ccd_1': (0.00, 0.00),
            'ccd_2': (0.00, 0.00),
            'ccd_3': (0.00, 0.00),
        }

    date = int(night)

    if date < 231120:
        return None

    elif date <= 240518:
        return {
            'ccd_1': (0.00, 0.00),
            'ccd_2': (0.00, 0.00),
            'ccd_3': (0.01, -0.01),
        }

    elif date <= 241106:
        return {
            'ccd_1': (-0.89, 7.02),
            'ccd_2': (-3.86, 3.10),
            'ccd_3': (3.08, 1.80),
        }

    elif date <= 250507:
        return {
            'ccd_1': (-0.74, 8.13),
            'ccd_2': (-3.58, 4.06),
            'ccd_3': (3.09, 2.91),
        }

    elif date <= 250823:
        return {
            'ccd_1': (-6.15, 1.11),
            'ccd_2': (-8.75, 2.80),
            'ccd_3': (2.51, 0.22),
        }

    elif date <= 260303:
        return {
            'ccd_1': (-6.08, 1.41),
            'ccd_2': (-8.76, 2.88),
            'ccd_3': (2.72, 0.34),
        }

    return None

def _select_registration_exposures(
    reduction_input,
    source,
    ccd,
):
    """Select suitable detector-registration exposures for one CCD."""

    use_ccd = f'use_ccd{ccd}'

    base_selection = (
        (reduction_input['type'] == source)
        & reduction_input['use']
        & reduction_input[use_ccd]
    )

    selection = base_selection.copy()

    if source == 'SimTh':

        mixed = (
            base_selection
            & reduction_input['lc_requested']
        )

        n_mixed = np.sum(mixed)

        if n_mixed > 0:
            logger.debug(
                'CCD%s: excluding %d SimTh exposures with simultaneous LC',
                ccd,
                n_mixed,
            )

        selection &= ~reduction_input['lc_requested']

    return reduction_input[selection]

def _load_registration_reference(
    source,
    ccd,
    paths,
):
    """Load and preprocess the appropriate 001122 reference exposure."""

    reference_run = _REFERENCE_REGISTRATION_RUNS[
        ccd
    ][source]

    filename = utils.raw_fits_path(
        paths,
        night='001122',
        run=reference_run,
        ccd=ccd,
    )

    if not filename.exists():
        raise FileNotFoundError(
            f'Reference registration image not found: {filename}'
        )

    reference_image, _, _ = utils.preprocess_image(
        filename,
        ccd=ccd,
    )

    return reference_image

def _combine_shift_measurements(
    measurements,
    expected,
    ccd,
    max_scatter=0.10,
):
    """Combine detector-shift measurements for one CCD."""

    if len(measurements) == 0:

        if expected is not None:
            dx, dy = expected[f'ccd_{ccd}']

            logger.warning(
                'CCD%s: no detector-shift measurements; '
                'using historical position (%+.2f, %+.2f)',
                ccd,
                dx,
                dy,
            )

            return {
                'ccd': int(ccd),
                'dx': dx,
                'dy': dy,
                'dx_scatter': np.nan,
                'dy_scatter': np.nan,
                'n_used': 0,
                'sources': '',
                'status': 'historical fallback',
            }

        logger.warning(
            'CCD%s: no detector-shift measurement or historical reference',
            ccd,
        )

        return {
            'ccd': int(ccd),
            'dx': 0.0,
            'dy': 0.0,
            'dx_scatter': np.nan,
            'dy_scatter': np.nan,
            'n_used': 0,
            'sources': '',
            'status': 'zero fallback',
        }

    dx = np.array(
        [m['dx'] for m in measurements]
    )

    dy = np.array(
        [m['dy'] for m in measurements]
    )

    adopted_dx = float(np.nanmedian(dx))
    adopted_dy = float(np.nanmedian(dy))

    dx_scatter = (
        float(np.nanstd(dx))
        if len(dx) > 1
        else np.nan
    )

    dy_scatter = (
        float(np.nanstd(dy))
        if len(dy) > 1
        else np.nan
    )

    status = 'good'

    if (
        len(dx) > 1
        and (
            dx_scatter > max_scatter
            or dy_scatter > max_scatter
        )
    ):
        logger.warning(
            'CCD%s: detector shift measurements show large scatter: '
            'dx=%.2f px, dy=%.2f px',
            ccd,
            dx_scatter,
            dy_scatter,
        )

        status = 'large scatter'

    sources = ','.join(
        sorted(set(m['source'] for m in measurements))
    )

    return {
        'ccd': int(ccd),
        'dx': adopted_dx,
        'dy': adopted_dy,
        'dx_scatter': dx_scatter,
        'dy_scatter': dy_scatter,
        'n_used': len(measurements),
        'sources': sources,
        'status': status,
    }

def measure_detector_shifts(
    reduction_input,
    config,
    paths,
):
    """
    Measure detector shifts relative to reference night 001122.
    """

    # -------------------------------------------------------------------------
    # Reference night
    # -------------------------------------------------------------------------

    if config.night == '001122':

        logger.info(
            'Reference night 001122: detector shifts are (0, 0)'
        )

        return Table(
            rows=[
                {
                    'ccd': ccd,
                    'dx': 0.0,
                    'dy': 0.0,
                    'dx_scatter': 0.0,
                    'dy_scatter': 0.0,
                    'n_used': 0,
                    'sources': 'reference',
                    'status': 'reference',
                }
                for ccd in [1, 2, 3]
            ]
        )

    expected = _expected_detector_shifts(
        config.night
    )

    results = []

    # -------------------------------------------------------------------------
    # Measure each CCD independently
    # -------------------------------------------------------------------------

    for ccd in ['1', '2', '3']:

        logger.info(
            'Measuring detector shift for CCD%s',
            ccd
        )

        measurements = []

        for source in _REGISTRATION_SOURCES[ccd]:

            candidates = _select_registration_exposures(
                reduction_input,
                source,
                ccd,
            )

            if len(candidates) == 0:
                logger.info(
                    'CCD%s: no usable %s exposures',
                    ccd,
                    source,
                )
                continue

            reference_image = _load_registration_reference(
                source,
                ccd,
                paths,
            )

            for row in candidates:

                filename = row[f'file_ccd{ccd}']

                image, _, _ = utils.preprocess_image(
                    filename,
                    ccd=ccd,
                )

                dx, dy, error = phase_correlation_shift(
                    reference_image,
                    image,
                )

                logger.debug(
                    'CCD%s %s run %s: dx=%+.3f dy=%+.3f error=%g',
                    ccd,
                    source,
                    row['run'],
                    dx,
                    dy,
                    error,
                )

                measurements.append({
                    'run': row['run'],
                    'source': source,
                    'dx': dx,
                    'dy': dy,
                    'error': error,
                })

        result = _combine_shift_measurements(
            measurements,
            expected,
            ccd,
        )

        # ---------------------------------------------------------------------
        # Compare with historical expectation, but don't overwrite.
        # ---------------------------------------------------------------------

        if expected is not None:

            expected_dx, expected_dy = expected[
                f'ccd_{ccd}'
            ]

            result['expected_dx'] = expected_dx
            result['expected_dy'] = expected_dy

            difference = np.hypot(
                result['dx'] - expected_dx,
                result['dy'] - expected_dy,
            )

            if difference > 0.5:

                logger.warning(
                    'CCD%s: measured shift (%+.2f,%+.2f) differs from '
                    'historical expectation (%+.2f,%+.2f) by %.2f px',
                    ccd,
                    result['dx'],
                    result['dy'],
                    expected_dx,
                    expected_dy,
                    difference,
                )

        else:
            result['expected_dx'] = np.nan
            result['expected_dy'] = np.nan

        logger.info(
            'CCD%s detector shift: dx=%+.2f, dy=%+.2f px',
            ccd,
            result['dx'],
            result['dy'],
        )

        results.append(result)

    return Table(rows=results)

POLY_DEGREE = 4
N_COEFF = POLY_DEGREE + 1

REGION_COLOURS = {
    'SimTh':   'C1',
    'Sky_1':   'C0',
    'Science': 'C4',
    'Sky_2':   'C0',
    'SimLC':   'C3',
}


def _plot_tramline_diagnostic(
    extracted,
    row,
    order,
    image_type,
    filename,
):
    """
    Save a diagnostic plot of one extracted tramline.

    Shows:
        - extracted detector data around the tramline
        - collapsed cross-dispersion profile
        - central tramline
        - final adopted extraction regions

    This function is diagnostic only and does not affect the reduction.
    """

    half_window = int(
        row['extraction_half_window']
    )

    profile = collapsed_profile(
        extracted
    )

    fig, (
        ax_image,
        ax_profile,
    ) = plt.subplots(
        2,
        1,
        figsize=(6, 7),
        sharex=True,
        constrained_layout=True,
        height_ratios=[3, 1],
    )

    # -------------------------------------------------------------------------
    # Extracted tramline image
    # -------------------------------------------------------------------------

    finite = np.isfinite(
        extracted
    )

    if np.any(finite):

        vmin = np.nanpercentile(
            extracted[finite],
            5,
        )

        vmax = np.nanpercentile(
            extracted[finite],
            95,
        )

    else:

        vmin = 0.0
        vmax = 1.0

    image_plot = ax_image.imshow(
        extracted,
        origin='lower',
        aspect='auto',
        cmap='Greys_r',
        vmin=vmin,
        vmax=vmax,
    )

    ax_image.set_ylabel(
        'Dispersion pixel'
    )

    cbar = fig.colorbar(
        image_plot,
        ax=ax_image,
        orientation='horizontal',
        location='top',
        pad=0.01,
        fraction=0.05,
    )

    cbar.set_label(
        f'Counts for {order} — {image_type}'
    )

    # Central tramline.
    ax_image.axvline(
        half_window,
        color='k',
        lw=0.8,
        ls='dotted',
    )

    # -------------------------------------------------------------------------
    # Collapsed cross-dispersion profile
    # -------------------------------------------------------------------------

    ax_profile.plot(
        profile / 4096,
        color='k',
        lw=1,
    )

    # -------------------------------------------------------------------------
    # Final adopted extraction regions
    # -------------------------------------------------------------------------

    for region, colour in REGION_COLOURS.items():

        begin_column = (
            f'{region}_begin'
        )

        end_column = (
            f'{region}_end'
        )

        # Allows this diagnostic to remain usable even if a future
        # tramline table does not contain every calibration region.
        if (
            begin_column not in row.colnames
            or end_column not in row.colnames
        ):
            continue

        begin = (
            float(row[begin_column])
            + half_window
        )

        end = (
            float(row[end_column])
            + half_window
        )

        if not (
            np.isfinite(begin)
            and np.isfinite(end)
        ):
            continue

        ax_image.axvline(
            begin+0.15,
            color=colour,
            lw=1,
            ls='dashed',
        )

        ax_image.axvline(
            end-0.15,
            color=colour,
            lw=1,
            ls='dashed',
        )

        ax_profile.axvspan(
            begin,
            end,
            color=colour,
            alpha=0.25,
            lw=0,
            label=region,
        )

    # -------------------------------------------------------------------------
    # Relative cross-dispersion coordinate
    # -------------------------------------------------------------------------

    ticks = np.arange(
        5,
        2 * half_window,
        10,
    )

    ax_profile.set_xticks(
        ticks,
        ticks - half_window,
    )

    ax_profile.set_xlabel(
        'Cross-dispersion pixel relative to central fibre'
    )

    ax_profile.set_ylabel(
        r'Counts / 4096'
    )

    handles, labels = (
        ax_profile.get_legend_handles_labels()
    )

    if handles:

        ax_profile.legend(
            ncol=2,
            loc='lower center',
            fontsize=8,
        )

    fig.align_ylabels([
        ax_image,
        ax_profile,
    ])

    # -------------------------------------------------------------------------
    # Save rather than display.
    # -------------------------------------------------------------------------

    filename = Path(
        filename
    )

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        filename,
        dpi=150,
    )

    plt.close(
        fig
    )

def collapsed_profile(extracted):
    """
    Collapse an extracted order along dispersion.

    Correct for CCD-edge regions where different numbers of dispersion
    pixels contribute to different cross-dispersion pixels.
    """

    valid = np.isfinite(
        extracted
    )

    n_valid = valid.sum(
        axis=0
    )

    summed = np.nansum(
        extracted,
        axis=0
    )

    profile = np.full(
        extracted.shape[1],
        np.nan,
        dtype=float,
    )

    good = n_valid > 0

    if np.any(good):

        typical_n = np.nanmedian(
            n_valid[good]
        )

        profile[good] = (
            summed[good]
            / n_valid[good]
            * typical_n
        )

    return profile


def _order_name(row):
    """Return order name as normal Python string."""

    value = row['order_name']

    if isinstance(value, bytes):
        return value.decode()

    return str(value)


def _get_detector_shift(
    detector_shifts,
    ccd,
):
    """
    Read dx/dy from either the new Table representation or legacy dict.
    """

    # New preferred representation: astropy Table.
    if isinstance(detector_shifts, Table):

        selection = (
            detector_shifts['ccd']
            == int(ccd)
        )

        if np.any(selection):

            row = detector_shifts[
                selection
            ][0]

            return (
                float(row['dx']),
                float(row['dy']),
            )

    # Temporary compatibility with older dictionary representation.
    if isinstance(detector_shifts, dict):

        for key in [
            f'ccd_{ccd}',
            ccd,
        ]:

            if key not in detector_shifts:
                continue

            value = detector_shifts[key]

            if isinstance(value, dict):
                return (
                    float(value['dx']),
                    float(value['dy']),
                )

            return (
                float(value[0]),
                float(value[1]),
            )

    logger.warning(
        'No detector shift available for CCD%s; using (0,0)',
        ccd,
    )

    return 0.0, 0.0

def extract_trace(
    image,
    trace,
    half_window,
):
    """
    Extract a fixed-width detector region around a tramline.

    Parameters
    ----------
    image : ndarray
        Shape (dispersion, cross-dispersion).

    trace : ndarray
        Cross-dispersion coordinate of the tramline for every
        dispersion pixel.

    half_window : int
        Extraction half-width.

    Returns
    -------
    extracted : ndarray
        Shape (n_dispersion, 2*half_window).

        Pixels outside the detector are NaN.

    centres : ndarray
        Integer detector pixel used as extraction centre at every
        dispersion position.
    """

    nx, ny = image.shape

    centres = np.rint(
        trace
    ).astype(int)

    extracted = np.full(
        (nx, 2 * half_window),
        np.nan,
        dtype=np.float32,
    )

    for x, centre in enumerate(centres):

        y0 = centre - half_window
        y1 = centre + half_window

        source0 = max(
            0,
            y0,
        )

        source1 = min(
            ny,
            y1,
        )

        if source0 >= source1:
            continue

        destination0 = (
            source0 - y0
        )

        destination1 = (
            destination0
            + source1
            - source0
        )

        extracted[
            x,
            destination0:destination1
        ] = image[
            x,
            source0:source1
        ]

    return extracted, centres

def _find_trough(
    profile,
    m,
    expected,
    radius=6,
    smooth=1.5,
    prominence_fraction=0.05,
):
    """Find a dark minimum near its expected cross-dispersion position."""

    use = (
        np.isfinite(profile)
        & (np.abs(m - expected) <= radius)
    )

    if use.sum() < 5:
        return np.nan

    x = m[use]

    y = gaussian_filter1d(
        profile[use],
        smooth,
    )

    dynamic = (
        np.nanpercentile(y, 95)
        - np.nanpercentile(y, 5)
    )

    if (
        not np.isfinite(dynamic)
        or dynamic <= 0
    ):
        return np.nan

    prominence = (
        prominence_fraction
        * dynamic
    )

    peaks, _ = find_peaks(
        -y,
        prominence=prominence,
    )

    if len(peaks) == 0:
        return np.nan

    i = peaks[
        np.argmin(
            np.abs(
                x[peaks] - expected
            )
        )
    ]

    # Sub-pixel quadratic minimum.
    if 0 < i < len(x) - 1:

        a, b, _ = np.polyfit(
            x[i - 1:i + 2],
            y[i - 1:i + 2],
            2,
        )

        if a > 0:

            position = (
                -b / (2 * a)
            )

            if (
                x[i - 1]
                <= position
                <= x[i + 1]
            ):
                return float(position)

    return float(x[i])


def _fit_flat_trace(
    x,
    left,
    right,
    sigma_clip=4.0,
    iterations=5,
):
    """
    Fit the midpoint of the two Flat minima with a 4th-order polynomial.

    The half-separation between the two minima is additionally required
    to remain approximately constant along the order.
    """

    centre = 0.5 * (
        left + right
    )

    width = 0.5 * (
        right - left
    )

    good = (
        np.isfinite(x)
        & np.isfinite(centre)
        & np.isfinite(width)
        & (width > 0)
    )

    if good.sum() < N_COEFF + 1:

        raise RuntimeError(
            'Too few Flat gap measurements'
        )

    for _ in range(iterations):

        p = Polynomial.fit(
            x[good],
            centre[good],
            POLY_DEGREE,
        ).convert()

        centre_residual = (
            centre - p(x)
        )

        width_residual = (
            width
            - np.nanmedian(
                width[good]
            )
        )

        centre_sigma = utils.robust_sigma(
            centre_residual[good]
        )

        width_sigma = utils.robust_sigma(
            width_residual[good]
        )

        new_good = good.copy()

        if (
            np.isfinite(centre_sigma)
            and centre_sigma > 0
        ):

            new_good &= (
                np.abs(centre_residual)
                < sigma_clip * centre_sigma
            )

        if (
            np.isfinite(width_sigma)
            and width_sigma > 0
        ):

            new_good &= (
                np.abs(width_residual)
                < sigma_clip * width_sigma
            )

        if np.array_equal(
            new_good,
            good,
        ):
            break

        good = new_good

    if good.sum() < N_COEFF + 1:

        raise RuntimeError(
            'Too few Flat measurements after clipping'
        )

    p = Polynomial.fit(
        x[good],
        centre[good],
        POLY_DEGREE,
    ).convert()

    coeffs = np.zeros(
        N_COEFF
    )

    coeffs[
        :len(p.coef)
    ] = p.coef

    half_width = float(
        np.nanmedian(
            width[good]
        )
    )

    rms = float(
        np.sqrt(
            np.nanmean(
                (
                    centre[good]
                    - p(x[good])
                ) ** 2
            )
        )
    )

    return (
        coeffs,
        half_width,
        good,
        rms,
    )


def _find_outer_edge(
    profile,
    m,
    expected,
    bright_side,
    radius,
    smooth=1.5,
    threshold_fraction=0.20,
    min_snr=2.5,
):
    """Find the outer boundary of a Sky region."""

    use = (
        np.isfinite(profile)
        & (np.abs(m - expected) <= radius)
    )

    if use.sum() < 7:
        return np.nan

    x = m[use]

    raw = profile[use]

    y = gaussian_filter1d(
        raw,
        smooth,
    )

    left = y[
        x < expected - 1
    ]

    right = y[
        x > expected + 1
    ]

    if (
        len(left) < 2
        or len(right) < 2
    ):
        return np.nan

    left_level = np.nanmedian(
        left
    )

    right_level = np.nanmedian(
        right
    )

    if bright_side == 'right':

        dark = left_level
        bright = right_level
        direction = +1

    else:

        bright = left_level
        dark = right_level
        direction = -1

    contrast = (
        bright - dark
    )

    noise = utils.robust_sigma(
        raw - y
    )

    if contrast <= 0:
        return np.nan

    if (
        np.isfinite(noise)
        and noise > 0
        and contrast < min_snr * noise
    ):
        return np.nan

    threshold = (
        dark
        + threshold_fraction * contrast
    )

    crossings = []

    for i in range(
        len(x) - 1
    ):

        crosses = (
            (y[i] - threshold)
            * (y[i + 1] - threshold)
            <= 0
        )

        correct_direction = (
            direction
            * (y[i + 1] - y[i])
            > 0
        )

        if not (
            crosses
            and correct_direction
        ):
            continue

        if y[i + 1] == y[i]:

            position = (
                0.5
                * (
                    x[i]
                    + x[i + 1]
                )
            )

        else:

            position = (
                x[i]
                + (
                    (threshold - y[i])
                    * (x[i + 1] - x[i])
                    / (y[i + 1] - y[i])
                )
            )

        crossings.append(
            position
        )

    if not crossings:
        return np.nan

    return float(
        min(
            crossings,
            key=lambda value:
                abs(value - expected),
        )
    )

def median_profile(
    extracted_rows,
    min_valid=1,
):
    """
    Median cross-dispersion profile from several dispersion rows.

    Columns with fewer than `min_valid` finite detector pixels remain NaN.
    This avoids warnings from taking np.nanmedian of all-NaN columns.

    Returns
    -------
    profile : ndarray
        Median profile.

    n_valid : ndarray
        Number of finite contributing pixels in each cross-dispersion column.
    """

    extracted_rows = np.asarray(
        extracted_rows,
        dtype=float,
    )

    n_valid = np.sum(
        np.isfinite(extracted_rows),
        axis=0,
    )

    profile = np.full(
        extracted_rows.shape[1],
        np.nan,
        dtype=float,
    )

    good = (
        n_valid >= min_valid
    )

    if np.any(good):

        # Every selected column contains at least one finite value,
        # so there are no all-NaN slices here.
        profile[good] = np.nanmedian(
            extracted_rows[:, good],
            axis=0,
        )

    return profile, n_valid

# =============================================================================
# FIT ONE ORDER
# =============================================================================

def _fit_flat_order(
    image,
    row,
    dx=0.0,
    dy=0.0,
    diagnostic_file=None,
):
    """
    Fit the nightly tramline and Sky/Science geometry for one order.

    If the Flat is too faint for a reliable fit, the detector-shifted
    reference polynomial is retained.
    """

    order = _order_name(
        row
    )

    half_window = int(
        row['extraction_half_window']
    )

    x = np.arange(
        image.shape[0],
        dtype=float,
    )

    m = (
        np.arange(
            2 * half_window
        )
        - half_window
    )

    # -------------------------------------------------------------------------
    # Shift long-term reference into current detector position.
    # -------------------------------------------------------------------------

    coeffs = np.array([
        float(
            row[f'tramline_coeff_{i}']
        )
        for i in range(N_COEFF)
    ])

    coeffs = utils.shifted_coefficients(
        coeffs,
        dx=dx,
        dy=dy,
    )

    trace = Polynomial(
        coeffs
    )(x)

    extracted, centres = (
        extract_trace(
            image,
            trace,
            half_window,
        )
    )

    # Reference relative positions of the two dark gaps.
    old_left = 0.5 * (
        float(row['Sky_1_end'])
        + float(row['Science_begin'])
    )

    old_right = 0.5 * (
        float(row['Science_end'])
        + float(row['Sky_2_begin'])
    )

    # -------------------------------------------------------------------------
    # Measure the two dark minima as function of dispersion pixel.
    # -------------------------------------------------------------------------

    measurements = []

    for x0 in range(
        0,
        len(extracted),
        4,
    ):

        lo = max(
            0,
            x0 - 2,
        )

        hi = min(
            len(extracted),
            x0 + 3,
        )

        # Edge-safe median profile.
        # A single valid detector row is sufficient here.
        profile, _ = median_profile(
            extracted[lo:hi],
            min_valid=1,
        )

        # Nothing from this part of the order is on the CCD.
        if not np.any(np.isfinite(profile)):
            continue

        # Account for integer centring of extracted array.
        fractional_offset = (
            trace[x0]
            - centres[x0]
        )

        expected_left = (
            old_left
            + fractional_offset
        )

        expected_right = (
            old_right
            + fractional_offset
        )

        # _find_trough() itself decides whether enough valid pixels
        # exist around each expected minimum.
        left = _find_trough(
            profile,
            m,
            expected_left,
        )

        right = _find_trough(
            profile,
            m,
            expected_right,
        )

        if (
            np.isfinite(left)
            and np.isfinite(right)
            and right > left
        ):

            measurements.append(
                (
                    x0,
                    centres[x0] + left,
                    centres[x0] + right,
                )
            )
            
    # -------------------------------------------------------------------------
    # Fit nightly trace.
    # -------------------------------------------------------------------------

    left_gap = old_left
    right_gap = old_right

    n_good = 0
    rms = np.nan
    trace_status = 'shifted reference'

    minimum_points = 20
    minimum_fractional_span = 0.40

    if len(measurements) >= minimum_points:

        measured_x, left, right = np.asarray(
            measurements
        ).T

        x_span = (
            np.nanmax(measured_x)
            - np.nanmin(measured_x)
        )

        fractional_span = (
            x_span
            / (image.shape[0] - 1)
        )

        if fractional_span >= minimum_fractional_span:

            try:

                (
                    fitted_coeffs,
                    half_width,
                    good,
                    rms,
                ) = _fit_flat_trace(
                    measured_x,
                    left,
                    right,
                )

                coeffs = fitted_coeffs

                left_gap = -half_width
                right_gap = +half_width

                n_good = int(
                    good.sum()
                )

                trace_status = 'fitted'

            except RuntimeError as error:

                logger.debug(
                    '%s: Flat trace fit failed: %s',
                    order,
                    error,
                )

        else:

            logger.info(
                '%s: Flat trace only covers %.0f%% of dispersion; '
                'retaining shifted reference',
                order,
                100 * fractional_span,
            )

    # -------------------------------------------------------------------------
    # Save adopted current polynomial.
    # -------------------------------------------------------------------------

    for i, value in enumerate(
        coeffs
    ):

        row[
            f'tramline_coeff_{i}'
        ] = value

    # Re-extract around final adopted trace.
    trace = Polynomial(
        coeffs
    )(x)

    extracted, _ = extract_trace(
        image,
        trace,
        half_window,
    )

    profile = collapsed_profile(
        extracted
    )

    # -------------------------------------------------------------------------
    # Refine Science-gap minima from collapsed profile.
    # -------------------------------------------------------------------------

    profile_left = _find_trough(
        profile,
        m,
        left_gap,
        radius=4,
        prominence_fraction=0.02,
    )

    profile_right = _find_trough(
        profile,
        m,
        right_gap,
        radius=4,
        prominence_fraction=0.02,
    )

    minima_measured = 0

    if (
        np.isfinite(profile_left)
        and abs(
            profile_left
            - left_gap
        ) < 3
    ):

        left_gap = profile_left
        minima_measured += 1

    if (
        np.isfinite(profile_right)
        and abs(
            profile_right
            - right_gap
        ) < 3
    ):

        right_gap = profile_right
        minima_measured += 1

    # -------------------------------------------------------------------------
    # Outer Sky boundaries.
    # -------------------------------------------------------------------------

    old_sky1_begin = float(
        row['Sky_1_begin']
    )

    old_sky2_end = float(
        row['Sky_2_end']
    )

    old_sky1_width = abs(
        float(row['Sky_1_end'])
        - old_sky1_begin
    )

    old_sky2_width = abs(
        old_sky2_end
        - float(row['Sky_2_begin'])
    )

    # Fallback: move the old outer edges with their corresponding gaps.
    sky1_begin = (
        old_sky1_begin
        + (
            left_gap
            - old_left
        )
    )

    sky2_end = (
        old_sky2_end
        + (
            right_gap
            - old_right
        )
    )

    measured_sky1 = _find_outer_edge(
        profile,
        m,
        sky1_begin,
        bright_side='right',
        radius=max(
            6.0,
            old_sky1_width,
        ),
    )

    measured_sky2 = _find_outer_edge(
        profile,
        m,
        sky2_end,
        bright_side='left',
        radius=max(
            6.0,
            old_sky2_width,
        ),
    )

    outer_measured = 0

    if (
        np.isfinite(measured_sky1)
        and measured_sky1 < left_gap
        and abs(
            measured_sky1
            - sky1_begin
        ) < max(
            4.0,
            old_sky1_width,
        )
    ):

        sky1_begin = measured_sky1
        outer_measured += 1

    if (
        np.isfinite(measured_sky2)
        and measured_sky2 > right_gap
        and abs(
            measured_sky2
            - sky2_end
        ) < max(
            4.0,
            old_sky2_width,
        )
    ):

        sky2_end = measured_sky2
        outer_measured += 1

    # -------------------------------------------------------------------------
    # Final extraction geometry.
    #
    # Science goes minimum-to-minimum.
    # -------------------------------------------------------------------------

    row['Sky_1_begin'] = sky1_begin
    row['Sky_1_end'] = left_gap

    row['Science_begin'] = left_gap
    row['Science_end'] = right_gap

    row['Sky_2_begin'] = right_gap
    row['Sky_2_end'] = sky2_end

    # Optional diagnostic columns.
    diagnostic_values = {
        'tramline_half_width':
            0.5 * (
                right_gap
                - left_gap
            ),

        'tramline_fit_rms':
            rms,

        'tramline_fit_npoints':
            n_good,
    }

    for name, value in (
        diagnostic_values.items()
    ):

        if name in row.colnames:
            row[name] = value

    if diagnostic_file is not None:

        _plot_tramline_diagnostic(
            extracted,
            row,
            order,
            image_type='Flat',
            filename=diagnostic_file,
        )

    return row

    logger.info(
        '%s: Flat trace=%s, N=%d, RMS=%s, '
        'minima=%d/2, outer Sky edges=%d/2',
        order,
        trace_status,
        n_good,
        (
            f'{rms:.3f}'
            if np.isfinite(rms)
            else '---'
        ),
        minima_measured,
        outer_measured,
    )

    return row


# =============================================================================
# PUBLIC NIGHTLY TRAMLINE FIT
# =============================================================================

def fit_nightly_tramlines(
    master_flat,
    detector_shifts,
    config,
    paths,
):
    """
    Fit all nightly tramlines from the master Flat.

    The long-term reference geometry is shifted by the detector-registration
    measurement before each order is locally refitted.
    """

    output_file = (
        paths.flat
        / 'tramlines.fits'
    )

    # Cache/restart behaviour.
    if (
        output_file.exists()
        and not config.overwrite
    ):

        logger.info(
            'Loading existing nightly tramlines: %s',
            output_file,
        )

        return Table.read(
            output_file
        )

    reference_file = (
        paths.repository
        / 'velocereduction'
        / 'veloce_reference_data'
        / 'tramline_reference_001122.fits'
    )

    if not reference_file.exists():

        raise FileNotFoundError(
            'Reference tramline file not found: '
            f'{reference_file}'
        )

    nightly_tramlines = Table.read(
        reference_file
    ).copy()

    logger.info(
        'Fitting nightly geometry for %d orders',
        len(nightly_tramlines),
    )

    for i, row in enumerate(
        nightly_tramlines
    ):

        order = _order_name(
            row
        )

        ccd = order[4]

        dx, dy = _get_detector_shift(
            detector_shifts,
            ccd,
        )

        diagnostic_file = None

        if config.diagnostics == 'full':

            diagnostic_file = (
                paths.debug
                / 'tramlines'
                / f'{order}_flat.png'
            )

        try:

            nightly_tramlines[i] = (
                _fit_flat_order(
                    master_flat[
                        f'ccd_{ccd}'
                    ],
                    row,
                    dx=dx,
                    dy=dy,
                    diagnostic_file=diagnostic_file,
                )
            )

        except Exception as error:

            logger.exception(
                '%s: nightly tramline fit failed: %s',
                order,
                error,
            )

    nightly_tramlines.write(
        output_file,
        overwrite=True,
    )

    logger.info(
        'Saved nightly tramlines: %s',
        output_file,
    )

    return nightly_tramlines


# =============================================================================
# BASIC NOTEBOOK SUMMARY
# =============================================================================

def show_summary(
    nightly_tramlines,
):
    """Show a compact summary of nightly tramline quality."""

    order_index = np.arange(
        len(nightly_tramlines)
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6, 4),
        sharex=True,
        constrained_layout=True,
    )

    if (
        'tramline_fit_rms'
        in nightly_tramlines.colnames
    ):

        axes[0].plot(
            order_index,
            nightly_tramlines[
                'tramline_fit_rms'
            ],
            '.',
        )

    axes[0].set_ylabel(
        'Tramline RMS / pixel'
    )

    if (
        'tramline_half_width'
        in nightly_tramlines.colnames
    ):

        axes[1].plot(
            order_index,
            nightly_tramlines[
                'tramline_half_width'
            ],
            '.',
        )

    axes[1].set_ylabel(
        'Science half-width / pixel'
    )

    axes[1].set_xlabel(
        'Order index'
    )

    plt.show()
    plt.close(fig)