import logging
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

from astropy.io import fits

from scipy.ndimage import gaussian_filter1d

from velocereduction import tramlines
from velocereduction import utils

logger = logging.getLogger(__name__)



FLAT_SMOOTH_SIGMA = 50.0


# =============================================================================
# SELECT FLAT EXPOSURES
# =============================================================================

def _select_flat_exposures(
    reduction_input,
    ccd,
):
    """
    Select Flat exposures appropriate for one CCD.

    The exposure-time selection has already been encoded in use_ccd1/2/3.
    """

    selection = (
        (reduction_input['type'] == 'Flat')
        & reduction_input['use']
        & reduction_input[f'use_ccd{ccd}']
    )

    return reduction_input[
        selection
    ]


def _flat_is_usable(
    image,
    minimum_signal=5000.0,
):
    """
    Test whether a nominal Flat actually contains useful illumination.
    """

    signal = float(
        np.nanpercentile(
            image,
            99,
        )
    )

    usable = (
        np.isfinite(signal)
        and signal >= minimum_signal
    )

    return usable, signal


def _normalise_flat(
    image,
):
    """
    Normalise one Flat before combination.

    The normalisation removes exposure-to-exposure lamp intensity changes
    while preserving the detector/order structure.
    """

    scale = float(
        np.nanpercentile(
            image,
            95,
        )
    )

    if (
        not np.isfinite(scale)
        or scale <= 0
    ):

        raise ValueError(
            'Could not determine Flat normalisation'
        )

    return (
        image / scale,
        scale,
    )


# =============================================================================
# MASTER FLAT INPUT / OUTPUT
# =============================================================================

def _master_flat_filename(
    paths,
):
    return (
        paths.flat
        / 'master_flat.fits'
    )


def _read_master_flat(
    filename,
):
    """Read a previously created three-CCD master Flat."""

    result = {}

    with fits.open(
        filename,
        memmap=False,
    ) as hdul:

        for ccd in ['1', '2', '3']:

            result[
                f'ccd_{ccd}'
            ] = np.asarray(
                hdul[f'CCD{ccd}'].data,
                dtype=np.float32,
            )

    return result


def _write_master_flat(
    master_flat,
    input_runs,
    config,
    paths,
):
    """Write the master Flat for all three CCDs."""

    filename = _master_flat_filename(
        paths
    )

    primary = fits.PrimaryHDU()

    primary.header[
        'NIGHT'
    ] = config.night

    primary.header[
        'PRODUCT'
    ] = 'MASTER_FLAT'

    hdus = [
        primary
    ]

    for ccd in [
        '1',
        '2',
        '3',
    ]:

        hdu = fits.ImageHDU(
            data=master_flat[
                f'ccd_{ccd}'
            ],
            name=f'CCD{ccd}',
        )

        runs = input_runs[
            f'ccd_{ccd}'
        ]

        hdu.header[
            'NCOMBINE'
        ] = len(runs)

        hdu.header[
            'RUNS'
        ] = ','.join(runs)

        hdus.append(
            hdu
        )

    fits.HDUList(
        hdus
    ).writeto(
        filename,
        overwrite=True,
    )

    logger.info(
        'Saved master Flat: %s',
        filename,
    )


# =============================================================================
# CREATE MASTER FLAT
# =============================================================================

def create_master_flat(
    reduction_input,
    config,
    paths,
):
    """
    Create one normalised median master Flat per CCD.

    Flats are processed just in time and only one CCD is held as a stack
    at a time.
    """

    filename = _master_flat_filename(
        paths
    )

    # Cache/restart.
    if (
        filename.exists()
        and not config.overwrite
    ):

        logger.info(
            'Loading existing master Flat: %s',
            filename,
        )

        return _read_master_flat(
            filename
        )

    master_flat = {}
    input_runs = {}

    for ccd in [
        '1',
        '2',
        '3',
    ]:

        candidate_runs = (
            _select_flat_exposures(
                reduction_input,
                ccd,
            )
        )

        logger.info(
            'CCD%s: %d candidate Flat exposures',
            ccd,
            len(candidate_runs),
        )

        accepted_images = []
        accepted_runs = []

        for row in candidate_runs:

            filename_raw = row[
                f'file_ccd{ccd}'
            ]

            (
                image,
                _,
                _,
            ) = utils.preprocess_image(
                filename_raw,
                ccd=ccd,
            )

            usable, signal = (
                _flat_is_usable(
                    image
                )
            )

            if not usable:

                logger.warning(
                    'CCD%s Flat run %s rejected: '
                    '99th percentile %.0f ADU',
                    ccd,
                    row['run'],
                    signal,
                )

                continue

            image, scale = (
                _normalise_flat(
                    image
                )
            )

            accepted_images.append(
                image.astype(
                    np.float32,
                    copy=False,
                )
            )

            accepted_runs.append(
                str(row['run'])
            )

            logger.debug(
                'CCD%s Flat run %s accepted: '
                'signal %.0f ADU, normalisation %.1f',
                ccd,
                row['run'],
                signal,
                scale,
            )

        if len(
            accepted_images
        ) == 0:

            raise RuntimeError(
                'No usable Flat exposures '
                f'for CCD{ccd}'
            )

        logger.info(
            'CCD%s: combining %d Flats',
            ccd,
            len(accepted_images),
        )

        stack = np.stack(
            accepted_images,
            axis=0,
        )

        master = np.nanmedian(
            stack,
            axis=0,
        ).astype(
            np.float32
        )

        master_flat[
            f'ccd_{ccd}'
        ] = master

        input_runs[
            f'ccd_{ccd}'
        ] = accepted_runs

        # Explicitly release the potentially large cube before moving
        # to the next CCD.
        del stack
        del accepted_images

    _write_master_flat(
        master_flat,
        input_runs,
        config,
        paths,
    )

    return master_flat


# =============================================================================
# SMOOTH EXTRACTED FLAT
# =============================================================================

def _smooth_nan_1d(
    values,
    sigma,
):
    """
    Gaussian smooth a 1D array while correctly handling NaNs.

    Uses weighted convolution rather than compressing the array around NaNs.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    good = np.isfinite(
        values
    )

    if good.sum() == 0:

        return np.full_like(
            values,
            np.nan,
        )

    data = np.where(
        good,
        values,
        0.0,
    )

    weights = good.astype(
        np.float32
    )

    smooth_data = gaussian_filter1d(
        data,
        sigma=sigma,
        mode='nearest',
    )

    smooth_weights = gaussian_filter1d(
        weights,
        sigma=sigma,
        mode='nearest',
    )

    result = np.full_like(
        values,
        np.nan,
    )

    valid = (
        smooth_weights > 0.05
    )

    result[valid] = (
        smooth_data[valid]
        / smooth_weights[valid]
    )

    return result


def _smooth_extracted_flat(
    extracted_flat,
    sigma=FLAT_SMOOTH_SIGMA,
):
    """
    Smooth an extracted Flat along the dispersion/fibre direction.

    No smoothing is performed across the cross-dispersion direction.
    """

    smooth_flat = np.full_like(
        extracted_flat,
        np.nan,
        dtype=np.float32,
    )

    for m in range(
        extracted_flat.shape[1]
    ):

        smooth_flat[
            :,
            m
        ] = _smooth_nan_1d(
            extracted_flat[
                :,
                m
            ],
            sigma=sigma,
        )

    return smooth_flat


def _create_response_flat(
    extracted_flat,
    smooth_flat,
):
    """
    Divide the measured Flat by its smooth illumination model.

    The result should be close to unity and contain small-scale detector
    response variations.
    """

    response = np.full_like(
        extracted_flat,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(extracted_flat)
        & np.isfinite(smooth_flat)
        & (smooth_flat > 0)
    )

    response[valid] = (
        extracted_flat[valid]
        / smooth_flat[valid]
    )

    return response


def _create_blaze(
    smooth_flat,
    row,
):
    """
    Derive a normalised 1D blaze function from the smooth Science aperture.
    """

    half_window = int(
        row[
            'extraction_half_window'
        ]
    )

    m = np.arange(
        -half_window,
        half_window + 1,
    )

    science_begin = float(
        row['Science_begin']
    )

    science_end = float(
        row['Science_end']
    )

    science = (
        (m >= science_begin)
        & (m <= science_end)
    )

    blaze = np.nansum(
        smooth_flat[
            :,
            science
        ],
        axis=1,
    ).astype(
        np.float32
    )

    finite = (
        np.isfinite(blaze)
        & (blaze > 0)
    )

    if np.any(finite):

        normalisation = np.nanmax(
            blaze[finite]
        )

        if normalisation > 0:
            blaze /= normalisation

    return blaze


# =============================================================================
# FLAT PRODUCT FITS I/O
# =============================================================================

def _write_order_product(
    filename,
    products,
    key,
    config,
):
    """
    Write one product as a FITS file with one extension per order.
    """

    primary = fits.PrimaryHDU()

    primary.header[
        'NIGHT'
    ] = config.night

    primary.header[
        'PRODUCT'
    ] = key.upper()

    hdus = [
        primary
    ]

    for order, product in (
        products.items()
    ):

        data = product[
            key
        ]

        hdu = fits.ImageHDU(
            data=data,
            name=str(order),
        )

        hdu.header[
            'ORDER'
        ] = str(order)

        hdus.append(
            hdu
        )

    fits.HDUList(
        hdus
    ).writeto(
        filename,
        overwrite=True,
    )


# =============================================================================
# CREATE EXTRACTED FLAT PRODUCTS
# =============================================================================

def create_flat_products(
    master_flat,
    nightly_tramlines,
    config,
    paths,
):
    """
    Extract the master Flat along the nightly tramlines and derive:

        extracted_flat
        smooth_flat
        response_flat
        blaze

    Returns
    -------
    products : dict
        products[order][product_name]
    """

    products = {}

    for row in nightly_tramlines:

        order_value = row[
            'order_name'
        ]

        order = (
            order_value.decode()
            if isinstance(
                order_value,
                bytes,
            )
            else str(order_value)
        )

        ccd = order[4]

        image = master_flat[
            f'ccd_{ccd}'
        ]

        half_window = int(
            row[
                'extraction_half_window'
            ]
        )

        x = np.arange(
            image.shape[0],
            dtype=float,
        )

        coeffs = np.array([
            float(
                row[
                    f'tramline_coeff_{i}'
                ]
            )
            for i in range(5)
        ])

        trace = Polynomial(
            coeffs
        )(x)

        extracted_flat, _ = (
            tramlines.extract_trace(
                image,
                trace,
                half_window,
            )
        )

        smooth_flat = (
            _smooth_extracted_flat(
                extracted_flat
            )
        )

        response_flat = (
            _create_response_flat(
                extracted_flat,
                smooth_flat,
            )
        )

        blaze = _create_blaze(
            smooth_flat,
            row,
        )

        products[
            order
        ] = {
            'extracted_flat':
                extracted_flat,

            'smooth_flat':
                smooth_flat,

            'response_flat':
                response_flat,

            'blaze':
                blaze,
        }

        logger.debug(
            '%s: created extracted Flat, response Flat and blaze',
            order,
        )

    # -------------------------------------------------------------------------
    # Save products.
    # -------------------------------------------------------------------------

    _write_order_product(
        paths.flat
        / 'extracted_flat.fits',
        products,
        'extracted_flat',
        config,
    )

    _write_order_product(
        paths.flat
        / 'smooth_flat.fits',
        products,
        'smooth_flat',
        config,
    )

    _write_order_product(
        paths.flat
        / 'response_flat.fits',
        products,
        'response_flat',
        config,
    )

    _write_order_product(
        paths.flat
        / 'blaze.fits',
        products,
        'blaze',
        config,
    )

    logger.info(
        'Created Flat products for %d orders',
        len(products),
    )

    return products