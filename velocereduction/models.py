"""
Data products passed between the main stages of the Veloce reduction.

The dataclasses in this module describe the scientific information retained by
the pipeline, rather than the algorithms used to derive it.  They therefore
provide a compact overview of the reduction products and their relationships.

Array convention
----------------
Detector and extracted-order arrays are dispersion-first:

    detector image       (n_dispersion, n_cross_dispersion)
    order matrix          (n_dispersion, n_order_pixels)
    extracted spectrum    (n_dispersion, n_components)

For Veloce, a trimmed detector normally has 4112 dispersion pixels and
4096 cross-dispersion pixels, while the extracted order matrix currently
contains 81 cross-dispersion pixels.

Variances have the same shape as their corresponding flux/count arrays and
are expressed in the square of the corresponding flux/count units.
"""

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Detector-level products
# ---------------------------------------------------------------------------

@dataclass
class DetectorFrame:
    """One preprocessed Veloce CCD exposure.

    This is the first calibrated data product created from a raw CCD frame.
    The overscan has been removed, but the detector counts remain in ADU.

    Attributes
    ----------
    image
        Overscan-subtracted detector counts, with shape
        ``(n_dispersion, n_cross_dispersion)``.
    variance
        Per-pixel statistical variance in ADU^2.  This includes read noise
        estimated from the exposure overscan and, where enabled, photon noise
        calculated using the characterised detector gain.
    header
        FITS header of the original exposure.
    ccd
        Veloce CCD identifier: ``"1"``, ``"2"``, or ``"3"``.
    readout_mode
        Detector readout configuration, currently ``"2Amp"`` or ``"4Amp"``.
    overscan_median
        Median overscan level measured independently for each amplifier.
    overscan_rms
        Read-noise estimate in ADU measured from each amplifier overscan.
    quality_mask
        Integer bit mask with the same shape as ``image``.  Zero denotes an
        unflagged pixel; non-zero bits identify pixels that should not normally
        contribute to extraction, such as saturated pixels.
    """

    image: np.ndarray
    variance: np.ndarray
    header: object
    ccd: str
    readout_mode: str
    overscan_median: dict
    overscan_rms: dict
    quality_mask: np.ndarray


# ---------------------------------------------------------------------------
# Flat-field and fibre-geometry products
# ---------------------------------------------------------------------------

@dataclass
class FibreGeometry:
    """Flat-derived cross-dispersion geometry of one echelle order.

    The fibre positions are measured from the high-S/N Flat and represented as
    smooth functions of dispersion.  This geometry is then held fixed when
    extracting calibration and science exposures, so noisy science spectra
    cannot change the inferred fibre positions or widths.

    Attributes
    ----------
    components
        Fibre identifiers in physical slit order.  This includes the 19 science
        fibres and five sky fibres.
    slots
        Nominal slit-slot offsets relative to the reference fibre.  Gaps in the
        physical slit are retained, so these are not simply consecutive fibre
        numbers.
    centres
        Cross-dispersion centre of every fibre at every dispersion pixel, with
        shape ``(n_dispersion, n_fibres)``.
    sigma
        Smooth common Gaussian width of the fibre profiles at each dispersion
        pixel, in detector pixels.
    separation
        Smooth separation between adjacent nominal slit slots at each
        dispersion pixel, in detector pixels.
    bundle_offset
        Smooth displacement of the fibre bundle relative to the traced order
        centre.
    fibre_offsets
        Small, fixed offsets of individual fibres from a perfectly regular
        slit grid, in detector pixels.
    sampled_x
        Dispersion pixels at which the Flat geometry was measured directly.
    sampled_sigma
        Fibre widths measured at ``sampled_x`` before smooth interpolation.
    sampled_separation
        Fibre separations measured at ``sampled_x`` before smooth interpolation.
    sampled_bundle_offset
        Bundle offsets measured at ``sampled_x`` before smooth interpolation.
    """

    components: tuple
    slots: np.ndarray
    centres: np.ndarray
    sigma: np.ndarray
    separation: np.ndarray
    bundle_offset: np.ndarray
    fibre_offsets: np.ndarray
    sampled_x: np.ndarray
    sampled_sigma: np.ndarray
    sampled_separation: np.ndarray
    sampled_bundle_offset: np.ndarray


@dataclass
class FlatOrder:
    """Flat-field products for one extracted echelle order.

    This object contains both the detector-space Flat products required for
    response correction and, when fibre extraction is requested, the
    fibre-resolved Flat products used to describe relative fibre throughput.

    Attributes
    ----------
    order_name
        Unique order label, for example ``"ccd_2_order_120"``.
    matrix
        Extracted Flat order in detector-pixel space, with shape
        ``(n_dispersion, n_order_pixels)``.
    smooth
        Smooth model of the Flat illumination used to separate large-scale
        illumination from small-scale detector response.
    response
        Detector response map derived from the ratio of the measured and smooth
        Flat order.  Science and calibration order matrices are divided by this
        response before spectral extraction.
    blaze
        Smooth one-dimensional blaze / order-throughput function along
        dispersion.
    trace_offset
        Sub-pixel displacement of the traced order relative to the
        integer-centred extracted order matrix at each dispersion pixel.
    geometry
        Flat-derived fibre geometry.  Present only when fibre-resolved
        extraction is requested.
    fibre_flat
        Flux extracted independently for each illuminated fibre.
    fibre_smooth
        Smooth model of ``fibre_flat`` along dispersion.
    fibre_relative_response
        Relative fibre-throughput response derived from the extracted Flat
        fibres.
    """

    order_name: str
    matrix: np.ndarray
    smooth: np.ndarray
    response: np.ndarray
    blaze: np.ndarray
    trace_offset: np.ndarray
    geometry: FibreGeometry | None = None
    fibre_flat: np.ndarray | None = None
    fibre_smooth: np.ndarray | None = None
    fibre_relative_response: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Spectral-extraction products
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Spectrum extracted from one detector-space order matrix.

    Two extraction modes are supported:

    ``"summed"``
        Direct fractional-pixel aperture extraction.  Components are typically
        the Science and sky apertures.

    ``"fibre"``
        Simultaneous profile fitting of the individual illuminated fibres.
        Components then correspond to the physical science and sky fibres.

    Attributes
    ----------
    flux
        Extracted flux with shape ``(n_dispersion, n_components)``.
    variance
        Statistical variance corresponding to ``flux``.
    components
        Names or physical fibre identifiers corresponding to the second array
        axis.
    extraction_mode
        ``"summed"`` or ``"fibre"``.
    covariance
        Optional covariance between simultaneously fitted fibre amplitudes,
        with shape ``(n_dispersion, n_components, n_components)``.  This is
        important when recombining neighbouring deblended fibres.
    background
        Optional fitted cross-dispersion background at each dispersion pixel,
        with shape ``(n_dispersion,)``.
    """

    flux: np.ndarray
    variance: np.ndarray
    components: tuple
    extraction_mode: str
    covariance: np.ndarray | None = None
    background: np.ndarray | None = None

    def component_indices(self, components):
        """Return second-axis indices corresponding to selected components."""
        return np.array(
            [self.components.index(component) for component in components],
            dtype=int,
        )


@dataclass
class ExtractedExposure:
    """Extracted calibration exposure from one CCD.

    This groups the detector-space spectra from the echelle orders belonging to
    a single exposure.  It is primarily used for calibration observations such
    as FibTh, SimTh, and SimLC before the wavelength model is constructed.

    Attributes
    ----------
    run
        Observatory run/exposure identifier.
    kind
        Calibration type, for example ``"FibTh"``, ``"SimTh"``, or ``"SimLC"``.
    ccd
        CCD identifier.
    mjd_mid
        Mid-exposure Modified Julian Date.
    exptime
        Exposure time in seconds.
    orders
        Physical echelle-order numbers represented in the exposure.
    result
        Extracted flux and variance.
    order_names
        Pipeline order labels corresponding to ``orders``.
    fibre_flux
        Optional per-fibre calibration spectra.
    fibre_variance
        Variance corresponding to ``fibre_flux``.
    """

    run: str
    kind: str
    ccd: str
    mjd_mid: float
    exptime: float
    orders: np.ndarray
    result: ExtractionResult
    order_names: tuple
    fibre_flux: np.ndarray | None = None
    fibre_variance: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Wavelength-calibrated science products
# ---------------------------------------------------------------------------

@dataclass
class ScienceOrder:
    """Final wavelength-calibrated spectrum of one science echelle order.

    Attributes
    ----------
    order
        Physical echelle-order number.
    wavelength_nm
        Wavelength solution evaluated for the extracted spectrum, in nm.
    barycentric_wavelength_nm
        Wavelength grid after applying the barycentric correction, in nm.
    flux
        Final science flux on the common wavelength grid.
    variance
        Statistical variance corresponding to ``flux``.
    sky
        Sky spectrum used for sky subtraction, when available.
    fibre_flux
        Optional wavelength-aligned spectra of the individual science fibres.
    fibre_variance
        Variance corresponding to ``fibre_flux``.
    fibre_native_wavelength_nm
        Native wavelength solution of each fibre before interpolation onto the
        common wavelength grid.
    """

    order: int
    wavelength_nm: np.ndarray
    barycentric_wavelength_nm: np.ndarray
    flux: np.ndarray
    variance: np.ndarray
    sky: np.ndarray | None = None
    fibre_flux: np.ndarray | None = None
    fibre_variance: np.ndarray | None = None
    fibre_native_wavelength_nm: np.ndarray | None = None


@dataclass
class ScienceExposure:
    """Final reduced science spectrum from one CCD exposure.

    The individual ``ScienceOrder`` objects retain the order-by-order spectra,
    while this container stores metadata that apply to the complete exposure.

    Attributes
    ----------
    run
        Observatory run/exposure identifier.
    object_name
        Target name from the observation metadata.
    ccd
        CCD identifier.
    mjd_mid
        Mid-exposure Modified Julian Date.
    extraction_mode
        Extraction mode used to produce the spectrum.
    orders
        List of wavelength-calibrated ``ScienceOrder`` products.
    berv_kms
        Barycentric Earth radial velocity in km/s used for the wavelength
        correction.
    """

    run: str
    object_name: str
    ccd: str
    mjd_mid: float
    extraction_mode: str
    orders: list[ScienceOrder]
    berv_kms: float = np.nan