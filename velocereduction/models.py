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

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectorFrame:
    """Overscan-corrected detector image with variance and pixel flags."""
    image: np.ndarray
    variance: np.ndarray
    header: object
    ccd: str
    readout_mode: str
    overscan_median: dict
    overscan_rms: dict
    quality_mask: np.ndarray


@dataclass
class OrderGeometry:
    """Compact location and extraction regions of one echelle order."""
    ccd: str
    order: int
    trace_coefficients: np.ndarray
    extraction_half_window: int = 40
    regions: dict = field(default_factory=dict)
    available: dict = field(default_factory=dict)
    trace_rms: float = np.nan
    trace_npoints: int = 0

    @property
    def name(self):
        return f"ccd_{self.ccd}_order_{self.order}"

    def trace(self, n_dispersion):
        y = np.arange(n_dispersion, dtype=float)
        return np.polynomial.polynomial.polyval(y, self.trace_coefficients)

    def region(self, name):
        if name not in self.regions:
            raise KeyError(f"{self.name} has no {name!r} extraction region")
        return self.regions[name]


@dataclass
class OrderMatrix:
    """Rectangular detector-space representation of one echelle order."""
    geometry: OrderGeometry
    flux: np.ndarray
    variance: np.ndarray
    quality_mask: np.ndarray
    relative_x: np.ndarray
    trace_offset: np.ndarray

    @property
    def ccd(self):
        return self.geometry.ccd

    @property
    def order(self):
        return self.geometry.order

    @property
    def name(self):
        return self.geometry.name


@dataclass
class FibreGeometry:
    """Compact smooth model of the science+sky fibre bundle in one order.

    For normalised dispersion coordinate ``u=(y-y_reference)/y_scale``:

        centre_i(y) = trace_offset(y) + bundle(u)
                      + separation(u) * slot_i + fibre_offset_i

    ``bundle``, ``separation`` and ``sigma`` are stored as polynomial
    coefficients in increasing order. The 4112-row evaluated geometry is not
    persisted; :meth:`evaluate` reconstructs it when extraction is performed.
    """
    ccd: str
    order: int
    components: tuple
    slots: np.ndarray
    fibre_offsets: np.ndarray
    bundle_coefficients: np.ndarray
    separation_coefficients: np.ndarray
    sigma_coefficients: np.ndarray
    y_reference: float
    y_scale: float
    fit_rms: float = np.nan
    fit_npoints: int = 0
    sampled_y: np.ndarray = field(default_factory=lambda: np.array([], int), repr=False)
    sampled_bundle: np.ndarray = field(default_factory=lambda: np.array([], float), repr=False)
    sampled_separation: np.ndarray = field(default_factory=lambda: np.array([], float), repr=False)
    sampled_sigma: np.ndarray = field(default_factory=lambda: np.array([], float), repr=False)

    @property
    def name(self):
        return f"ccd_{self.ccd}_order_{self.order}"

    @property
    def degree(self):
        return max(
            len(self.bundle_coefficients),
            len(self.separation_coefficients),
            len(self.sigma_coefficients),
        ) - 1

    def _u(self, y):
        return (np.asarray(y, float) - self.y_reference) / self.y_scale

    def bundle(self, y):
        return np.polynomial.polynomial.polyval(self._u(y), self.bundle_coefficients)

    def separation(self, y):
        return np.polynomial.polynomial.polyval(self._u(y), self.separation_coefficients)

    def sigma(self, y):
        return np.polynomial.polynomial.polyval(self._u(y), self.sigma_coefficients)

    def evaluate(self, n_dispersion, trace_offset=None):
        """Return evaluated centres, widths and separation for extraction."""
        y = np.arange(n_dispersion, dtype=float)
        trace_offset = np.zeros(n_dispersion) if trace_offset is None else np.asarray(trace_offset, float)
        bundle = self.bundle(y)
        separation = self.separation(y)
        sigma = self.sigma(y)
        centres = (
            trace_offset[:, None]
            + bundle[:, None]
            + separation[:, None] * np.asarray(self.slots)[None, :]
            + np.asarray(self.fibre_offsets)[None, :]
        )
        return centres, sigma, separation, bundle


@dataclass
class ExtractionResult:
    """One or more 1D spectra extracted from an OrderMatrix."""
    flux: np.ndarray
    variance: np.ndarray
    components: tuple
    extraction_mode: str
    covariance: np.ndarray | None = None
    background: np.ndarray | None = None

    def component_indices(self, components):
        return np.array([self.components.index(value) for value in components], dtype=int)

    def select(self, components):
        idx = self.component_indices(components)
        covariance = None if self.covariance is None else self.covariance[:, idx][:, :, idx]
        return ExtractionResult(
            self.flux[:, idx], self.variance[:, idx], tuple(components),
            self.extraction_mode, covariance, self.background,
        )


@dataclass
class FlatOrderCalibration:
    """Compact 1D Flat products for one order."""
    ccd: str
    order: int
    summed_flat: np.ndarray
    summed_smooth: np.ndarray
    summed_response: np.ndarray
    fibre_flat: np.ndarray | None = None
    fibre_smooth: np.ndarray | None = None
    fibre_response: np.ndarray | None = None

    @property
    def name(self):
        return f"ccd_{self.ccd}_order_{self.order}"


@dataclass
class ExtractedExposure:
    """Detector-coordinate calibration spectra for one CCD exposure."""
    run: str
    kind: str
    ccd: str
    mjd_mid: float
    exptime: float
    orders: np.ndarray
    summed: ExtractionResult
    order_names: tuple
    fibre_flux: np.ndarray | None = None
    fibre_variance: np.ndarray | None = None


@dataclass
class ScienceOrder:
    """Final wavelength-calibrated product for one physical echelle order."""
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
    """Collection of final ScienceOrder products for one CCD exposure."""
    run: str
    object_name: str
    ccd: str
    mjd_mid: float
    mode: str
    orders: list
    berv_kms: float = np.nan
