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
        """Return indices of requested fibre components."""
        lookup = {
            str(value).strip(): i
            for i, value in enumerate(self.components)
        }
        return np.array(
            [lookup[str(value).strip()] for value in components],
            dtype=int,
        )

    
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


@dataclass
class LineSpreadFunctionModel:
    """Smooth detector-coordinate LSF model for one calibration exposure.

    ``fwhm_coefficients`` stores a 2-D Legendre surface in normalized
    dispersion coordinate and physical echelle order.  Additional dimensionless
    shape parameters (for example ``one_over_beta`` or ``wing_fraction``) are
    represented by one-dimensional Legendre series in order.
    """
    shape: str
    fwhm_coefficients: np.ndarray
    y_center: float
    y_scale: float
    order_center: float
    order_scale: float
    fwhm_covariance: np.ndarray | None = None
    parameter_coefficients: dict = field(default_factory=dict)
    parameter_covariances: dict = field(default_factory=dict)
    order_snr_thresholds: dict = field(default_factory=dict)
    order_n_shape_lines: dict = field(default_factory=dict)
    reference_source: str = ""

    def _coordinates(self, y, order):
        y, order = np.broadcast_arrays(
            np.asarray(y, dtype=float), np.asarray(order, dtype=float)
        )
        return (
            (y - self.y_center) / self.y_scale,
            (order - self.order_center) / self.order_scale,
        )

    @property
    def fwhm_y_degree(self):
        return int(np.asarray(self.fwhm_coefficients).shape[0] - 1)

    @property
    def fwhm_order_degree(self):
        return int(np.asarray(self.fwhm_coefficients).shape[1] - 1)

    def fwhm(self, y, order):
        from numpy.polynomial.legendre import legval2d
        yn, mn = self._coordinates(y, order)
        return legval2d(yn, mn, np.asarray(self.fwhm_coefficients, float))

    def fwhm_uncertainty(self, y, order):
        if self.fwhm_covariance is None:
            return np.full(np.broadcast(y, order).shape, np.nan, dtype=float)
        from numpy.polynomial.legendre import legvander
        yn, mn = self._coordinates(y, order)
        yn = np.asarray(yn, float).ravel()
        mn = np.asarray(mn, float).ravel()
        yv = legvander(yn, self.fwhm_y_degree)
        mv = legvander(mn, self.fwhm_order_degree)
        design = np.einsum("ni,nj->nij", yv, mv).reshape(len(yn), -1)
        cov = np.asarray(self.fwhm_covariance, float)
        variance = np.einsum("ni,ij,nj->n", design, cov, design)
        shape = np.broadcast(np.asarray(y), np.asarray(order)).shape
        return np.sqrt(np.clip(variance, 0.0, None)).reshape(shape)

    def parameter(self, name, order, default=np.nan):
        coeff = self.parameter_coefficients.get(str(name))
        if coeff is None:
            return np.full(np.asarray(order).shape, default, dtype=float)
        from numpy.polynomial.legendre import legval
        order = np.asarray(order, dtype=float)
        mn = (order - self.order_center) / self.order_scale
        return legval(mn, np.asarray(coeff, float))

    def parameter_uncertainty(self, name, order):
        coeff = self.parameter_coefficients.get(str(name))
        cov = self.parameter_covariances.get(str(name))
        order = np.asarray(order, dtype=float)
        if coeff is None or cov is None:
            return np.full(order.shape, np.nan, dtype=float)
        from numpy.polynomial.legendre import legvander
        mn = ((order - self.order_center) / self.order_scale).ravel()
        design = legvander(mn, len(np.asarray(coeff)) - 1)
        variance = np.einsum(
            "ni,ij,nj->n", design, np.asarray(cov, float), design
        )
        return np.sqrt(np.clip(variance, 0.0, None)).reshape(order.shape)

    def evaluate(self, offset, y, order):
        """Evaluate the normalized pixel-integrated LSF at one line location."""
        from .utils import pixel_integrated_lsf

        kwargs = dict(fwhm=float(np.asarray(self.fwhm(y, order))))
        if self.shape == "moffat":
            kwargs["one_over_beta"] = float(
                np.asarray(self.parameter("one_over_beta", order, 0.0))
            )
        elif self.shape == "core_wing_gaussians":
            kwargs["wing_fraction"] = float(
                np.asarray(self.parameter("wing_fraction", order, 0.30))
            )
            kwargs["wing_sigma_ratio"] = float(
                np.asarray(self.parameter("wing_sigma_ratio", order, 1.9))
            )
        return pixel_integrated_lsf(offset, self.shape, **kwargs)
