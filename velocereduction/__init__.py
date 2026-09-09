__version__ = "0.8.0"

from .config import ReductionConfig, ReductionPaths, prepare_reduction, setup_logging
from . import (
    calibration, constants, detector, diagnostics, extraction, flat, flux_comparison, observations,
    pipeline, science, tellurics, tramlines, utils, velocities, wavelength,
)

__all__ = [
    "ReductionConfig", "ReductionPaths", "prepare_reduction", "setup_logging",
    "calibration", "constants", "detector", "diagnostics", "extraction", "flat", "flux_comparison",
    "observations", "pipeline", "science", "tellurics", "tramlines", "utils",
    "velocities", "wavelength",
]
