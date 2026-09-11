from dataclasses import dataclass
from pathlib import Path
import logging
import sys

from .constants import REFERENCE_NIGHT


@dataclass
class ReductionConfig:
    night: str
    log_level: str = "INFO"
    diagnostics: str = "basic"
    extraction_mode: str = "summed"
    overwrite: bool = False
    use_poisson_variance: bool = True
    gain_file: str | Path | None = None
    reference_night: str = REFERENCE_NIGHT
    flat_smooth_sigma: float = 50.0
    fibre_sample_step: int = 16
    fibre_sample_half_width: int = 4
    fibre_geometry_degree: int = 3

    def validate(self):
        for name in ("night", "reference_night"):
            value = getattr(self, name)
            if len(value) != 6 or not value.isdigit():
                raise ValueError(f"{name} must be a six-digit YYMMDD string")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"Unknown log level: {self.log_level}")
        if self.diagnostics not in {"none", "basic", "full"}:
            raise ValueError(f"Unknown diagnostics level: {self.diagnostics}")
        if self.extraction_mode not in {"summed", "fibre"}:
            raise ValueError(f"Unknown extraction mode: {self.extraction_mode}")
        return self


@dataclass(frozen=True)
class ReductionPaths:
    repository: Path
    observations: Path
    root: Path
    calibrations: Path
    science: Path
    figures: Path
    debug: Path
    reduction_input: Path
    process_log: Path
    reduction_summary: Path
    detector_shifts: Path
    order_geometry: Path
    fibre_geometry: Path
    flat_summed: Path
    flat_smooth_summed: Path
    response_summed: Path
    flat_fibres: Path
    flat_smooth_fibres: Path
    response_fibres: Path

    @property
    def reference_data(self):
        return self.repository / "velocereduction" / "veloce_reference_data"

    def reference_product(self, stem, night=None):
        night = night or REFERENCE_NIGHT
        return self.reference_data / f"{stem}_{night}.fits"


def prepare_reduction(config, version, repository=None):
    """Create the shallow nightly output tree and return all canonical paths."""
    config.validate()
    repository = (
        Path(repository).expanduser().resolve()
        if repository else Path(__file__).resolve().parents[1]
    )
    observations = repository / "observations" / config.night
    if not observations.exists():
        raise FileNotFoundError(f"Observation directory does not exist: {observations}")

    version_dir = version if str(version).startswith("vr_") else f"vr_{version}"
    root = repository / "reduced_data" / version_dir / config.night
    calibrations = root / "calibrations"
    science = root / "science"
    figures = root / "figures"
    debug = root / "debug"
    night = config.night

    paths = ReductionPaths(
        repository=repository,
        observations=observations,
        root=root,
        calibrations=calibrations,
        science=science,
        figures=figures,
        debug=debug,
        reduction_input=root / f"reduction_input_{night}.txt",
        process_log=root / f"reduction_process_log_{night}.txt",
        reduction_summary=root / f"reduction_summary_{night}.txt",
        detector_shifts=root / f"detector_shifts_{night}.fits",
        order_geometry=root / f"order_geometry_{night}.fits",
        fibre_geometry=root / f"fibre_geometry_{night}.fits",
        flat_summed=root / f"flat_summed_{night}.fits",
        flat_smooth_summed=root / f"flat_smooth_summed_{night}.fits",
        response_summed=root / f"response_summed_{night}.fits",
        flat_fibres=root / f"flat_fibres_{night}.fits",
        flat_smooth_fibres=root / f"flat_smooth_fibres_{night}.fits",
        response_fibres=root / f"response_fibres_{night}.fits",
    )

    required = [root, calibrations, science]
    if config.diagnostics != "none":
        required.append(figures)
    if config.diagnostics == "full":
        required.append(debug)
    for path in required:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(config, paths):
    logger = logging.getLogger("velocereduction")
    logger.setLevel(getattr(logging, config.log_level))
    logger.propagate = False
    for handler in logger.handlers[:]:
        if getattr(handler, "_velocereduction", False):
            logger.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    file_handler = logging.FileHandler(paths.process_log, mode="w" if config.overwrite else "a")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S",
    ))
    for handler in (console, file_handler):
        handler.setLevel(getattr(logging, config.log_level))
        handler._velocereduction = True
        logger.addHandler(handler)

    logger.info("=" * 72)
    logger.info(
        "Starting VeloceReduction: night=%s, extraction=%s, reference=%s",
        config.night, config.extraction_mode, config.reference_night,
    )
    return logger
