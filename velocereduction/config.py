from dataclasses import dataclass
from pathlib import Path
import logging
import sys


@dataclass
class ReductionConfig:
    night: str
    log_level: str = "INFO"
    diagnostics: str = "basic"
    extraction_mode: str = "summed"
    overwrite: bool = False
    use_poisson_variance: bool = True
    gain_file: str | Path | None = None
    flat_smooth_sigma: float = 50.0
    fibre_sample_step: int = 16
    fibre_sample_half_width: int = 4
    fibre_geometry_degree: int = 3
    wavelength_degree_y: int = 7
    wavelength_degree_m: int = 5
    fibre_wavelength_degree_y: int = 3
    fibre_wavelength_degree_m: int = 2

    def validate(self):
        if len(self.night) != 6 or not self.night.isdigit():
            raise ValueError("night must be a six-digit YYMMDD string")
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
    detector: Path
    flat: Path
    flat_mode: Path
    wavelength: Path
    wavelength_mode: Path
    science: Path
    science_products: Path
    figures: Path
    debug: Path
    reduction_input: Path
    process_log: Path
    reduction_summary: Path


def prepare_reduction(config, version, repository=None):
    config.validate()
    repository = Path(repository).expanduser().resolve() if repository else Path(__file__).resolve().parents[1]
    observations = repository / "observations" / config.night
    if not observations.exists():
        raise FileNotFoundError(f"Observation directory does not exist: {observations}")

    version_dir = version if str(version).startswith("vr_") else f"vr_{version}"
    root = repository / "reduced_data" / version_dir / config.night
    calibrations = root / "calibrations"
    wavelength = calibrations / "wavelength"
    paths = ReductionPaths(
        repository, observations, root, calibrations,
        calibrations / "detector", calibrations / "flat", calibrations / "flat" / config.extraction_mode, wavelength,
        wavelength / config.extraction_mode, root / "science",
        root / "science" / config.extraction_mode, root / "figures", root / "debug",
        root / f"reduction_input_{config.night}.txt",
        root / f"reduction_process_log_{config.night}.txt",
        root / f"reduction_summary_{config.night}.txt",
    )
    required = [paths.root, paths.detector, paths.flat, paths.flat_mode, paths.wavelength_mode, paths.science_products]
    if config.diagnostics != "none":
        required.append(paths.figures)
    if config.diagnostics == "full":
        required.append(paths.debug)
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
    logger.info("Starting VeloceReduction: night=%s, extraction=%s", config.night, config.extraction_mode)
    return logger
