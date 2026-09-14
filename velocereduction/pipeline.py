import logging

from . import extraction, flat, observations, science, tramlines, wavelength
from .config import prepare_reduction, setup_logging

logger = logging.getLogger(__name__)


def write_summary(state, config, paths):
    model = state["wavelength_model"]
    lines = [
        f"VeloceReduction night {config.night}", f"extraction_mode = {config.extraction_mode}",
        f"observations = {len(state['reduction_input'])}", f"tramlines = {len(state['nightly_tramlines'])}",
        f"flat_orders = {len(state['flat_products'])}", f"science_ccd_exposures = {len(state['science_exposures'])}",
    ]
    if config.diagnostics != "none":
        lines.append(f"diagnostic_figures = {len(list(paths.figures.rglob('*.png')))}")
    for ccd in ("1", "2", "3"):
        lines.append(f"fibth_nodes_ccd{ccd} = {len(model.fibth[ccd])}")
        lines.append(f"simlc_nodes_ccd{ccd} = {len(model.simlc[ccd])}")
        if config.extraction_mode == "fibre":
            lines.append(f"fibre_wavelength_solutions_ccd{ccd} = {sum(bool(model.fibre[ccd][f]) for f in model.fibre[ccd])}")
    paths.reduction_summary.write_text("\n".join(lines) + "\n")


def reduce_night(config, version="0.8.0", repository=None):
    """Run the complete reduction; algorithms remain exposed through the individual modules."""
    paths = prepare_reduction(config, version, repository); setup_logging(config, paths)
    logger.info("[1/6] Identifying observations")
    reduction_input = observations.identify_observations(config, paths)
    logger.info("[2/6] Registering detectors")
    detector_shifts = tramlines.measure_detector_shifts(reduction_input, config, paths)
    logger.info("[3/6] Building master Flat and nightly tramlines")
    master_flat = flat.create_master_flat(reduction_input, config, paths)
    nightly_tramlines = tramlines.fit_nightly_tramlines(reduction_input, master_flat, detector_shifts, config, paths)
    logger.info("[4/6] Building %s Flat products", config.extraction_mode)
    flat_products = flat.create_flat_products(master_flat, nightly_tramlines, config, paths)

    logger.info("[5/6] Extracting calibrations and fitting wavelength solution")
    wavelength_file = paths.wavelength_mode / "wavelength_model.fits"
    calibration_exposures = None
    if wavelength_file.exists() and not config.overwrite:
        wavelength_model = wavelength.load_model(wavelength_file)
    else:
        calibration_exposures = extraction.extract_calibration_exposures(
            reduction_input, nightly_tramlines, flat_products, config,
        )
        extraction.save_calibration_exposures(calibration_exposures, paths.wavelength_mode / "extracted", True)
        wavelength_model = wavelength.build_night_model(calibration_exposures, detector_shifts, config, paths)

    logger.info("[6/6] Extracting science spectra")
    science_exposures = science.extract_science_exposures(
        reduction_input, nightly_tramlines, flat_products, wavelength_model, config, paths,
    )
    state = {
        "paths": paths, "reduction_input": reduction_input, "detector_shifts": detector_shifts,
        "master_flat": master_flat, "nightly_tramlines": nightly_tramlines, "flat_products": flat_products,
        "calibration_exposures": calibration_exposures, "wavelength_model": wavelength_model,
        "science_exposures": science_exposures,
    }
    write_summary(state, config, paths)
    if config.diagnostics != "none":
        logger.info("Saved %d nightly QA figures in %s", len(list(paths.figures.rglob("*.png"))), paths.figures)
    logger.info("Reduction complete: %s", paths.root)
    return state
