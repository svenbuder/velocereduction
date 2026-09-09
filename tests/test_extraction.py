import numpy as np
from velocereduction import extraction


def test_fractional_aperture_variance():
    m = np.arange(-2, 3, dtype=float)
    data = np.ones((1, 5))
    variance = np.ones_like(data) * 4.0
    result = extraction.extract_apertures(
        data, variance, m, [("A", -0.25, 1.25)]
    )
    weights = extraction.aperture_weights(m, -0.25, 1.25)
    assert np.allclose(result.flux[0, 0], weights.sum())
    assert np.allclose(result.variance[0, 0], 4.0 * np.sum(weights ** 2))


def test_fibre_slit_order():
    assert extraction.SCIENCE_COMPONENTS == (
        7, 18, 17, 6, 16, 15, 5, 14, 13, 1,
        12, 11, 4, 10, 9, 3, 8, 19, 2,
    )
    assert extraction.SKY_COMPONENTS == ("S5", "S2", "S4", "S3", "S1")
    assert np.allclose(extraction.FIBRE_SLOTS[:2], [-12, -11])
    assert np.allclose(extraction.FIBRE_SLOTS[2:21], np.arange(-9, 10))
    assert np.allclose(extraction.FIBRE_SLOTS[-3:], [11, 12, 13])


def test_recombine_fibres_uses_covariance():
    flux = np.tile(np.arange(len(extraction.FIBRE_COMPONENTS), dtype=float), (3, 1))
    variance = np.ones_like(flux)
    covariance = np.zeros((3, flux.shape[1], flux.shape[1]))
    for i in range(flux.shape[1]):
        covariance[:, i, i] = 2.0

    # Add anti-covariance between the first two science fibres.
    i0 = extraction.FIBRE_COMPONENTS.index(extraction.SCIENCE_COMPONENTS[0])
    i1 = extraction.FIBRE_COMPONENTS.index(extraction.SCIENCE_COMPONENTS[1])
    covariance[:, i0, i1] = covariance[:, i1, i0] = -0.5

    result = extraction.ExtractionResult(
        flux, variance, extraction.FIBRE_COMPONENTS, "fibre", covariance
    )
    combined, combined_var = extraction.recombine_fibres(result)
    idx = result.component_indices(extraction.SCIENCE_COMPONENTS)

    assert np.allclose(combined, flux[:, idx].sum(axis=1))
    assert np.allclose(combined_var, 2.0 * len(idx) - 1.0)


def test_fibre_recombination_qa_detects_shape_structure():
    nx = 256
    direct = np.ones(nx) * 1000.0
    summed = extraction.ExtractionResult(
        direct[:, None],
        np.ones((nx, 1)),
        ("Science",),
        "summed",
    )

    nf = len(extraction.FIBRE_COMPONENTS)
    fibre_flux = np.zeros((nx, nf))
    science_idx = [extraction.FIBRE_COMPONENTS.index(c) for c in extraction.SCIENCE_COMPONENTS]
    wiggle = 1.0 + 0.01 * np.sin(np.arange(nx) * 2 * np.pi / 8.0)
    for i in science_idx:
        fibre_flux[:, i] = direct * wiggle / len(science_idx)

    fibre = extraction.ExtractionResult(
        fibre_flux,
        np.ones_like(fibre_flux),
        extraction.FIBRE_COMPONENTS,
        "fibre",
    )
    qa = extraction.fibre_recombination_qa(summed, fibre, smooth_sigma=20.0)

    assert np.isclose(np.nanmedian(qa["ratio"]), 1.0, atol=2e-3)
    assert qa["robust_rms_fractional_structure"] > 0.005


def test_regular_profile_debug_logging(caplog):
    import logging

    m = np.arange(-40, 41, dtype=float)
    centres = 2.1 * extraction.FIBRE_SLOTS
    amplitudes = np.ones(len(centres)) * 1000.0
    profile = extraction._gaussian_model(m, 100.0, amplitudes, centres, 0.7)

    caplog.set_level(logging.DEBUG, logger=extraction.logger.name)
    fit = extraction._fit_regular_profile(
        profile, m, extraction.FIBRE_SLOTS,
        separation0=2.1, sigma0=0.7, label="CCD3 order 85",
    )

    assert fit.success
    assert "CCD3 order 85: regular fibre fit" in caplog.text
    assert "initial d=2.100 sigma=0.700" in caplog.text


def test_fibre_qa_debug_logging(caplog):
    import logging

    nx = 128
    direct = np.ones(nx) * 1000.0
    summed = extraction.ExtractionResult(
        direct[:, None], np.ones((nx, 1)), ("Science",), "summed"
    )

    nf = len(extraction.FIBRE_COMPONENTS)
    fibre_flux = np.zeros((nx, nf))
    idx = [extraction.FIBRE_COMPONENTS.index(c) for c in extraction.SCIENCE_COMPONENTS]
    for i in idx:
        fibre_flux[:, i] = direct / len(idx)
    fibre = extraction.ExtractionResult(
        fibre_flux, np.ones_like(fibre_flux), extraction.FIBRE_COMPONENTS, "fibre"
    )

    caplog.set_level(logging.DEBUG, logger=extraction.logger.name)
    extraction.fibre_recombination_qa(summed, fibre, label="CCD3 order 85")
    assert "CCD3 order 85: fibre/summed QA" in caplog.text
