from types import SimpleNamespace
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from velocereduction import detector
from velocereduction.config import ReductionConfig, prepare_reduction


def test_robust_amplifier_and_blocks(monkeypatch):
    assert np.isnan(detector._robust_sigma([np.nan]))

    block = np.ones((8, 8)) * 10
    block[0] = block[-1] = 8
    block[:, 0] = block[:, -1] = 8
    science, median, rms = detector._amplifier(block, border=1)
    assert science.shape == (6, 6) and median == 8 and rms == 0

    # Large row/column bias structure should not be interpreted as read noise.
    rng = np.random.default_rng(12)
    border, nx, ny, sigma = 16, 60, 80, 2.0
    structured = np.full((nx + 2 * border, ny + 2 * border), 100.0)
    row_level = np.linspace(-20, 20, nx)[:, None]
    col_level = np.linspace(-15, 15, ny)[None, :]
    structured[border:-border, :border] = 100 + row_level + rng.normal(0, sigma, (nx, border))
    structured[border:-border, -border:] = 100 + row_level + rng.normal(0, sigma, (nx, border))
    structured[:border, border:-border] = 100 + col_level + rng.normal(0, sigma, (border, ny))
    structured[-border:, border:-border] = 100 + col_level + rng.normal(0, sigma, (border, ny))
    structured[border + 5, 2] = 1000  # one bad overscan pixel
    _, measured = detector._overscan_statistics(structured, border=border)
    assert measured == pytest.approx(sigma, rel=0.12)

    raw4 = np.empty((4240, 4224), np.uint8)
    blocks, mode = detector._raw_amplifier_blocks(raw4)
    assert mode == "4Amp" and set(blocks) == {"q1", "q2", "q3", "q4"}
    del raw4

    raw2 = np.empty((4176, 4224), np.uint8)
    blocks, mode = detector._raw_amplifier_blocks(raw2)
    assert mode == "2Amp" and set(blocks) == {"q1", "q2"}
    with pytest.raises(ValueError):
        detector._raw_amplifier_blocks(np.empty((2, 2)))

    monkeypatch.setattr(
        detector, "_raw_amplifier_blocks",
        lambda raw: ({"q1": np.zeros((4, 4)), "q2": np.zeros((4, 4))}, "2Amp"),
    )
    monkeypatch.setattr(detector, "_amplifier", lambda block: (np.ones((2, 2)), 2.0, 3.0))
    image, med, noise, readout = detector.subtract_overscan(np.zeros((2, 2)))
    assert image.shape == (2, 4) and readout == "2Amp" and med["q1"] == 2 and noise["q2"] == 3

    monkeypatch.setattr(
        detector, "_raw_amplifier_blocks",
        lambda raw: ({x: np.zeros((4, 4)) for x in ("q1", "q2", "q3", "q4")}, "4Amp"),
    )
    image, *_ = detector.subtract_overscan(np.zeros((2, 2)))
    assert image.shape == (4, 4)


def test_quality_mask(monkeypatch):
    q1 = np.zeros((4, 4), float)
    q2 = np.zeros((4, 4), float)
    q1[1, 1] = detector.RAW_16BIT_MAX
    q2[2, 2] = detector.RAW_16BIT_MAX - 1

    monkeypatch.setattr(
        detector, "_raw_amplifier_blocks",
        lambda raw: ({"q1": q1, "q2": q2}, "2Amp"),
    )
    mask = detector.quality_mask_from_raw(np.zeros((1, 1)), border=1)
    assert mask.dtype == np.uint16
    assert mask.shape == (2, 4)
    assert np.count_nonzero(mask) == 1
    assert mask[0, 0] & detector.MASK_SATURATED
    assert not np.any(mask[:, 2:] & detector.MASK_SATURATED)

def test_slices_gains_and_variance(tmp_path):
    s4 = detector.amplifier_slices((4, 4), "4Amp"); assert set(s4) == {"q1", "q2", "q3", "q4"}
    s2 = detector.amplifier_slices((2, 4), "2Amp"); assert set(s2) == {"q1", "q2"}
    with pytest.raises(ValueError): detector.amplifier_slices((2, 2), "bad")

    gains = detector.load_detector_gains(); assert detector.gain_for_amplifier(gains, "1", "4Amp", "q1") > 0.0
    with pytest.raises(KeyError): detector.gain_for_amplifier(gains, "9", "4Amp", "q1")
    custom = tmp_path / "g.ecsv"; Table(rows=[("1", "2Amp", "q1", 2.0), ("1", "2Amp", "q2", 4.0)], names=("ccd", "readout_mode", "amplifier", "gain_e_per_adu")).write(custom, format="ascii.ecsv")
    loaded = detector.load_detector_gains(custom); assert detector.gain_for_amplifier(loaded, 1, "2Amp", "q2") == 4

    image = np.array([[2., -1., 8., 4.], [4., 6., 0., 12.]])
    variance = detector.variance_image(image, {"q1": 2., "q2": 3.}, "1", "2Amp", loaded, True)
    assert np.isclose(variance[0, 0], 4 + 2 / 2) and np.isclose(variance[0, 2], 9 + 8 / 4)
    read_only = detector.variance_image(image, {"q1": 2., "q2": 3.}, "1", "2Amp", loaded, False)
    assert np.all(read_only[:, :2] == 4) and np.all(read_only[:, 2:] == 9)


def test_preprocess_and_response(tmp_path, monkeypatch):
    filename = tmp_path / "raw.fits"
    fits.PrimaryHDU(np.ones((2, 2))).writeto(filename)

    monkeypatch.setattr(
        detector, "subtract_overscan",
        lambda raw: (
            np.array([[1., -2.], [3., 4.]], np.float32),
            {"q1": 0.}, {"q1": 2.}, "fake",
        ),
    )
    quality = np.zeros((2, 2), np.uint16)
    quality[1, 0] = detector.MASK_SATURATED
    monkeypatch.setattr(detector, "quality_mask_from_raw", lambda raw: quality.copy())
    monkeypatch.setattr(detector, "variance_image", lambda image, *a, **k: np.ones_like(image) * 7)

    frame = detector.preprocess_image(
        filename, "1", ReductionConfig("001122", use_poisson_variance=False)
    )
    assert frame.image[0, 1] == -2
    assert frame.variance[0, 0] == 7
    assert np.isinf(frame.variance[1, 0])
    assert frame.quality_mask[1, 0] & detector.MASK_SATURATED
    assert frame.ccd == "1"

    flux, var = detector.apply_response(
        np.array([2., 4., 6.]),
        np.array([1., 4., 9.]),
        np.array([2., 2., 0.]),
    )
    assert np.allclose(flux[:2], [1., 2.])
    assert np.allclose(var[:2], [.25, 1.])
    assert np.isnan(flux[2])

def test_gain_characterisation_helpers(tmp_path, monkeypatch):
    metadata = [
        {"filename": tmp_path / f"f{i}", "readout_mode": "2Amp", "exptime": 1., "mjd": float(i), "run": i}
        for i in range(5)
    ]
    iterator = iter(metadata); monkeypatch.setattr(detector, "_file_metadata", lambda f: next(iterator))
    pairs = detector._pair_files(range(5)); assert len(pairs) == 2
    monkeypatch.setattr(detector, "_file_metadata", lambda f: {"filename": tmp_path / "x", "readout_mode": "2Amp", "exptime": 1., "mjd": 1., "run": 1})
    with pytest.raises(ValueError): detector._pair_files(["one"])

    rng = np.random.default_rng(2); base = np.linspace(1000, 30000, 20000).reshape(100, 200)
    a = base + rng.normal(0, 20, base.shape); b = base * 1.01 + rng.normal(0, 20, base.shape)
    points = detector._pair_binned_statistics(a, b, n_bins=8, signal_range=(1000, 30000), max_level_difference=.05, min_pixels=100)
    assert len(points) >= 4
    assert detector._pair_binned_statistics(np.ones((2, 2)), np.ones((2, 2)), min_pixels=10) == []
    assert detector._pair_binned_statistics(base, base * 2, n_bins=4, signal_range=(1000, 30000), min_pixels=100) == []

    gain, rn = 2.0, 3.0; synthetic = []
    for signal in np.linspace(1000, 20000, 12):
        r = 1.0; variance = signal * (1 + r) / gain + rn ** 2 * (1 + r ** 2)
        synthetic.append({"signal_adu": signal, "variance_difference_adu2": variance, "pair_scale": r, "n_pixels": 100000})
    fit = detector._fit_gain(synthetic, overscan_rms=rn); assert fit["gain_e_per_adu"] == pytest.approx(gain, rel=2e-3); assert fit["read_noise_adu"] == pytest.approx(rn, rel=.05)


def test_characterise_and_plot(tmp_path, monkeypatch):
    meta = {"filename": tmp_path / "a", "readout_mode": "2Amp", "exptime": 1., "mjd": 1., "run": 1}
    monkeypatch.setattr(detector, "_pair_files", lambda files: [(meta, {**meta, "filename": tmp_path / "b", "mjd": 2.})])
    amps1 = {"q1": {"image": np.ones((4, 4)), "overscan_rms": 2.}, "q2": {"image": np.ones((4, 4)), "overscan_rms": 3.}}
    calls = iter([{"readout_mode": "2Amp", "amplifiers": amps1}, {"readout_mode": "2Amp", "amplifiers": amps1}])
    monkeypatch.setattr(detector, "_read_amplifiers", lambda f: next(calls))
    stats = [{"signal_adu": float(x), "variance_difference_adu2": float(2*x/2 + 18), "pair_scale": 1., "n_pixels": 10000} for x in (1000, 2000, 3000, 4000, 5000)]
    monkeypatch.setattr(detector, "_pair_binned_statistics", lambda *a, **k: stats)
    output = tmp_path / "g.ecsv"; table, diagnostic = detector.characterise_detector_gain(["a", "b"], 1, output, min_pixels=1)
    assert len(table) == 2 and output.exists() and len(diagnostic) == 2
    figures = detector.plot_gain_characterisation(diagnostic, "1", "001122", 1.0, ["a", "b"], tmp_path / "figs"); assert len(figures) == 2

    monkeypatch.setattr(detector, "_pair_binned_statistics", lambda *a, **k: [])
    calls = iter([{"readout_mode": "2Amp", "amplifiers": amps1}, {"readout_mode": "2Amp", "amplifiers": amps1}]); monkeypatch.setattr(detector, "_read_amplifiers", lambda f: next(calls))
    with pytest.raises(RuntimeError): detector.characterise_detector_gain(["a", "b"], 1)


def test_detector_debug_logging(tmp_path, monkeypatch):
    filename = tmp_path / "raw.fits"
    fits.PrimaryHDU(np.ones((2, 2))).writeto(filename)
    monkeypatch.setattr(detector, "subtract_overscan", lambda raw: (
        np.ones((2, 2), np.float32), {"q1": 0.}, {"q1": 2.5}, "fake"
    ))
    monkeypatch.setattr(detector, "quality_mask_from_raw", lambda raw: np.zeros((2, 2), np.uint16))
    monkeypatch.setattr(detector, "variance_image", lambda *a, **k: np.ones((2, 2)))
    messages = []
    monkeypatch.setattr(detector.logger, "debug", lambda message, *args: messages.append(message % args))
    detector.preprocess_image(filename, "1", ReductionConfig("001122", use_poisson_variance=False))
    assert "overscan RMS" in messages[0] and "CCD1" in messages[0]
    


def test_detector_shift_helpers():
    assert detector.expected_detector_shifts("001122")["1"] == (0, 0)
    assert detector.expected_detector_shifts("220101") is None

    for night in ("240101", "240801", "250101", "250701", "260101"):
        assert detector.expected_detector_shifts(night) is not None

    assert detector.expected_detector_shifts("270101") is None

    table = Table(rows=[(2, 1.2, -0.4)], names=("ccd", "dx", "dy"))
    assert detector.detector_shift(table, "2") == (1.2, -0.4)
    assert detector.detector_shift(table, "1") == (0.0, 0.0)


def test_phase_correlation_shift():
    reference = np.zeros((32, 32))
    reference[10:13, 15:18] = 1
    moving = np.roll(reference, 2, axis=0)

    dx, dy, error = detector.phase_correlation_shift(
        reference,
        moving,
        upsample_factor=1,
    )

    assert dx == pytest.approx(2.0)
    assert dy == pytest.approx(0.0)
    assert np.isfinite(error)


def test_detector_shift_measurement(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "observations" / "001122").mkdir(parents=True)
    reference_config = ReductionConfig("001122", diagnostics="none")
    paths = prepare_reduction(reference_config, "0.8.0", repo)

    table = detector.measure_detector_shifts(Table(), reference_config, paths)
    assert len(table) == 3
    assert set(table["status"]) == {"reference"}

    # A second call should load the cached result.
    cached = detector.measure_detector_shifts(Table(), reference_config, paths)
    assert len(cached) == 3

    # With no suitable nightly exposure, use the historical fallback where defined.
    (repo / "observations" / "240101").mkdir(parents=True)
    fallback_config = ReductionConfig("240101", diagnostics="none")
    paths2 = prepare_reduction(fallback_config, "0.8.0", repo)
    monkeypatch.setattr(detector, "_registration_candidates", lambda *a: Table())
    fallback = detector.measure_detector_shifts(Table(), fallback_config, paths2)
    assert set(fallback["status"]) == {"historical fallback"}

    # For nights without a historical fallback, use measured SimTh registrations.
    (repo / "observations" / "270101").mkdir(parents=True)
    measured_config = ReductionConfig("270101", diagnostics="none")
    paths3 = prepare_reduction(measured_config, "0.8.0", repo)
    candidates = Table(rows=[
        {"file_ccd1": "a", "file_ccd2": "b", "file_ccd3": "c"},
        {"file_ccd1": "d", "file_ccd2": "e", "file_ccd3": "f"},
    ])
    monkeypatch.setattr(detector, "_registration_candidates", lambda *a: candidates)
    monkeypatch.setattr(
        detector,
        "_reference_registration_image",
        lambda *a: np.zeros((4, 4)),
    )
    monkeypatch.setattr(
        detector,
        "preprocess_image",
        lambda *a, **k: SimpleNamespace(image=np.zeros((4, 4))),
    )
    monkeypatch.setattr(
        detector,
        "phase_correlation_shift",
        lambda *a, **k: (1.0, 2.0, 0.1),
    )

    measured = detector.measure_detector_shifts(Table(), measured_config, paths3)
    assert np.allclose(measured["dx"], 1.0)
    assert np.allclose(measured["dy"], 2.0)
    assert np.all(measured["n_used"] == 2)
    assert set(measured["status"]) == {"good"}
