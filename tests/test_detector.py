from types import SimpleNamespace
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from velocereduction import detector
from velocereduction.config import ReductionConfig


def test_robust_amplifier_and_blocks(monkeypatch):
    assert np.isnan(detector._robust_sigma([np.nan]))
    block = np.ones((8, 8)) * 10; block[0] = 8; block[-1] = 8; block[:, 0] = 8; block[:, -1] = 8
    science, median, rms = detector._amplifier(block, border=1)
    assert science.shape == (6, 6) and median == 8 and rms == 0

    raw4 = np.empty((4240, 4224), np.uint8); blocks, mode = detector._raw_amplifier_blocks(raw4)
    assert mode == "4Amp" and set(blocks) == {"q1", "q2", "q3", "q4"}
    del raw4
    raw2 = np.empty((4176, 4224), np.uint8); blocks, mode = detector._raw_amplifier_blocks(raw2)
    assert mode == "2Amp" and set(blocks) == {"q1", "q2"}
    with pytest.raises(ValueError): detector._raw_amplifier_blocks(np.empty((2, 2)))

    monkeypatch.setattr(detector, "_raw_amplifier_blocks", lambda raw: ({"q1": np.zeros((4, 4)), "q2": np.zeros((4, 4))}, "2Amp"))
    monkeypatch.setattr(detector, "_amplifier", lambda block: (np.ones((2, 2)), 2.0, 3.0))
    image, med, noise, readout = detector.subtract_overscan(np.zeros((2, 2)))
    assert image.shape == (2, 4) and readout == "2Amp" and med["q1"] == 2 and noise["q2"] == 3

    monkeypatch.setattr(detector, "_raw_amplifier_blocks", lambda raw: ({x: np.zeros((4, 4)) for x in ("q1", "q2", "q3", "q4")}, "4Amp"))
    image, *_ = detector.subtract_overscan(np.zeros((2, 2))); assert image.shape == (4, 4)


def test_slices_gains_and_variance(tmp_path):
    s4 = detector.amplifier_slices((4, 4), "4Amp"); assert set(s4) == {"q1", "q2", "q3", "q4"}
    s2 = detector.amplifier_slices((2, 4), "2Amp"); assert set(s2) == {"q1", "q2"}
    with pytest.raises(ValueError): detector.amplifier_slices((2, 2), "bad")

    gains = detector.load_detector_gains(); assert detector.gain_for_amplifier(gains, "1", "4Amp", "q1") == 1.01
    with pytest.raises(KeyError): detector.gain_for_amplifier(gains, "9", "4Amp", "q1")
    custom = tmp_path / "g.ecsv"; Table(rows=[("1", "2Amp", "q1", 2.0), ("1", "2Amp", "q2", 4.0)], names=("ccd", "readout_mode", "amplifier", "gain_e_per_adu")).write(custom, format="ascii.ecsv")
    loaded = detector.load_detector_gains(custom); assert detector.gain_for_amplifier(loaded, 1, "2Amp", "q2") == 4

    image = np.array([[2., -1., 8., 4.], [4., 6., 0., 12.]])
    variance = detector.variance_image(image, {"q1": 2., "q2": 3.}, "1", "2Amp", loaded, True)
    assert np.isclose(variance[0, 0], 4 + 2 / 2) and np.isclose(variance[0, 2], 9 + 8 / 4)
    read_only = detector.variance_image(image, {"q1": 2., "q2": 3.}, "1", "2Amp", loaded, False)
    assert np.all(read_only[:, :2] == 4) and np.all(read_only[:, 2:] == 9)


def test_preprocess_and_response(tmp_path, monkeypatch):
    filename = tmp_path / "raw.fits"; fits.PrimaryHDU(np.ones((2, 2))).writeto(filename)
    monkeypatch.setattr(detector, "subtract_overscan", lambda raw: (np.array([[1., -2.], [3., 4.]], np.float32), {"q1": 0.}, {"q1": 2.}, "fake"))
    monkeypatch.setattr(detector, "variance_image", lambda image, *a, **k: np.ones_like(image) * 7)
    frame = detector.preprocess_image(filename, "1", ReductionConfig("001122", use_poisson_variance=False))
    assert frame.image[0, 1] == -2 and np.all(frame.variance == 7) and frame.ccd == "1"

    flux, var = detector.apply_response(np.array([2., 4., 6.]), np.array([1., 4., 9.]), np.array([2., 2., 0.]))
    assert np.allclose(flux[:2], [1., 2.]) and np.allclose(var[:2], [.25, 1.]) and np.isnan(flux[2])


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
    fit = detector._fit_gain(synthetic); assert fit["gain_e_per_adu"] == pytest.approx(gain, rel=2e-3); assert fit["read_noise_adu"] == pytest.approx(rn, rel=.05)


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
    figures = detector.plot_gain_characterisation(diagnostic, tmp_path / "figs"); assert len(figures) == 2

    monkeypatch.setattr(detector, "_pair_binned_statistics", lambda *a, **k: [])
    calls = iter([{"readout_mode": "2Amp", "amplifiers": amps1}, {"readout_mode": "2Amp", "amplifiers": amps1}]); monkeypatch.setattr(detector, "_read_amplifiers", lambda f: next(calls))
    with pytest.raises(RuntimeError): detector.characterise_detector_gain(["a", "b"], 1)


def test_detector_debug_logging(tmp_path, monkeypatch):
    filename = tmp_path / "raw.fits"
    fits.PrimaryHDU(np.ones((2, 2))).writeto(filename)
    monkeypatch.setattr(detector, "subtract_overscan", lambda raw: (
        np.ones((2, 2), np.float32), {"q1": 0.}, {"q1": 2.5}, "fake"
    ))
    monkeypatch.setattr(detector, "variance_image", lambda *a, **k: np.ones((2, 2)))
    messages = []
    monkeypatch.setattr(detector.logger, "debug", lambda message, *args: messages.append(message % args))
    detector.preprocess_image(filename, "1", ReductionConfig("001122", use_poisson_variance=False))
    assert "overscan RMS" in messages[0] and "CCD1" in messages[0]
