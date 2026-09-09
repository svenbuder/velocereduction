from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
from astropy.table import Table

from velocereduction import diagnostics
from velocereduction.constants import FIBRE_COMPONENTS, FIBRE_SLOTS
from velocereduction.models import FibreGeometry, FlatOrder, ScienceExposure, ScienceOrder


def _row(ccd=2, order=120):
    row = {
        "order_name": f"ccd_{ccd}_order_{order}", "extraction_half_window": 4,
        "Science_begin": -2., "Science_end": 2., "Sky_1_begin": -4., "Sky_1_end": -2.,
        "Sky_2_begin": 2., "Sky_2_end": 4., "SimTh_begin": -3.5, "SimTh_end": -1.,
        "SimLC_begin": 1., "SimLC_end": 3.5,
    }
    row.update({f"tramline_coeff_{i}": 10. if i == 0 else 0. for i in range(5)})
    return row


def _product(ccd, order, fibre=True):
    rng = np.random.default_rng(ccd)
    matrix = 100 + rng.normal(0, 2, (24, 9))
    smooth = np.full_like(matrix, 100.)
    response = matrix / smooth
    geometry = None
    if fibre:
        nx = len(matrix)
        geometry = FibreGeometry(
            FIBRE_COMPONENTS, FIBRE_SLOTS.copy(),
            np.repeat((.25 * FIBRE_SLOTS)[None, :], nx, axis=0),
            np.linspace(.77, .81, nx), np.linspace(2.39, 2.43, nx), np.zeros(nx), np.zeros(24),
            np.arange(2, nx - 2, 4), np.full(5, .79), np.full(5, 2.41), np.zeros(5),
        )
    return FlatOrder(f"ccd_{ccd}_order_{order}", matrix, smooth, response, np.ones(24), np.zeros(24), geometry)


def test_helpers(tmp_path):
    fig = matplotlib.pyplot.figure()
    path = diagnostics._save(fig, tmp_path / "x.png")
    assert path.exists() and path.stat().st_size > 0
    assert diagnostics._representative_name(["ccd_2_order_100", "ccd_2_order_120", "ccd_2_order_130"], "2") == "ccd_2_order_120"
    assert diagnostics._representative_name([], "2") is None
    centres, p16, p50, p84 = diagnostics._binned_percentiles(np.arange(20), np.arange(20), 4, 2)
    assert len(centres) == 4 and np.isfinite(p50).all()
    assert all(len(x) == 0 for x in diagnostics._binned_percentiles([1], [1], 3, 5))


def test_detector_and_read_noise_plots(tmp_path):
    shifts = Table(rows=[
        (1, 0., 0., np.nan, np.nan), (2, -1., 2., .1, .2), (3, 1., -2., .2, .1),
    ], names=("ccd", "dx", "dy", "dx_scatter", "dy_scatter"))
    assert diagnostics.plot_detector_shifts(shifts, tmp_path / "shifts.png").exists()
    noise = Table(rows=[
        (ccd, run, amp, 3. + .1 * run)
        for ccd in (1, 2, 3) for run in (1, 2) for amp in ("q1", "q2")
    ], names=("ccd", "run", "amplifier", "rms_adu"))
    assert diagnostics.plot_read_noise(noise, tmp_path / "noise.png").exists()


def test_tramline_diagnostic_and_wrapper(tmp_path, monkeypatch):
    row = Table(rows=[_row()])[0]
    flat = np.ones((24, 20)) * 100
    simth = np.ones_like(flat) * 20
    simlc = np.ones_like(flat) * 30
    path = diagnostics.plot_tramline_diagnostic(flat, simth, simlc, row, tmp_path / "tram.png")
    assert path.exists()
    path = diagnostics.plot_tramline_diagnostic(flat, None, None, row, tmp_path / "tram_missing.png")
    assert path.exists()

    table = Table(rows=[_row(2, 110), _row(2, 120), _row(2, 130)])
    paths = SimpleNamespace(figures=tmp_path / "figs", debug=tmp_path / "debug")
    calls = []
    monkeypatch.setattr(diagnostics, "plot_tramline_diagnostic", lambda *a: calls.append(a[-1]) or a[-1])
    master = {"ccd_2": flat}
    calibration = {("SimTh", "2"): simth, ("SimLC", "2"): simlc}
    assert diagnostics.save_tramline_diagnostics(master, calibration, table, SimpleNamespace(diagnostics="none"), paths) == []
    diagnostics.save_tramline_diagnostics(master, calibration, table, SimpleNamespace(diagnostics="basic"), paths)
    assert len(calls) == 1
    calls.clear()
    diagnostics.save_tramline_diagnostics(master, calibration, table, SimpleNamespace(diagnostics="full"), paths)
    assert len(calls) == 3


def test_flat_and_fibre_diagnostics(tmp_path):
    products = {p.order_name: p for p in (_product(1, 150), _product(2, 120), _product(3, 90))}
    assert diagnostics.plot_flat_response_summary(products, tmp_path / "response.png").exists()
    assert diagnostics.plot_fibre_geometry_summary(products, tmp_path / "geometry.png").exists()
    assert diagnostics.plot_fibre_profile_summary(products, tmp_path / "profile.png").exists()
    assert diagnostics.plot_fibre_profile_order(products["ccd_2_order_120"], tmp_path / "one.png").exists()
    # Missing CCD / missing geometry branches.
    sparse = {"ccd_2_order_120": _product(2, 120, fibre=False)}
    assert diagnostics.plot_flat_response_summary(sparse, tmp_path / "sparse_response.png").exists()
    assert diagnostics.plot_fibre_geometry_summary(sparse, tmp_path / "sparse_geometry.png").exists()
    assert diagnostics.plot_fibre_profile_summary(sparse, tmp_path / "sparse_profile.png").exists()


def test_wavelength_diagnostics(tmp_path):
    rows = []
    for order in (110, 120, 130):
        for y in np.linspace(200, 3900, 20):
            residual = .02 * np.sin(y / 400) + .001 * (order - 120)
            rows.append({
                "y": y, "order": order, "pixel_residual": residual,
                "velocity_residual_mps": residual * 250., "used_surface": True,
            })
    data = Table(rows=rows)
    node = SimpleNamespace(ccd="2", source="FibTh:0010", coefficients=np.zeros((8, 6)))
    assert diagnostics.plot_wavelength_fit(data, node, tmp_path / "wave.png", "FibTh").exists()
    data.remove_column("used_surface")
    assert diagnostics.plot_wavelength_fit(data, node, tmp_path / "wave_all.png").exists()

    drift = {"2": (np.array([1., 2., 3.]), np.array([0., 10., -5.])), "3": (np.array([1., 2.]), np.array([0., 4.]))}
    assert diagnostics.plot_lc_drift(drift, tmp_path / "drift.png").exists()
    offsets = {
        "1": (np.array([7, 18]), np.array([-9., -8.]), np.array([-20., 10.]), np.array([30., 25.])),
        "2": (np.array([], int), np.array([]), np.array([]), np.array([])),
        "3": (np.array([1]), np.array([0.]), np.array([5.]), np.array([20.])),
    }
    assert diagnostics.plot_fibre_wavelength_offsets(offsets, tmp_path / "fibres.png").exists()


def test_science_summary(tmp_path):
    exposures = []
    for ccd in ("1", "2", "3"):
        orders = []
        for order in (100, 110, 120):
            wave = np.linspace(500, 501, 30)
            flux = 100 + np.sin(np.linspace(0, 4, 30)) * 10
            orders.append(ScienceOrder(order, wave, wave, flux, np.full(30, 4.), np.full(30, 5.)))
        exposures.append(ScienceExposure("10", "Star", ccd, 60000., "summed", orders, 10.))
    assert diagnostics.plot_science_summary(exposures, tmp_path / "science.png").exists()
    assert diagnostics.plot_science_summary(exposures[:1], tmp_path / "science_sparse.png").exists()


def test_required_layouts_and_colours(tmp_path, monkeypatch):
    assert diagnostics.REGION_COLOURS == {
        "SimTh": "C1", "Sky_1": "C0", "Science": "C4", "Sky_2": "C0", "SimLC": "C3"
    }
    captured = []
    monkeypatch.setattr(diagnostics, "_save", lambda fig, filename, dpi=160: captured.append(fig) or filename)
    row = Table(rows=[_row()])[0]
    image = np.ones((24, 20))
    diagnostics.plot_tramline_diagnostic(image, image, image, row, tmp_path / "tram.png")
    assert len(captured[-1].axes) == 6  # 2 rows x 3 columns
    matplotlib.pyplot.close(captured[-1])

    data = Table(rows=[{
        "y": float(y), "order": order, "pixel_residual": .01 * np.sin(y),
        "velocity_residual_mps": 5. * np.sin(y), "used_surface": True,
    } for order in (110, 120) for y in np.linspace(0, 10, 20)])
    node = SimpleNamespace(ccd="2", source="SimLC:1", coefficients=np.zeros((8, 6)))
    diagnostics.plot_wavelength_fit(data, node, tmp_path / "wave.png")
    assert len(captured[-1].axes) == 7  # 6 main panels plus the residual colourbar
    matplotlib.pyplot.close(captured[-1])


def test_fibre_extraction_qa(tmp_path):
    from types import SimpleNamespace

    nx = 128
    qa = {
        "ratio": 1.0 + 0.002 * np.sin(np.arange(nx) / 4),
        "ratio_smooth": np.ones(nx),
        "fractional_structure": 0.002 * np.sin(np.arange(nx) / 4),
        "robust_rms_fractional_structure": 0.002,
    }
    geometry = SimpleNamespace(
        sigma=np.linspace(0.75, 0.85, nx),
        separation=np.linspace(2.4, 2.3, nx),
    )
    filename = tmp_path / "fibre_qa.png"
    assert diagnostics.plot_fibre_extraction_qa(
        qa, geometry, filename, title="CCD2 order 120"
    ) == filename
    assert filename.exists()
