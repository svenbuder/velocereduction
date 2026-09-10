from types import SimpleNamespace
import numpy as np
import pytest
from astropy.table import Table

from velocereduction import extraction, flat
from velocereduction.config import ReductionConfig, prepare_reduction
from velocereduction.constants import FIBRE_COMPONENTS, FIBRE_SLOTS, SCIENCE_FIBRES
from velocereduction.models import ExtractionResult, FibreGeometry


def test_flat_math(trace_row):
    data = np.arange(20., dtype=float).reshape(10,2); data[3,0] = np.nan
    smooth = flat.smooth_matrix(data, 1); assert smooth.shape == data.shape
    assert np.isnan(flat._smooth_nan(np.full(5, np.nan), 1)).all()
    response = flat.create_response(data, smooth); assert response.shape == data.shape
    blaze = flat._blaze(np.ones((10,5)), trace_row); assert np.nanmax(blaze) == 1
    relative = flat._fibre_relative_response(np.ones((10,24)), FIBRE_COMPONENTS); assert np.allclose(relative, 1)
    geometry = _geometry(10); reconstructed = flat.reconstruct_fibre_flat(geometry, np.ones((10,24)), np.zeros(10), 81); assert reconstructed.shape == (10,81)


def test_master_flat(tmp_path, monkeypatch):
    repo = tmp_path / "repo"; (repo / "observations" / "001122").mkdir(parents=True)
    paths = prepare_reduction(ReductionConfig("001122"), "0.8.0", repo)
    monkeypatch.setattr(flat.observations, "select_usable_observations", lambda table, kind, ccd: Table(rows=[{f"file_ccd{ccd}":f"{ccd}.fits","run":ccd,"mjd_mid":60000.+int(ccd)}]))
    monkeypatch.setattr(flat.detector, "preprocess_image", lambda filename, ccd, config: SimpleNamespace(
        image=np.ones((8,12)) * (10000 + int(ccd)), readout_mode="2Amp", overscan_rms={"q1":3.0,"q2":3.1}
    ))
    master = flat.create_master_flat(Table(), ReductionConfig("001122", diagnostics="none"), paths); assert set(master) == {"ccd_1","ccd_2","ccd_3"}
    assert flat.create_master_flat(Table(), ReductionConfig("001122", diagnostics="none"), paths)["ccd_1"].shape == (8,12)
    paths2 = prepare_reduction(ReductionConfig("001122", overwrite=True), "0.8.1", repo)
    monkeypatch.setattr(flat.detector, "preprocess_image", lambda *a, **k: SimpleNamespace(
        image=np.ones((8,12))*10, readout_mode="2Amp", overscan_rms={"q1":3.0,"q2":3.1}
    ))
    with pytest.raises(RuntimeError): flat.create_master_flat(Table(), ReductionConfig("001122", overwrite=True, diagnostics="none"), paths2)


def _geometry(nx):
    centres = np.repeat((2.4 * FIBRE_SLOTS)[None,:], nx, axis=0)
    return FibreGeometry(FIBRE_COMPONENTS, FIBRE_SLOTS.copy(), centres, np.full(nx,.8), np.full(nx,2.4), np.zeros(nx), np.zeros(24))


def test_flat_products_io(tmp_path, monkeypatch, trace_row):
    repo = tmp_path / "repo"; (repo / "observations" / "001122").mkdir(parents=True)
    master = {"ccd_2": np.ones((8,12))}; traces = Table(rows=[trace_row])
    config = ReductionConfig("001122", extraction_mode="summed", diagnostics="none"); paths = prepare_reduction(config, "0.8.0", repo)
    products = flat.create_flat_products(master, traces, config, paths); assert "ccd_2_order_120" in products
    loaded = flat.load_flat_products(config, paths); assert loaded["ccd_2_order_120"].response.shape == (8,5)

    config = ReductionConfig("001122", extraction_mode="fibre", overwrite=True, diagnostics="none"); paths = prepare_reduction(config, "0.8.1", repo)
    geometry = _geometry(8); monkeypatch.setattr(flat.extraction, "fit_fibre_geometry", lambda *a, **k: geometry)
    monkeypatch.setattr(flat.extraction, "extract_fibre_order", lambda *a, **k: ExtractionResult(np.ones((8,24)), np.ones((8,24)), FIBRE_COMPONENTS, "fibre", background=np.zeros(8)))
    products = flat.create_flat_products(master, traces, config, paths); assert products["ccd_2_order_120"].geometry is geometry
    loaded = flat.load_flat_products(ReductionConfig("001122", extraction_mode="fibre"), paths)
    assert loaded["ccd_2_order_120"].geometry.components == FIBRE_COMPONENTS


def test_aperture_extraction(trace_row):
    m = np.arange(-2,3.); weights = extraction.aperture_weights(m, -.25, 1.25); assert np.isclose(weights.sum(), 1.5)
    data = np.ones((3,5)); variance = np.ones_like(data) * 4
    result = extraction.extract_apertures(data, variance, m, [("A",-.25,1.25)])
    assert np.allclose(result.flux[:,0], weights.sum())
    assert np.allclose(result.variance[:,0], 4*np.sum(weights**2))
    result = extraction.extract_summed_order(data, variance, trace_row); assert result.components == ("Science","Sky_1","Sky_2")


def synthetic_profile(sigma=.78, separation=2.41, background=5.):
    m = np.arange(-40,41.); centres = separation * FIBRE_SLOTS; amplitudes = np.linspace(100,300,24)
    profile = background + extraction.integrated_gaussian_matrix(m, centres, sigma) @ amplitudes
    return m, profile, centres, amplitudes


def test_gaussian_and_geometry():
    m, profile, centres, amplitudes = synthetic_profile(); p = extraction.integrated_gaussian_matrix(m, centres, .78); assert np.allclose(p.sum(axis=0),1)
    cube = extraction.integrated_gaussian_cube(m, np.repeat(centres[None,:],2,axis=0), np.array([.78,.78])); assert cube.shape == (2,81,24) and np.allclose(cube.sum(axis=1),1)
    coeff, model = extraction._linear_profile_fit(profile, m, centres, .78); assert np.allclose(model, profile) and coeff[-1] == pytest.approx(5)
    regular = extraction._fit_regular_geometry(profile, m); assert regular[1] == pytest.approx(2.41, abs=.03)
    matrix = np.repeat(profile[None,:], 24, axis=0); collapsed = extraction.fit_collapsed_fibre_profile(matrix)
    assert collapsed["sigma"] == pytest.approx(.78, abs=.04) and collapsed["separation"] == pytest.approx(2.41, abs=.04)

    trace = np.linspace(-.2,.2,24); shifted = np.vstack([
        5 + extraction.integrated_gaussian_matrix(m, 2.41*FIBRE_SLOTS + off, .78) @ amplitudes for off in trace
    ])
    geometry = extraction.fit_fibre_geometry(shifted, trace, sample_step=3, sample_half_width=1, degree=1)
    assert geometry.centres.shape == (24,24); assert np.nanmedian(geometry.sigma) == pytest.approx(.78, abs=.08)


def test_geometry_helpers_and_failures(monkeypatch):
    x = np.arange(10.); y = 2 + .3*x; y[-1] += 20
    assert np.isfinite(extraction._smooth_samples(x,y,np.arange(10),1)).all()
    assert np.all(extraction._smooth_samples([1],[2],[0,1],3) == 2)
    assert np.isnan(extraction._smooth_samples([],[],[0,1],1)).all()
    monkeypatch.setattr(extraction, "fit_collapsed_fibre_profile", lambda *a: {"fibre_offsets":np.zeros(24),"x0":0.,"trace_offset_median":0.,"separation":2.4,"sigma":.8})
    monkeypatch.setattr(extraction, "_fit_geometry_profile", lambda *a: (_ for _ in ()).throw(ValueError()))
    with pytest.raises(RuntimeError): extraction.fit_fibre_geometry(np.ones((10,81))*100, np.zeros(10), 2, 1, 1)


def test_fibre_extraction_and_dispatch(small_geometry):
    nx = 8; m = np.arange(-40,41.); true_flux = np.linspace(100,300,24)
    matrix = np.vstack([extraction.integrated_gaussian_matrix(m, small_geometry.centres[x], .8) @ true_flux + 5 for x in range(nx)])
    variance = np.ones_like(matrix)
    result = extraction.extract_fibre_order(matrix, variance, small_geometry, return_covariance=True)
    assert np.allclose(np.nanmedian(result.flux,axis=0), true_flux, atol=1e-6); assert result.covariance.shape == (nx,24,24)
    assert np.allclose(result.background,5,atol=1e-6); assert result.select(SCIENCE_FIBRES).flux.shape == (nx,19)
    blank = extraction.extract_fibre_order(matrix, np.zeros_like(matrix), small_geometry); assert np.isnan(blank.flux).all()
    assert extraction.extract_order(matrix, variance, {}, "fibre", small_geometry).flux.shape == (nx,24)
    with pytest.raises(ValueError): extraction.extract_order(matrix, variance, {}, "fibre")
    with pytest.raises(ValueError): extraction.extract_order(matrix, variance, {}, "bad")
    assert extraction._calibration_region("FibTh") == "Science" and extraction._calibration_region("SimLC") == "SimLC"


def test_calibration_exposure_and_save(tmp_path, monkeypatch):
    ndisp = 6; m = np.arange(-40,41.); geometry = _geometry(ndisp)
    flat_product = SimpleNamespace(response=np.ones((ndisp,81)), geometry=geometry, fibre_relative_response=np.ones((ndisp,24)))
    flat_products = {"ccd_1_order_167":flat_product}
    row = {"order_name":"ccd_1_order_167","extraction_half_window":40,"Science_begin":-23.,"Science_end":23.,"Sky_1_begin":-32.,"Sky_1_end":-24.,"Sky_2_begin":24.,"Sky_2_end":34.,"SimLC_begin":30.,"SimLC_end":35.,"SimTh_begin":-35.,"SimTh_end":-30.}
    row.update({f"tramline_coeff_{i}":0. for i in range(5)}); nightly = Table(rows=[row])
    obs = Table(rows=[{"run":"0001","use_ccd1":True,"use_ccd2":False,"use_ccd3":False,"file_ccd1":"a","file_ccd2":"","file_ccd3":"","mjd_mid":60000.,"exptime":60.}])
    monkeypatch.setattr(extraction, "_calibration_rows", lambda table, kind: obs)
    monkeypatch.setattr(extraction.detector, "preprocess_image", lambda *a, **k: SimpleNamespace(image=np.ones((ndisp,81))*100, variance=np.ones((ndisp,81))))
    monkeypatch.setattr(extraction.tramlines, "extract_order_matrix", lambda image,row: (image,m))
    config = ReductionConfig("001122", extraction_mode="fibre")
    exposures = extraction.extract_calibration_exposures(Table(), nightly, flat_products, config)
    assert len(exposures["FibTh"]["1"]) == 1 and exposures["FibTh"]["1"][0].fibre_flux.shape == (1,ndisp,19)
    paths = extraction.save_calibration_exposures(exposures, tmp_path, True); assert paths and all(p.exists() for p in paths)
    assert extraction.save_extracted_exposure(exposures["FibTh"]["1"][0], tmp_path, False).exists()


def test_calibration_rows_include_simultaneous_lc():
    table = Table(rows=[
        {"type":"SimLC","use":True,"lc_requested":False},
        {"type":"SimTh","use":True,"lc_requested":True},
        {"type":"Science","use":True,"lc_requested":True},
        {"type":"Science","use":True,"lc_requested":False},
    ])
    assert len(extraction._calibration_rows(table,"SimLC")) == 3
