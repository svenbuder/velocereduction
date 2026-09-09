from types import SimpleNamespace
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from velocereduction import observations, tramlines
from velocereduction.config import ReductionConfig, prepare_reduction


def _header_file(filename, run=1, object_name="Star", exptime=10.):
    h = fits.Header(); h["RUN"] = run; h["OBJECT"] = object_name; h["MJD-OBS"] = 60000.; h["EXPTIME"] = exptime
    h["UTSTART"] = "01:00:00"; h["UTEND"] = "01:00:10"; h["MEANRA"] = 10.; h["MEANDEC"] = -20.; h["AIRMASS"] = 1.2; h["LCUT"] = ""
    fits.PrimaryHDU(np.zeros((2, 2)), h).writeto(filename)


def test_observation_parsing_and_metadata(tmp_path):
    paths = SimpleNamespace(repository=tmp_path)
    assert observations.raw_fits_path(paths, "001122", "12", "3").name == "22nov30012.fits"
    expected = {"SimLC":"SimLC","BiasFrame":"Bias","FlatField-Quartz":"Flat","ARC-ThAr":"FibTh","SimTh":"SimTh","SimThLong":"SimTh","DarkFrame":"Dark","Acquire":"Acquire","Star":"Science"}
    for name, kind in expected.items(): assert observations.classify_object(name) == kind
    assert observations.useful_ccds("Flat", 60.) == ("1",); assert observations.useful_ccds("SimLC", 1.) == ("2", "3")
    assert observations.useful_ccds("Science", 5.) == ("1", "2", "3"); assert observations.useful_ccds("bad", 5.) == ()
    assert observations._float("2.5") == 2.5 and np.isnan(observations._float("x"))
    assert observations._parse_log_line("bad") is None and observations._parse_log_line("0001 no colon") is None
    line = "0001  3 Target                    12:34:56" + " " * 90 + "\n"
    parsed = observations._parse_log_line(line); assert parsed["run"] == "0001" and parsed["ccd"] == "3"
    log = tmp_path / "night.log"; log.write_text(line + line.replace("0001", "0002")); assert len(observations.parse_observing_log(log)) == 2

    empty = observations.read_exposure_metadata({"1": tmp_path / "missing"}); assert empty["run_fits"] == -1
    f = tmp_path / "a.fits"; _header_file(f); meta = observations.read_exposure_metadata({"3": f})
    assert meta["object_fits"] == "Star" and meta["mjd_mid"] > meta["mjd_obs"]


def test_build_blocks_identify_and_select(tmp_path, monkeypatch):
    repo = tmp_path / "repo"; obsroot = repo / "observations" / "001122"; obsroot.mkdir(parents=True)
    paths = SimpleNamespace(repository=repo, observations=obsroot, root=tmp_path / "out", reduction_input=tmp_path / "input.txt")
    paths.root.mkdir()
    config = ReductionConfig("001122")
    for ccd in ("1", "2", "3"):
        filename = observations.raw_fits_path(paths, config.night, "0001", ccd); filename.parent.mkdir(parents=True, exist_ok=True); _header_file(filename)
    log_runs = {"0001": {"3": {"object_log":"Star", "marked_bad":False, "comments":""}}}
    table = observations.build_observation_table(log_runs, config, paths)
    assert len(table) == 1 and table["type"][0] == "Science" and all(table[f"use_ccd{c}"][0] for c in ("1","2","3"))

    block_table = Table(rows=[
        {"type":"Flat","use":True,"mjd_mid":1.0,"use_ccd1":True}, {"type":"Flat","use":True,"mjd_mid":1.001,"use_ccd1":True},
        {"type":"Science","use":True,"mjd_mid":1.002,"use_ccd1":True}, {"type":"Flat","use":True,"mjd_mid":1.1,"use_ccd1":False},
    ])
    observations.assign_calibration_blocks(block_table)
    assert block_table["calibration_block"][0] == block_table["calibration_block"][1] and block_table["calibration_block"][2] == ""
    assert len(observations.select(block_table, "Flat", "1")) == 2

    observations.write_reduction_input(table, config, paths); assert paths.reduction_input.exists()
    log = obsroot / "night.log"; log.write_text("0001  3 Target                    12:34:56" + " " * 90 + "\n")
    monkeypatch.setattr(observations, "build_observation_table", lambda *a: table.copy())
    identified = observations.identify_observations(config, paths); assert len(identified) == 1
    (obsroot / "another.log").write_text(log.read_text()); assert len(observations.identify_observations(config, paths)) == 1


def test_order_helpers():
    row = {"order_name": b"ccd_2_order_120"}; assert tramlines.order_name(row) == "ccd_2_order_120"
    assert tramlines.ccd_from_order_name("ccd_2_order_120") == "2" and tramlines.physical_order("ccd_2_order_120") == 120


def test_phase_trace_profile_and_finders(trace_row):
    image = np.arange(60.).reshape(6, 10); trace = np.array([0,1,5,8,9,12.])
    matrix, centres = tramlines.extract_trace(image, trace, 1); assert matrix.shape == (6,3) and np.isnan(matrix[-1]).all()
    row = dict(trace_row); row["tramline_coeff_0"] = 4.; assert np.allclose(tramlines.trace_from_row(row, 3), 4)
    matrix, m, offset = tramlines.extract_order_matrix(image, row, True); assert matrix.shape == (6,5) and len(offset) == 6
    assert tramlines.collapsed_profile(matrix).shape == (5,) and np.isnan(tramlines.collapsed_profile(np.full((3,4), np.nan))).all()

    x = np.arange(-10, 11.); profile = (x - 2.2) ** 2
    assert abs(tramlines._find_trough(profile, x, 2, 4) - 2.2) < .3
    assert np.isnan(tramlines._find_trough(np.full_like(profile, np.nan), x, 2))
    xx = np.arange(20.); yy = 3 + 2*xx + .01*xx**2; yy[-1] += 100
    coeff = tramlines._robust_polyfit(xx, yy, 2); assert len(coeff) == 3
    with pytest.raises(ValueError): tramlines._robust_polyfit([1,2], [1,2], 4)
    bright = np.ones_like(x) * 2; bright[(x >= -2) & (x <= 3)] = 20
    begin, end = tramlines._find_bright_interval(bright, x, -1, 2); assert begin < 0 < end
    assert tramlines._find_bright_interval(np.array([1.,2.]), np.array([0.,1.]), 0, 1) == (0,1)


def test_fit_flat_and_nightly(tmp_path, monkeypatch, trace_row):
    trace_table = Table(rows=[trace_row]); trace_table["trace_rms"] = [np.nan]; trace_table["trace_npoints"] = [0]
    image = np.ones((30, 20)); row = trace_table[0]
    monkeypatch.setattr(tramlines, "_find_trough", lambda profile, m, expected, **k: -1. if expected < 0 else 1.)
    fitted = tramlines._fit_flat_order(image, row, 0, 0, step=2)
    assert fitted["Science_begin"] == -1 and fitted["Science_end"] == 1 and fitted["trace_npoints"] > 0
    assert np.isfinite(fitted["trace_rms"])
    assert tramlines._representative_image(Table(), "SimTh", "1", ReductionConfig("001122")) is None
    rows = Table(rows=[{"file_ccd1":"x"}]); monkeypatch.setattr(tramlines.observations, "select", lambda *a: rows)
    monkeypatch.setattr(tramlines.detector, "preprocess_image", lambda *a, **k: SimpleNamespace(image=np.ones((4,4))))
    assert tramlines._representative_image(Table(), "SimTh", "1", ReductionConfig("001122")).shape == (4,4)

    repo = tmp_path / "repo"; (repo / "observations" / "001122").mkdir(parents=True); ref = repo / "velocereduction" / "veloce_reference_data"; ref.mkdir(parents=True)
    Table(rows=[trace_row]).write(ref / "tramline_reference_001122.fits")
    paths = prepare_reduction(ReductionConfig("001122", diagnostics="none"), "0.8.0", repo)
    shifts = Table(rows=[(2,0.,0.)], names=("ccd","dx","dy"))
    monkeypatch.setattr(tramlines, "_fit_flat_order", lambda image, row, dx, dy: row); monkeypatch.setattr(tramlines, "_representative_image", lambda *a: None)
    nightly = tramlines.fit_nightly_tramlines(Table(), {"ccd_2": np.ones((8,12))}, shifts, ReductionConfig("001122", diagnostics="none"), paths)
    assert len(nightly) == 1 and len(tramlines.fit_nightly_tramlines(Table(), {"ccd_2": np.ones((8,12))}, shifts, ReductionConfig("001122", diagnostics="none"), paths)) == 1
