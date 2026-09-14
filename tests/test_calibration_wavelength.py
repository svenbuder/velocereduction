from types import SimpleNamespace
import numpy as np
import pytest
from astropy.table import Table

from velocereduction import calibration, wavelength
from velocereduction.calibration import PeakConfig, PeakFlag
from velocereduction.config import ReductionConfig, prepare_reduction
from velocereduction.constants import FIBRE_COMPONENTS, FIBRE_SLOT, SCIENCE_FIBRES
from velocereduction.models import ExtractionResult, ExtractedExposure


def test_peak_detection_fit_and_flags():
    y = np.arange(100.); counts = calibration.line_model(y, 1000, 50.2, .8, 10, .01, 50); variance = np.ones_like(counts)*4
    peaks, background, snr = calibration.detect_peaks(counts, variance); assert len(peaks)==1 and snr[peaks[0]]>5
    fit = calibration.fit_peak(counts, int(peaks[0]), variance, background); assert fit["y"] == pytest.approx(50.2, abs=.02); assert fit["quality_flag"] == 0
    empty, bg, s = calibration.detect_peaks(np.full(10,np.nan)); assert len(empty)==0 and np.isnan(bg).all() and np.isnan(s).all()
    assert calibration.fit_peak(np.ones(3),1,config=PeakConfig(fit_half_width=1)) is None

    config = PeakConfig(minimum_fit_snr=1e9, saturation=5); peaks,bg,_ = calibration.detect_peaks(counts, config=PeakConfig(detection_snr=1,prominence_snr=1))
    flagged = calibration.fit_peak(counts,int(peaks[0]),background=bg,config=config)
    assert flagged["quality_flag"] & PeakFlag.LOW_SNR and flagged["quality_flag"] & PeakFlag.SATURATED


def test_blends_measure_and_exposure():
    blend = Table(rows=[{"order":100,"y":10.,"quality_flag":0,"used_for_wavelength_fit":True},{"order":100,"y":10.5,"quality_flag":0,"used_for_wavelength_fit":True}])
    calibration._flag_blends(blend,1.2); assert all(np.asarray(blend["quality_flag"]) & int(PeakFlag.BLENDED)); assert not any(blend["used_for_wavelength_fit"])
    calibration._flag_blends(Table())

    x=np.arange(60.); line=calibration.line_model(x,1000,30,.8,5,0,30); spectra=np.vstack([line,np.roll(line,2)])
    table=calibration.measure_spectrum(spectra,[120,119],np.ones_like(spectra),2,"1",1.,"FibTh"); assert len(table)>=2
    with pytest.raises(ValueError): calibration.measure_spectrum(line,[120])
    result=ExtractionResult(line[:,None],np.ones((60,1)),(120,),"summed")
    exposure=ExtractedExposure("1","FibTh","2",1.,60.,np.array([120]),result,("ccd_2_order_120",))
    assert len(calibration.measure_extracted_exposure(exposure))==1
    exposure.fibre_flux=np.repeat(line[None,:,None],19,axis=2); exposure.fibre_variance=np.ones_like(exposure.fibre_flux)
    table=calibration.measure_extracted_exposure(exposure,fibres=True); assert "summed" in set(table["fibre"]) and "1" in set(table["fibre"])


def _norm_coeff(ccd="2", constant=500.):
    norm=wavelength.normalisation_for_ccd(ccd); coeff=np.zeros((2,2)); coeff[0,0]=120*constant; coeff[1,0]=20.; coeff[0,1]=5.; return norm,coeff


def test_model_bracketing_transfer_and_components():
    norm,coeff=_norm_coeff(); a=wavelength.SurfaceNode("2",1.,coeff,norm,"FibTh"); bcoeff=coeff.copy(); bcoeff[0,0]+=120; b=wavelength.SurfaceNode("2",3.,bcoeff,norm,"FibTh")
    lc0=coeff.copy(); lc0[0,0]-=12; lc1=lc0.copy(); lc1[0,0]+=24
    model=wavelength.NightWavelengthModel(
        fibth={"1":[],"2":[a,b],"3":[]}, simlc={"1":[],"2":[wavelength.SurfaceNode("2",1.,lc0,norm,"LC"),wavelength.SurfaceNode("2",3.,lc1,norm,"LC")],"3":[]},
        fibre={"1":{},"2":{f:[] for f in SCIENCE_FIBRES},"3":{}},
    )
    y=np.array([1000.,2000.]); assert wavelength.NightWavelengthModel._bracket([a],2.) == (a,a,0.)
    with pytest.raises(ValueError): wavelength.NightWavelengthModel._bracket([],1.)
    w=model.wavelength("2",120,y,2.); assert np.isfinite(w).all()
    no_lc=wavelength.NightWavelengthModel(fibth={"2":[a]},simlc={"2":[]},fibre={"2":{}}); assert np.allclose(no_lc.wavelength("2",120,y,1.),wavelength.evaluate_surface(y,120,coeff,norm))
    with pytest.raises(ValueError): wavelength.NightWavelengthModel().wavelength("2",120,y,1.)

    # Give every science fibre a correction linear with physical slit slot; sky extrapolation should recover it.
    for fibre in SCIENCE_FIBRES:
        c=np.array([[FIBRE_SLOT[fibre]*1e-4]])
        model.fibre["2"][fibre]=[wavelength.CorrectionNode("2",fibre,2.,c,norm)]
    base=model.wavelength("2",120,y,2.)
    assert np.allclose(model.component_wavelength("2",120,y,2.,1)-base,FIBRE_SLOT[1]*1e-4)
    assert np.allclose(model.component_wavelength("2",120,y,2.,"S5")-base,FIBRE_SLOT["S5"]*1e-4,atol=1e-10)
    assert np.allclose(model.component_wavelength("2",120,y,2.,"LC"),base)


def test_surface_helpers():
    norm,coeff=_norm_coeff(); y=np.array([0.,2055.5,4111.]); wave=wavelength.evaluate_surface(y,120,coeff,norm); assert wave.shape==(3,)
    correction=np.array([[.01]]); assert np.allclose(wavelength.evaluate_correction(y,120,correction,norm),.01)
    derivative=wavelength._surface_derivative(y,120,coeff,norm); assert derivative.shape==(3,)
    assert wavelength._nearest(np.array([1.,3.,5.]),np.array([2.8]))[0]==3
    assert wavelength._nearest(np.array([2.]),np.array([1.,3.])).tolist()==[2.,2.]


def test_reference_loading_and_identification(tmp_path):
    repo=tmp_path; coeffdir=repo/"velocereduction"/"wavelength_coefficients"; coeffdir.mkdir(parents=True)
    np.savetxt(coeffdir/"wavelength_coefficients_ccd_2_order_120_lc.txt",[50.,.001]); np.savetxt(coeffdir/"wavelength_coefficients_ccd_2_order_120_thxe.txt",[500.])
    assert "ccd_2_order_120" in wavelength.load_initial_wavelength_solutions(repo,"SimLC")
    ref=repo/"velocereduction"/"veloce_reference_data"; ref.mkdir(); (ref/"thar_UVES_MM090311.dat").write_text("1 5000.0 2 Th I A\n2 5100.0 1 Ar I A\n")
    lines=wavelength.load_th_reference_lines(repo); assert len(lines)==1
    table=Table(rows=[{"ccd":2,"order":120,"y":2048.,"mjd_mid":1.,"used_for_wavelength_fit":True,"y_uncertainty":.01,"signal_to_noise":100.,"fibre":"summed"}])
    shifts=Table(rows=[(2,0.,0.)],names=("ccd","dx","dy"))
    sim=wavelength.identify_simlc(table,wavelength.load_initial_wavelength_solutions(repo,"SimLC"),shifts); assert np.isfinite(sim["wavelength_angstrom"][0])
    # Build a Th line at exactly the legacy expected wavelength.
    initial=wavelength.load_initial_wavelength_solutions(repo,"FibTh"); expected=10*np.polynomial.polynomial.polyval(0.,initial["ccd_2_order_120"])
    th=wavelength.identify_th(table,initial,shifts,np.array([expected]),.1); assert th["wavelength_angstrom"][0]==expected
    th_bad=wavelength.identify_th(table,initial,shifts,np.array([expected+2]),.1); assert np.isnan(th_bad["wavelength_angstrom"][0])


def synthetic_surface_table(ccd="2"):
    norm=wavelength.normalisation_for_ccd(ccd); coeff=np.zeros((3,2)); coeff[0,0]=60000.; coeff[1,0]=20.; coeff[2,0]=2.; coeff[0,1]=30.
    rows=[]
    for order in (110,120,130):
        for y in np.linspace(300,3800,15):
            wave=wavelength.evaluate_surface(y,order,coeff,norm)
            rows.append({"ccd":int(ccd),"order":order,"y":y,"mjd_mid":60000.,"used_for_wavelength_fit":True,"y_uncertainty":.005,"signal_to_noise":100.,"wavelength_angstrom":wave*10,"fibre":"summed"})
    return Table(rows=rows),coeff


def test_surface_and_fibre_fit():
    table,truth=synthetic_surface_table(); node,data=wavelength.fit_surface(table,"2",2,1,"test",minimum_snr=1,maximum_y_uncertainty=1)
    assert node.rms_pixel<1e-6 and node.n_lines==len(table); assert "velocity_residual_mps" in data.colnames
    model=wavelength.NightWavelengthModel(fibth={"2":[node]},simlc={"2":[]},fibre={"2":{f:[] for f in SCIENCE_FIBRES}})
    fibre=table.copy(); fibre["fibre"]=np.full(len(fibre),"1"); fibre["wavelength_angstrom"] += .01
    corr=wavelength.fit_fibre_correction(fibre,model,"2",1,1,1); assert corr.rms_mps<1e-5
    too_small=table[:2]
    with pytest.raises(RuntimeError): wavelength.fit_surface(too_small,"2",2,2,minimum_snr=1,maximum_y_uncertainty=1)
    with pytest.raises(RuntimeError): wavelength.fit_fibre_correction(fibre[:2],model,"2",1,2,2)


def test_identify_from_model_save_load_summary(tmp_path):
    table,_=synthetic_surface_table(); node,_=wavelength.fit_surface(table,"2",2,1,minimum_snr=1,maximum_y_uncertainty=1)
    model=wavelength.NightWavelengthModel(fibth={"1":[],"2":[node],"3":[]},simlc={"1":[],"2":[],"3":[]},fibre={"1":{},"2":{f:[] for f in SCIENCE_FIBRES},"3":{}})
    row=table[:1].copy(); row.remove_column("wavelength_angstrom"); expected=model.wavelength("2",int(row[0]["order"]),float(row[0]["y"]),60000.)*10
    identified=wavelength.identify_th_from_model(row,model,np.array([expected]),.1); assert np.isfinite(identified["wavelength_angstrom"][0])
    filename=tmp_path/"model.fits"; wavelength.save_model(model,filename,ReductionConfig("001122")); loaded=wavelength.load_model(filename)
    assert len(loaded.fibth["2"])==1 and len(wavelength.model_summary(loaded))==1


def test_build_night_model(tmp_path,monkeypatch):
    repo=tmp_path/"repo"; (repo/"observations"/"001122").mkdir(parents=True); ref=repo/"velocereduction"/"veloce_reference_data"; ref.mkdir(parents=True); (ref/"thar_UVES_MM090311.dat").write_text("1 5000 2 Th I A\n")
    paths=prepare_reduction(ReductionConfig("001122", diagnostics="none"),"0.8.0",repo); table,_=synthetic_surface_table()
    monkeypatch.setattr(wavelength,"_measure_identify_absolute",lambda *a,**k: table)
    monkeypatch.setattr(wavelength,"fit_surface",lambda data,ccd,*a,**k:(wavelength.SurfaceNode(str(ccd),1.,np.array([[int(ccd)*500.]]),wavelength.normalisation_for_ccd(ccd),"x"),data))
    exposures={"FibTh":{c:[SimpleNamespace(run="1",fibre_flux=None)] for c in ("1","2","3")},"SimLC":{c:[] for c in ("1","2","3")},"SimTh":{c:[] for c in ("1","2","3")}}
    model=wavelength.build_night_model(exposures,Table(),ReductionConfig("001122", diagnostics="none"),paths); assert all(len(model.fibth[c])==1 for c in ("1","2","3"))
    assert wavelength.build_night_model(exposures,Table(),ReductionConfig("001122", diagnostics="none"),paths).fibth["1"]


def test_build_night_model_fibre_and_simlc(tmp_path, monkeypatch):
    repo=tmp_path/"repo"; (repo/"observations"/"001122").mkdir(parents=True); ref=repo/"velocereduction"/"veloce_reference_data"; ref.mkdir(parents=True); (ref/"thar_UVES_MM090311.dat").write_text("1 5000 2 Th I A\n")
    config=ReductionConfig("001122", extraction_mode="fibre", overwrite=True, diagnostics="none"); paths=prepare_reduction(config,"0.8.1",repo)
    absolute=Table(rows=[{"ccd":2,"order":120,"y":1000.,"mjd_mid":1.,"used_for_wavelength_fit":True,"y_uncertainty":.005,"signal_to_noise":100.,"wavelength_angstrom":5000.,"fibre":"summed"}])
    monkeypatch.setattr(wavelength,"_measure_identify_absolute",lambda *a,**k:absolute)
    def fake_surface(table,ccd,*a,**k): return wavelength.SurfaceNode(str(ccd),1.,np.array([[float(ccd)*500.]]),wavelength.normalisation_for_ccd(ccd),k.get("source","x") if k else "x"),table
    monkeypatch.setattr(wavelength,"fit_surface",fake_surface)
    monkeypatch.setattr(calibration,"measure_spectrum",lambda *a,**k:Table(rows=[{"ccd":int(a[3]),"order":120,"y":1000.,"mjd_mid":1.,"used_for_wavelength_fit":True,"y_uncertainty":.005,"signal_to_noise":100.,"wavelength_angstrom":5000.,"fibre":str(a[7])}]))
    monkeypatch.setattr(wavelength,"identify_th_from_model",lambda table,*a,**k:table)
    monkeypatch.setattr(wavelength,"fit_fibre_correction",lambda table,model,ccd,fibre,*a,**k:wavelength.CorrectionNode(str(ccd),int(fibre),1.,np.array([[0.]]),wavelength.normalisation_for_ccd(ccd)))
    fib=SimpleNamespace(run="1",fibre_flux=np.ones((1,10,19)),fibre_variance=np.ones((1,10,19)),orders=np.array([120]),ccd="2",mjd_mid=1.,kind="FibTh")
    exposures={"FibTh":{c:[fib] for c in ("1","2","3")},"SimLC":{"1":[],"2":[SimpleNamespace(run="2")],"3":[SimpleNamespace(run="3")]},"SimTh":{c:[] for c in ("1","2","3")}}
    # Make the shared fake FibTh exposure report the CCD requested by the loop.
    monkeypatch.setattr(wavelength,"_measure_identify_absolute",lambda exposure,*a,**k:absolute)
    model=wavelength.build_night_model(exposures,Table(),config,paths)
    assert len(model.simlc["2"])==1 and all(len(model.fibre[c][f])==1 for c in ("1","2","3") for f in SCIENCE_FIBRES)


def test_measure_identify_absolute(monkeypatch, tmp_path):
    exposure=SimpleNamespace(kind="SimLC")
    measured=Table(rows=[{"ccd":2,"order":120,"y":1000.}])
    monkeypatch.setattr(calibration,"measure_extracted_exposure",lambda *a,**k:measured)
    monkeypatch.setattr(wavelength,"load_initial_wavelength_solutions",lambda *a:{})
    monkeypatch.setattr(wavelength,"identify_simlc",lambda table,*a,**k:table)
    assert wavelength._measure_identify_absolute(exposure,tmp_path,Table(),np.array([5000.])) is measured
    exposure.kind="FibTh"; monkeypatch.setattr(wavelength,"identify_th",lambda table,*a,**k:table)
    assert wavelength._measure_identify_absolute(exposure,tmp_path,Table(),np.array([5000.])) is measured


def test_wavelength_qa_helpers(tmp_path, monkeypatch):
    norm = wavelength.normalisation_for_ccd("2")
    base_coeff = np.zeros((1, 1)); base_coeff[0, 0] = 120 * 500.0
    lc0 = wavelength.SurfaceNode("2", 1.0, base_coeff.copy(), norm, "SimLC:1")
    lc1_coeff = base_coeff.copy(); lc1_coeff[0, 0] += 120 * 0.001
    lc1 = wavelength.SurfaceNode("2", 2.0, lc1_coeff, norm, "SimLC:2")
    fib = wavelength.SurfaceNode("2", 1.0, base_coeff.copy(), norm, "FibTh:1")
    model = wavelength.NightWavelengthModel(
        fibth={"1": [], "2": [fib], "3": []},
        simlc={"1": [], "2": [lc0, lc1], "3": []},
        fibre={"1": {}, "2": {f: [] for f in SCIENCE_FIBRES}, "3": {}},
    )
    mjd, drift = wavelength.simlc_velocity_drift(model, "2", order=120)
    assert len(mjd) == 2 and drift[0] == pytest.approx(0.) and drift[1] > 0
    empty_mjd, empty_drift = wavelength.simlc_velocity_drift(model, "1")
    assert len(empty_mjd) == len(empty_drift) == 0

    for fibre in SCIENCE_FIBRES[:2]:
        coeff = np.array([[1e-4 * (1 + fibre / 100.)]])
        model.fibre["2"][fibre] = [wavelength.CorrectionNode("2", fibre, 1.0, coeff, norm, 20., 10)]
    fibres, slots, offsets, rms = wavelength.fibre_velocity_offsets(model, "2", order=120, mjd=1.0)
    assert len(fibres) == 2 and len(slots) == len(offsets) == len(rms) == 2
    assert all(np.isfinite(offsets))
    empty = wavelength.fibre_velocity_offsets(model, "1")
    assert all(len(x) == 0 for x in empty)

    calls = []
    monkeypatch.setattr(wavelength.diagnostics, "plot_lc_drift", lambda *a, **k: calls.append("lc") or tmp_path / "lc.png")
    monkeypatch.setattr(wavelength.diagnostics, "plot_fibre_wavelength_offsets", lambda *a, **k: calls.append("fibre") or tmp_path / "f.png")
    config = ReductionConfig("001122", extraction_mode="fibre", diagnostics="basic")
    paths = SimpleNamespace(figures=tmp_path)
    wavelength.save_model_diagnostics(model, config, paths)
    assert calls == ["lc", "fibre"]
    assert wavelength.save_model_diagnostics(model, ReductionConfig("001122", diagnostics="none"), paths) == []
