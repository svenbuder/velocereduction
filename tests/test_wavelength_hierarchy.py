import numpy as np
from astropy.table import Table

from velocereduction.wavelength import (
    FibreShiftModel,
    PixelShiftSurface,
    WavelengthModel,
    build_hybrid_static_peak_table,
    fit_fibre_corrections,
    fit_shift_from_peak_table,
    fit_time_corrections,
    fit_wavelength_surface,
)

Y_BOUNDS = (0.0, 4111.0)
ORDER_BOUNDS = (100.0, 105.0)


def _static_solution(seed=2):
    rng = np.random.default_rng(seed)
    y = rng.uniform(80, 4030, 500)
    order = rng.integers(100, 106, len(y))
    yn = (y - 2055.5) / 2055.5
    mn = (order - 102.5) / 2.5
    mlambda = 60000.0 + 900.0 * yn + 7.0 * yn**2 + 18.0 * mn + 2.5 * yn * mn
    wave = mlambda / order
    return fit_wavelength_surface(
        y, order, wave,
        y_degree=2, order_degree=1,
        y_bounds=Y_BOUNDS, order_bounds=ORDER_BOUNDS,
    ).solution


def _line_table(solution, shift, mjd=60000.0, seed=4, n=180):
    rng = np.random.default_rng(seed)
    yref = rng.uniform(100, 4000, n)
    order = rng.integers(100, 106, n)
    wave = solution.wavelength(yref, order)
    measured = yref + shift(yref, order)
    return Table(dict(
        y=measured,
        y_uncertainty=np.full(n, 0.01),
        order=order,
        wavelength_nm=wave,
        wavelength_uncertainty_nm=np.full(n, np.nan),
        used_for_wavelength_fit=np.ones(n, dtype=bool),
        quality_flag=np.zeros(n, dtype=np.int64),
        mjd_mid=np.full(n, mjd),
        calibration_type=np.full(n, 'TEST'),
    )), yref


def test_shift_surface_recovers_detector_displacement():
    static = _static_solution()
    truth = lambda y, m: 0.18 + 3e-5 * (y - 2055.5) + 0.012 * (m - 102.5)
    lines, yref = _line_table(static, truth)
    fit, _ = fit_shift_from_peak_table(
        lines, static, y_bounds=Y_BOUNDS, order_bounds=ORDER_BOUNDS,
        y_degree=1, order_degree=1,
    )
    test_y = np.array([500.0, 2055.5, 3600.0])
    test_m = np.array([100, 103, 105])
    np.testing.assert_allclose(fit.surface.shift(test_y, test_m), truth(test_y, test_m), atol=2e-5)


def test_hybrid_transfer_removes_fixed_simlc_coordinate_offset():
    static = _static_solution()
    fibth, _ = _line_table(static, lambda y, m: np.zeros_like(y), seed=10)
    simlc, _ = _line_table(static, lambda y, m: 0.73 + 2e-5*(y-2055.5), seed=11)
    hybrid, transfer, adjusted = build_hybrid_static_peak_table(
        fibth, simlc, static,
        y_bounds=Y_BOUNDS, order_bounds=ORDER_BOUNDS,
        transfer_y_degree=1, transfer_order_degree=0,
    )
    good = np.isfinite(np.asarray(adjusted['reference_y'], float))
    np.testing.assert_allclose(
        np.asarray(adjusted['y'], float)[good],
        np.asarray(adjusted['reference_y'], float)[good],
        atol=3e-5,
    )
    assert len(hybrid) == len(fibth) + len(simlc)


def test_fibre_and_time_models_have_correct_sign_and_source_zero_point():
    static = _static_solution()
    fibre_sets = {}
    for fibre, shift in [(-1, -0.22), (1, 0.18)]:
        table, _ = _line_table(static, lambda y, m, s=shift: np.full_like(y, s), seed=20+fibre)
        fibre_sets[fibre] = table
    fibre_model, qa, _ = fit_fibre_corrections(
        fibre_sets, static,
        y_bounds=Y_BOUNDS, order_bounds=ORDER_BOUNDS,
        y_degree=0, order_degree=0,
    )
    # Linear slit interpolation: fibre 0 lies halfway between -1 and +1.
    assert abs(float(fibre_model.shift(2000.0, 103, 0)) + 0.02) < 2e-5

    # SimLC contains a fixed illumination offset of +1.1 pix which must vanish
    # when each source is differenced against its own reference exposure.
    simlc = []
    for j, (mjd, drift) in enumerate([(60000.0, 0.0), (60000.5, 0.14), (60001.0, 0.28)]):
        table, _ = _line_table(
            static,
            lambda y, m, d=drift: np.full_like(y, 1.1 + d),
            mjd=mjd, seed=30+j,
        )
        simlc.append(table)
    time_model, qa_time, all_models = fit_time_corrections(
        static_solution=static,
        simlc_line_sets=simlc,
        reference_mjd=60000.25,
        y_bounds=Y_BOUNDS, order_bounds=ORDER_BOUNDS,
        y_degree=0, order_degree=0,
    )
    assert time_model.source == 'SimLC'
    assert abs(float(time_model.shift(2000.0, 103, 60000.25))) < 2e-5
    assert abs(float(time_model.shift(2000.0, 103, 60000.5)) - 0.07) < 2e-5

    # Verify final sign convention: current coordinate = reference + fibre + time.
    fibre_surface = PixelShiftSurface(
        coefficients=np.array([[0.18]]),
        y_center=2055.5, y_scale=2055.5,
        order_center=102.5, order_scale=2.5,
    )
    model = WavelengthModel(
        static=static,
        fibre=FibreShiftModel({1: fibre_surface}),
        time=time_model,
    )
    yref = np.array([800.0, 2100.0, 3500.0])
    order = np.array([100, 103, 105])
    current_y = yref + 0.18 + 0.07
    np.testing.assert_allclose(
        model.wavelength(current_y, order, fibre=1, mjd=60000.5),
        static.wavelength(yref, order),
        atol=5e-10,
    )
