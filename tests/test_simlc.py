"""Tests for velocereduction.simlc."""

import numpy as np
from astropy.table import Table

from velocereduction.calibration import CalibrationPeakFlag
from velocereduction.simlc import (
    SimLCLSFConfig,
    effective_fwhm,
    fit_simlc_lsf,
    pixel_integrated_moffat,
    refit_simlc_peaks,
)


def test_pixel_integrated_moffat_is_symmetric_and_normalised():
    x = np.linspace(-20.0, 20.0, 4001)
    p = pixel_integrated_moffat(x, alpha=0.8, beta=2.5)

    assert np.allclose(p, p[::-1], atol=1e-12)
    assert np.isclose(np.trapezoid(p, x), 1.0, rtol=2e-3)
    assert 1.0 < effective_fwhm(x, p) < 1.5


def _synthetic_simlc(seed=123):
    rng = np.random.default_rng(seed)
    order = 91
    npix = 512
    centers = np.arange(35.13, 480.0, 9.4)
    modes = np.arange(30000, 30000 - len(centers), -1)

    alpha, beta = 0.72, 2.2
    background = 120.0
    counts = np.full((1, npix), background)
    variance = np.full_like(counts, 25.0)

    for j, center in enumerate(centers):
        amplitude = 12000.0 * (1.0 + 0.15 * np.sin(j))
        y = np.arange(npix, dtype=float)
        counts[0] += amplitude * pixel_integrated_moffat(y - center, alpha, beta)

    counts += rng.normal(0.0, np.sqrt(variance), counts.shape)

    nearest = np.full(len(centers), 9.4)
    table = Table(
        dict(
            peak_id=np.arange(len(centers)),
            calibration_type=np.full(len(centers), "SimLC", dtype="U8"),
            ccd=np.full(len(centers), "3", dtype="U2"),
            exposure_index=np.zeros(len(centers), dtype=int),
            mjd_mid=np.full(len(centers), 59000.0),
            fibre=np.full(len(centers), -1, dtype=int),
            order=np.full(len(centers), order, dtype=int),
            # Deliberately phase-dependent initial centroid bias.
            y=centers + 0.03 * np.sin(2 * np.pi * (centers % 1.0)),
            y_uncertainty=np.full(len(centers), 0.02),
            pixel_phase=np.zeros(len(centers)),
            integrated_counts=np.full(len(centers), 12000.0),
            integrated_counts_uncertainty=np.full(len(centers), 100.0),
            signal_to_noise=np.full(len(centers), 100.0),
            background=np.full(len(centers), background),
            background_slope=np.zeros(len(centers)),
            reduced_chi2=np.full(len(centers), 20.0),
            fit_rms=np.full(len(centers), 1.0),
            fit_success=np.ones(len(centers), dtype=bool),
            maximum_signal=np.full(len(centers), 10000.0),
            fwhm=np.full(len(centers), 1.3),
            fwhm_uncertainty=np.full(len(centers), 0.05),
            nearest_peak_distance_pixel=nearest,
            quality_flag=np.full(
                len(centers), int(CalibrationPeakFlag.BAD_PROFILE_FIT), dtype=np.int64
            ),
            comb_mode=modes,
            wavelength_nm=600.0 + 0.01 * np.arange(len(centers)),
            used_for_wavelength_fit=np.zeros(len(centers), dtype=bool),
        )
    )
    return counts, variance, np.array([order]), table, centers


def test_fit_simlc_lsf_recovers_non_gaussian_profile():
    counts, variance, orders, peaks, _ = _synthetic_simlc()
    config = SimLCLSFConfig(
        minimum_lsf_snr=20.0,
        minimum_lines=10,
        maximum_lines_per_order=50,
    )

    lsf = fit_simlc_lsf(counts, orders, peaks, variance=variance, config=config)
    row = lsf.metadata[0]

    assert int(row["order"]) == 91
    assert abs(float(row["moffat_alpha"]) - 0.72) < 0.20
    assert abs(float(row["moffat_beta"]) - 2.2) < 1.0
    assert str(row["model"]) in {"moffat", "empirical"}
    assert np.isclose(np.trapezoid(lsf.profile[0], lsf.offset), 1.0, rtol=2e-3)


def test_refit_simlc_peaks_improves_centroids_and_preserves_identification():
    counts, variance, orders, peaks, truth = _synthetic_simlc()
    config = SimLCLSFConfig(
        minimum_lsf_snr=20.0,
        minimum_lines=10,
        maximum_lines_per_order=50,
    )

    lsf = fit_simlc_lsf(counts, orders, peaks, variance=variance, config=config)
    refined = refit_simlc_peaks(
        counts,
        orders,
        peaks,
        lsf,
        variance=variance,
        config=config,
    )

    before = np.median(np.abs(np.asarray(peaks["y"]) - truth))
    after = np.median(np.abs(np.asarray(refined["y"]) - truth))

    assert after < before
    assert np.array_equal(refined["comb_mode"], peaks["comb_mode"])
    assert np.allclose(refined["wavelength_nm"], peaks["wavelength_nm"])
    assert np.all(np.isfinite(refined["lsf_y_shift"]))


def test_comb_number_wavelength_roundtrip():
    from velocereduction.simlc import lasercomb_numbers_from_wavelength, lasercomb_wavelength_from_numbers
    n = np.array([19000, 22000, 25000])
    wavelength = lasercomb_wavelength_from_numbers(n)
    np.testing.assert_allclose(lasercomb_numbers_from_wavelength(wavelength), n, atol=1e-10)


def test_refit_persists_effective_fwhm_pixel():
    counts, variance, orders, peaks, _ = _synthetic_simlc()
    config = SimLCLSFConfig(minimum_lsf_snr=20.0, minimum_lines=10, maximum_lines_per_order=50)
    lsf = fit_simlc_lsf(counts, orders, peaks, variance=variance, config=config)
    refined = refit_simlc_peaks(counts, orders, peaks, lsf, variance=variance, config=config)
    assert "fwhm_pixel" in refined.colnames
    used = np.asarray(refined["comb_mode"]) >= 0
    assert np.all(np.isfinite(np.asarray(refined["fwhm_pixel"])[used]))
