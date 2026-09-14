import numpy as np
from astropy.table import Table

from velocereduction.wavelength import (
    cross_validate_wavelength_surface,
    fit_wavelength_surface,
    make_surface_cv_folds,
    select_wavelength_surface_degree,
)


def synthetic_table(seed=5, n=1600):
    rng=np.random.default_rng(seed)
    y=rng.uniform(0,4111,n); m=rng.integers(103,141,n)
    yn=(y-2055.5)/2055.5; mn=(m-121.5)/18.5
    ml=72000 + 1800*yn + 22*yn**2 + 8*mn + 3*yn*mn
    wavelength=ml/m + rng.normal(0,1.5e-5,n)
    return Table(dict(
        y=y, order=m, wavelength_nm=wavelength,
        y_uncertainty=np.full(n,0.02),
        wavelength_uncertainty_nm=np.zeros(n),
        used_for_wavelength_fit=np.ones(n,dtype=bool),
    ))


def test_blocked_folds_cover_every_line():
    t=synthetic_table(n=300)
    fold=make_surface_cv_folds(t["y"],t["order"],y_bounds=(0,4111),n_folds=5,y_blocks=10)
    assert len(fold)==len(t)
    assert set(np.unique(fold))==set(range(5))


def test_surface_validation_includes_7x5_and_prefers_compact_model():
    t=synthetic_table()
    cv=cross_validate_wavelength_surface(
        t,y_bounds=(0,4111),order_bounds=(103,140),
        y_degrees=(2,3,5,7),order_degrees=(1,2,3,5),n_folds=5,
    )
    assert np.any((cv["y_degree"]==7)&(cv["order_degree"]==5))
    chosen=select_wavelength_surface_degree(cv)
    # The generator is only quadratic in y and linear in order; CV should not
    # require the maximal 7x5 model.
    assert int(chosen["n_parameters"]) < (7+1)*(5+1)


def test_wavelength_solution_broadcasts_scalar_order():
    coefficients = np.zeros((2, 2))
    coefficients[0, 0] = 60000.0
    coefficients[1, 0] = 1000.0

    solution = WavelengthSolution(
        coefficients=coefficients,
        y_center=2055.5,
        y_scale=2055.5,
        order_center=120.0,
        order_scale=20.0,
    )

    y = np.arange(4112, dtype=float)

    wavelength = solution.wavelength(y, 125)

    assert wavelength.shape == y.shape
    assert np.all(np.isfinite(wavelength))