import velocereduction as VR
from pathlib import Path
import pytest

def test_calibrate_wavelength():
    print('\n  --> Testing: calibrate_wavelength()')

    VR.config.date = '001122'
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'
    science_object = 'HIP69673'

    print('\n  --> Testing: calibrate_wavelength() with optimise_lc_solution=False, correct_barycentric_velocity=False, fit_voigt_for_rv=False, create_overview_pdf=False, debug=False')
    VR.calibration.calibrate_wavelength(science_object, optimise_lc_solution=False, correct_barycentric_velocity=False, fit_voigt_for_rv=False, create_overview_pdf=False, debug=False)

    print('\n  --> Testing: calibrate_wavelength() with optimise_lc_solution=True, correct_barycentric_velocity=False, fit_voigt_for_rv=False, create_overview_pdf=False, debug=False')
    VR.calibration.calibrate_wavelength(science_object, optimise_lc_solution=True, correct_barycentric_velocity=False, fit_voigt_for_rv=False, create_overview_pdf=False, debug=False)

    print('\n  --> Testing: calibrate_wavelength() with optimise_lc_solution=True, correct_barycentric_velocity=True, fit_voigt_for_rv=True, create_overview_pdf=True, debug=True')
    VR.calibration.calibrate_wavelength(science_object, optimise_lc_solution=True, correct_barycentric_velocity=True, fit_voigt_for_rv=True, create_overview_pdf=True, debug=True)

    print('\n  --> DONE Testing: calibrate_wavelength()')

def test_fit_thxe_polynomial_coefficients():
    print('\n  --> Testing: fit_thxe_polynomial_coefficients()')

    VR.calibration.fit_thxe_polynomial_coefficients(debug=True)

    print('\n  --> DONE Testing: fit_thxe_polynomial_coefficients()')

def test_optimise_wavelength_solution_with_laser_comb():
    print('\n  --> Testing: optimise_wavelength_solution_with_laser_comb() to raise ValueError')

    with pytest.raises(ValueError) as excinfo:
        VR.calibration.optimise_wavelength_solution_with_laser_comb(order_name = 'not_a_valid_order', lc_pixel_values=[0,0])
    print(f'  --> ValueError raised: {excinfo.value}')

    print('\n  --> DONE Testing: optimise_wavelength_solution_with_laser_comb()')

# Run the test function
if __name__ == "__main__":
    
    print('\n  START Testing: VR.calibration.py')

    test_calibrate_wavelength()
    test_fit_thxe_polynomial_coefficients()
    test_optimise_wavelength_solution_with_laser_comb()

    print('\n  DONE Testing: VR.calibration.py')
