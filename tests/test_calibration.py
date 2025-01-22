import velocereduction as VR
from pathlib import Path
import pytest

# def test_optimise_wavelength_solution_with_laser_comb():
#     print('\n  --> Testing: optimise_wavelength_solution_with_laser_comb()')

#     VR.calibration.optimise_wavelength_solution_with_laser_comb()

#     print('\n  --> DONE Testing: optimise_wavelength_solution_with_laser_comb()')

# def test_calibrate_single_order():
#     print('\n  --> Testing: calibrate_single_order()')

#     VR.calibration.calibrate_single_order()

#     print('\n  --> DONE Testing: calibrate_single_order()')

# def test_plot_wavelength_calibrated_order_data():
#     print('\n  --> Testing: plot_wavelength_calibrated_order_data()')

#     VR.calibration.plot_wavelength_calibrated_order_data()

#     print('\n  --> DONE Testing: plot_wavelength_calibrated_order_data()')

def test_calibrate_wavelength():
    print('\n  --> Testing: calibrate_wavelength()')

    VR.config.date = '001122'
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'
    science_object = 'HIP69673'

    print('\n  --> Testing: calibrate_wavelength() with optimise_lc_solution=False, correct_barycentric_velocity=True, fit_voigt_for_rv=False, create_overview_pdf=False')
    VR.calibration.calibrate_wavelength(science_object, optimise_lc_solution=False, correct_barycentric_velocity=True, fit_voigt_for_rv=False, create_overview_pdf=False)

    print('\n  --> Testing: calibrate_wavelength() with optimise_lc_solution=True, correct_barycentric_velocity=True, fit_voigt_for_rv=True, create_overview_pdf=True')
    VR.calibration.calibrate_wavelength(science_object, optimise_lc_solution=True, correct_barycentric_velocity=True, fit_voigt_for_rv=True, create_overview_pdf=True)

    print('\n  --> DONE Testing: calibrate_wavelength()')

def test_fit_thxe_polynomial_coefficients():
    print('\n  --> Testing: fit_thxe_polynomial_coefficients()')

    VR.calibration.fit_thxe_polynomial_coefficients(debug=True)

    print('\n  --> DONE Testing: fit_thxe_polynomial_coefficients()')

# Run the test function
if __name__ == "__main__":

    # test_optimise_wavelength_solution_with_laser_comb()

    # test_calibrate_single_order()

    # test_plot_wavelength_calibrated_order_data()

    test_calibrate_wavelength()

    # test_fit_thxe_polynomial_coefficients()