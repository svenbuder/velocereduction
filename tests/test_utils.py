import numpy as np
from velocereduction import utils

def test_apply_velocity_shift_to_wavelength_array():
    print('\n  --> Testing: apply_velocity_shift_to_wavelength_array()')

    velocity_in_kms = 10.0
    wavelength_array = np.arange(5000, 6000, 1)
    shifted_wavelength = utils.apply_velocity_shift_to_wavelength_array(velocity_in_kms, wavelength_array)
    print(f"wavelength after velocity shift of {velocity_in_kms} km/s: {shifted_wavelength[:3]} from {wavelength_array[:3]} (truncated at first 3 elements)")

    print('\n  --> DONE Testing: apply_velocity_shift_to_wavelength_array()')

def test_radial_velocity_from_line_shift():
    print('\n  --> Testing: radial_velocity_from_line_shift()')

    line_centre_observed = 6560.0
    line_centre_rest = 6562.7970
    vrad = utils.radial_velocity_from_line_shift(line_centre_observed, line_centre_rest)
    print(f"Radial Velocity: {vrad} km/s based on observed line centre at {line_centre_observed} Angstroms and rest line centre at {line_centre_rest} Angstroms.")

    print('\n  --> DONE Testing: radial_velocity_from_line_shift()')

def test_voigt_absorption_profile():
    print('\n  --> Testing: fit_voigt_absorption_profile() while using voigt_absorption_profile()')

    # Create a mock absorption profile
    wavelength = np.arange(5000, 6000, 1)
    flux = 1.0 - np.exp(-0.7 * (wavelength - 5500.0)**2 / 10.0**2)
    # line_centre, line_offset, line_depth, sigma, gamma
    initial_guess = [5450.0, 0.5, 0.5, 0.5, 0.5]

    for bounds in [None, ([5400.0, 0.0, 0.0, 0.0, 0.0], [5600.0, 1.0, 1.0, 1.0, 1.0])]:
        print(f"Bounds: {bounds}")
        fit_parameters, fit_covariances = utils.fit_voigt_absorption_profile(wavelength, flux, initial_guess, bounds=bounds)
        print(f"Fit Parameters: {fit_parameters}")
        print(f"Fit Covariances: {fit_covariances}")

    print('\n  --> DONE Testing: fit_voigt_absorption_profile() and voigt_absorption_profile()')

def test_lc_peak_gauss():
    print('\n  --> Testing: lc_peak_gauss()')

    pixels = np.arange(0, 10, 1)
    center = 5.0
    sigma = 2.0
    amplitude = 1.0
    offset = 0.0
    lc_peak = utils.lc_peak_gauss(pixels, center, sigma, amplitude, offset)
    print(f"Light Curve Peak for Gaussian with center at {center}, sigma of {sigma}, amplitude of {amplitude}, and offset of {offset}:")
    print(lc_peak)

    print('\n  --> DONE Testing: lc_peak_gauss()')

def test_gaussian_absorption_profile():
    print('\n  --> Testing: fit_gaussian_absorption_profile() while using gaussian_absorption_profile()')

    # Create a mock absorption profile
    wavelength = np.arange(5000, 6000, 1)
    flux = 1.0 - np.exp(-0.7 * (wavelength - 5500.0)**2 / 10.0**2)
    # line_centre, line_depth, line_sigma
    initial_guess = [5500.0, 0.5, 0.5]

    for bounds in [None, ([5400.0, 0.0, 0.0], [5600.0, 1.0, 1.0])]:
        print(f"Bounds: {bounds}")
        fit_parameters, fit_covariances = utils.fit_gaussian_absorption_profile(wavelength, flux, initial_guess, bounds=bounds)
        print(f"Fit Parameters: {fit_parameters}")
        print(f"Fit Covariances: {fit_covariances}")
    
    print('\n  --> DONE Testing: fit_gaussian_absorption_profile() and gaussian_absorption_profile()')
    
def test_calculate_barycentric_velocity_correction():
    print('\n  --> Testing: calculate_barycentric_velocity_correction()')

    # Mock FITS headers for testing
    fits_header_hip = {
        'OBJECT': 'HIP69673',
        'MEANRA': 213.907739365913,  # Right Ascension in decimal degrees
        'MEANDEC': 19.1682209854537, # Declination in decimal degrees
        'UTMJD': 60359.7838614119    # Modified Julian Date at start of exposure
    }

    # Call the function with the mock FITS header
    barycentric_velocity_correction = utils.calculate_barycentric_velocity_correction(fits_header_hip)
    # Print the barycentric velocity correction
    print(f"Barycentric Velocity Correction: {barycentric_velocity_correction} km/s")

    print('\n  --> DONE Testing: calculate_barycentric_velocity_correction()')

def test_match_month_to_date():
    print('\n  --> Testing: match_month_to_date()')

    # Call the function with the mock FITS header
    for date in ['000122','000222','000322','000422','000522','000622','000722','000822','000922','001022','001122','001222']:
        month = utils.match_month_to_date(date)
        # Print the month
        print(f"Month for {date}: {month}")

    print('\n  --> DONE Testing: match_month_to_date()')
    
def test_polynomial_function():
    print('\n  --> Testing: polynomial_function()')

    # Call the function with mock data
    x = np.arange(0, 10, 1)
    coeffs = [320.0, 0.5, 0.1, 0.01, 0.001]
    y = utils.polynomial_function(x, *coeffs)

    # Print the polynomial function
    print(f"Polynomial Function: {coeffs[0]} + {coeffs[1]}*x + {coeffs[2]}*x^2 + {coeffs[3]}*x^3 + {coeffs[4]}*x^4")

    print('\n  --> DONE Testing: polynomial_function()')

def test_read_veloce_fits_image_and_metadata():
    print('\n  --> Testing: read_veloce_fits_image_and_metadata()')

    # Call the function with one of the provided FITS images
    image, metadata = utils.read_veloce_fits_image_and_metadata('observations/001122/ccd_1/22nov10030.fits')
    # Print the image and header
    print(f"Image shape: {np.shape(image)}")
    print(f"Metadata: {metadata}")

    print('\n  --> DONE Testing: read_veloce_fits_image_and_metadata()')

def test_identify_calibration_and_science_runs():
    print('\n  --> Testing: identify_calibration_and_science_runs()')

    # Call the function with the provided raw data directory
    date = '001122'
    raw_data_dir = 'observations/'

    for each_science_run_separately in [False, True]:
        print(f"Each Science Run Separately: {each_science_run_separately}")
        utils.identify_calibration_and_science_runs(date, raw_data_dir, each_science_run_separately)

    print('\n  --> DONE Testing: identify_calibration_and_science_runs()')

def test_interpolate_spectrum():
    print('\n  --> Testing: interpolate_spectrum()')

    # Create a mock spectrum
    wavelength = np.arange(5000, 6000, 1)
    flux = 1.0 - np.exp(-0.5 * (wavelength - 5500.0)**2 / 100.0**2)
    target_wavelength = np.arange(5000, 6000, 0.5)

    # Call the function with the mock spectrum
    interpolated_flux = utils.interpolate_spectrum(wavelength, flux, target_wavelength)
    # Print the interpolated flux
    print(f"Interpolated flux with shape {np.shape(interpolated_flux)} from flux with shape {np.shape(flux)} onto target wavelength with shape {np.shape(target_wavelength)}.")

    print('\n  --> DONE Testing: interpolate_spectrum()')

def test_lasercomb_wavelength_from_numbers():
    print('\n  --> Testing: lasercomb_wavelength_from_numbers()')

    n = np.arange(18000, 19000, 100)
    repeat_frequency_ghz = 25.00000000
    offset_frequency_ghz = 9.56000000000
    wavelength = utils.lasercomb_wavelength_from_numbers(n, repeat_frequency_ghz, offset_frequency_ghz)
    print(f"n = {n}, repeat_frequency_ghz = {repeat_frequency_ghz}, offset_frequency_ghz = {offset_frequency_ghz} --> wavelength = {wavelength}")

    print('\n  --> DONE Testing: lasercomb_wavelength_from_numbers()')

def test_lasercomb_numbers_from_wavelength():
    print('\n  --> Testing: lasercomb_numbers_from_wavelength()')

    wavelength = np.arange(5000, 6000, 100)
    repeat_frequency_ghz = 25.00000000
    offset_frequency_ghz = 9.56000000000
    n = utils.lasercomb_numbers_from_wavelength(wavelength, repeat_frequency_ghz, offset_frequency_ghz)
    print(f"wavelength = {wavelength}, repeat_frequency_ghz = {repeat_frequency_ghz}, offset_frequency_ghz = {offset_frequency_ghz} --> n = {n}")

    print('\n  --> DONE Testing: lasercomb_numbers_from_wavelength()')

def test_read_in_wavelength_solution_coefficients_tinney():
    print('\n  --> Testing: read_in_wavelength_solution_coefficients_tinney()')

    # Call the function with the provided wavelength solution coefficients
    coefficients = utils.read_in_wavelength_solution_coefficients_tinney()
    # Print the coefficients
    print(f"Found Tinney Wavelength Solution Coefficients for {len(coefficients)} orders.")
    expected_order = 'ccd_3_order_70'
    print(f"First entry ({expected_order}): f{coefficients[expected_order]}")

    print('\n  --> DONE Testing: read_in_wavelength_solution_coefficients_tinney()')

def test_wavelength_vac_to_air():
    print('\n  --> Testing: wavelength_vac_to_air()')

    # Call the function with the provided wavelength solution coefficients
    for wavelength_vac in [6562.7970, np.arange(5000, 6000, 250)]:
        wavelength_air = utils.wavelength_vac_to_air(wavelength_vac)
        # Print the coefficients
        print(f"Vacuum Wavelength: {wavelength_vac} Angstroms --> Air Wavelength: {wavelength_air} Angstroms")

    print('\n  --> DONE Testing: wavelength_vac_to_air()')

def test_wavelength_air_to_vac():
    print('\n  --> Testing: wavelength_air_to_vac()')

    # Call the function with the provided wavelength solution coefficients
    for wavelength_air in [6563.2410, np.arange(5000, 6000, 250)]:
        wavelength_vac = utils.wavelength_air_to_vac(wavelength_air)
        # Print the coefficients
        print(f"Air Wavelength: {wavelength_air} Angstroms --> Vacuum Wavelength: {wavelength_vac} Angstroms")

    print('\n  --> DONE Testing: wavelength_air_to_vac()')

def test_check_repeated_observations():
    print('\n  --> Testing: check_repeated_observations()')

    science_runs_with_repeated_observations = {
        'HIP1234': ['0001','0002'],
        'HIP69673_0150': ['0150'],
        'HIP69673_0151': ['0151'],
    }
    science_runs_witout_repeated_observations = {
        'HIP69673': ['0150','0151']
    }

    # Run function with and without repeat observations
    for science_runs in [science_runs_with_repeated_observations,science_runs_witout_repeated_observations]:
        repeated_observations = utils.check_repeated_observations(science_runs)
        print(f"Repeated Observations: {repeated_observations}")

    print('\n  --> DONE Testing: check_repeated_observations()')

def test_monitor_vrad_for_repeat_observations():
    print('\n  --> Testing: monitor_vrad_for_repeat_observations()')

    date = '001122'
    repeated_observations = {'HIP69673': ['0150', '0151']}

    utils.monitor_vrad_for_repeat_observations(date, repeated_observations)

    print('\n  --> DONE Testing: monitor_vrad_for_repeat_observations()')

def test_get_memory_usage():
    print('\n  --> Testing: get_memory_usage()')

    memory_usage = utils.get_memory_usage()
    print(f"Memory Usage: {memory_usage}")

    print('\n  --> DONE Testing: get_memory_usage()')

def test_degrade_spectral_resolution():
    print('\n  --> Testing: degrade_spectral_resolution()')

    # Create a mock spectrum
    wavelength = np.arange(5400, 5600, 1)
    flux = 1.0 - np.exp(-0.5 * (wavelength - 5500.0)**2 / 10.0**2)
    original_resolution = 300000.
    target_resolution = 80000.

    # Call the function with the provided wavelength solution coefficients
    degraded_flux = utils.degrade_spectral_resolution(wavelength, flux, original_resolution, target_resolution)
    # Print the coefficients
    high_res_not_one = np.where(flux != 1.0)[0]
    low_res_not_one = np.where(degraded_flux != 1.0)[0]
    print(f"Original Resolution: {original_resolution} --> Target Resolution: {target_resolution}")
    print(f"Flux != 1.0 at {len(high_res_not_one)} indices in high resolution spectrum.")
    print(f"Flux != 1.0 at {len(low_res_not_one)} indices in low resolution spectrum.")

    print('\n  --> DONE Testing: degrade_spectral_resolution()')

def test_update_fits_header_via_crossmatch_with_simbad():
    print('\n  --> Testing: update_fits_header_via_crossmatch_with_simbad()')

    # Mock FITS headers for testing
    fits_header_hip = {
        'OBJECT': 'HIP69673',
        'MEANRA': 213.907739365913,  # Right Ascension in decimal degrees
        'MEANDEC': 19.1682209854537, # Declination in decimal degrees
        'UTMJD': 60359.7838614119    # Modified Julian Date at start of exposure
    }

    fits_header_gaia = {
        'OBJECT': '2349658070839982976',
        'MEANRA': 13.16940534,  # Right Ascension in decimal degrees
        'MEANDEC': -21.43891317, # Declination in decimal degrees
        'UTMJD': 60359.7838614119    # Modified Julian Date at start of exposure
    }

    fits_header_other = {
        'OBJECT': '23_LZ_Gmag8',
        'MEANRA': 180.2385559,  # Right Ascension in decimal degrees
        'MEANDEC': 19.1682209854537, # Declination in decimal degrees
        'UTMJD': -21.25097632    # Modified Julian Date at start of exposure
    }

    for fits_header in [fits_header_hip, fits_header_gaia, fits_header_other]:
        # Call the function with the mock FITS header
        updated_header = utils.update_fits_header_via_crossmatch_with_simbad(fits_header)
        # Print the updated header to see the changes
        print("Updated FITS Header for OBJECT "+fits_header['OBJECT']+":")
        for key, value in updated_header.items():
            print(f"{key}: {value}")

    print('\n  --> DONE Testing: update_fits_header_via_crossmatch_with_simbad()')

# Run the test function
if __name__ == "__main__":

    test_apply_velocity_shift_to_wavelength_array()
    
    test_radial_velocity_from_line_shift()
    
    test_voigt_absorption_profile()
    
    test_lc_peak_gauss()
    
    test_gaussian_absorption_profile()

    test_calculate_barycentric_velocity_correction()

    test_match_month_to_date()

    test_polynomial_function()

    test_read_veloce_fits_image_and_metadata()

    test_identify_calibration_and_science_runs()

    test_interpolate_spectrum()

    test_lasercomb_wavelength_from_numbers()

    test_lasercomb_numbers_from_wavelength()

    test_read_in_wavelength_solution_coefficients_tinney()

    test_wavelength_vac_to_air()

    test_wavelength_air_to_vac()

    test_check_repeated_observations()

    test_monitor_vrad_for_repeat_observations()

    test_get_memory_usage()

    test_degrade_spectral_resolution()

    test_update_fits_header_via_crossmatch_with_simbad()

    print('\n  DONE Testing: utils.py')