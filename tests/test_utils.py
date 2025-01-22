import numpy as np
import velocereduction as VR
from pathlib import Path
import pytest

def test_apply_velocity_shift_to_wavelength_array():
    print('\n  --> Testing: apply_velocity_shift_to_wavelength_array()')

    velocity_in_kms = 10.0
    wavelength_array = np.arange(5000, 6000, 1)
    shifted_wavelength = VR.utils.apply_velocity_shift_to_wavelength_array(velocity_in_kms, wavelength_array)
    print(f"  --> Wavelength after velocity shift of {velocity_in_kms} km/s: {shifted_wavelength[:3]} from {wavelength_array[:3]} (truncated at first 3 elements)")

    print('\n  --> DONE Testing: apply_velocity_shift_to_wavelength_array()')

def test_radial_velocity_from_line_shift():
    print('\n  --> Testing: radial_velocity_from_line_shift()')

    line_centre_observed = 6560.0
    line_centre_rest = 6562.7970
    vrad = VR.utils.radial_velocity_from_line_shift(line_centre_observed, line_centre_rest)
    print(f"  --> Radial Velocity: {vrad} km/s based on observed line centre at {line_centre_observed} Angstroms and rest line centre at {line_centre_rest} Angstroms.")

    print('\n  --> DONE Testing: radial_velocity_from_line_shift()')

def test_voigt_absorption_profile():
    print('\n  --> Testing: fit_voigt_absorption_profile() while using voigt_absorption_profile()')

    # Create a mock absorption profile
    wavelength = np.arange(5000, 6000, 1)
    flux = 1.0 - np.exp(-0.7 * (wavelength - 5500.0)**2 / 10.0**2)
    # line_centre, line_offset, line_depth, sigma, gamma
    initial_guess = [5450.0, 0.5, 0.5, 0.5, 0.5]

    for bounds in [None, ([5400.0, 0.0, 0.0, 0.0, 0.0], [5600.0, 1.0, 1.0, 1.0, 1.0])]:
        print(f"      Bounds: {bounds}")
        fit_parameters, fit_covariances = VR.utils.fit_voigt_absorption_profile(wavelength, flux, initial_guess, bounds=bounds)
        print(f"      Fit Parameters: {[format(value, '.3e') for value in fit_parameters]}")
        print(f"      Fit Covariances: {np.shape(fit_covariances)}")

    # Let's test using not the correct bounds
    with pytest.raises(ValueError) as excinfo:
        # Call the function with the mock FITS header
        print('  --> Testing with incorrect bounds -- should raise ValueError and continue testing.')
        fit_parameters, fit_covariances = VR.utils.fit_voigt_absorption_profile(wavelength, flux, initial_guess, bounds = [(10,10)])
    print(f'  --> ValueError raised: {excinfo.value}')

    # Let's test using not exactly 5 initial guess parameters
    with pytest.raises(ValueError) as excinfo:
        initial_guess = [5450.0, 0.5, 0.5, 0.5]
        # Call the function with the mock FITS header
        print('  --> Testing with not exactly 5 initial guess parameters -- should raise ValueError and continue testing.')
        fit_parameters, fit_covariances = VR.utils.fit_voigt_absorption_profile(wavelength, flux, initial_guess)
    print(f'  --> ValueError raised: {excinfo.value}')

    print('\n  --> DONE Testing: fit_voigt_absorption_profile() and voigt_absorption_profile()')

def test_lc_peak_gauss():
    print('\n  --> Testing: lc_peak_gauss()')

    pixels = np.arange(0, 10, 1)
    center = 5.0
    sigma = 2.0
    amplitude = 1.0
    offset = 0.0
    lc_peak = VR.utils.lc_peak_gauss(pixels, center, sigma, amplitude, offset)
    print(f"  --> Light Curve Peak for Gaussian with center at {center}, sigma of {sigma}, amplitude of {amplitude}, and offset of {offset}:")
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
        print(f"      Bounds: {bounds}")
        fit_parameters, fit_covariances = VR.utils.fit_gaussian_absorption_profile(wavelength, flux, initial_guess, bounds=bounds)
        print(f"      Fit Parameters: {[format(value, '.3e') for value in fit_parameters]}")
        print(f"      Fit Covariances: {np.shape(fit_covariances)}")
    
    # Let's test using not the correct bounds
    with pytest.raises(ValueError) as excinfo:
        # Call the function with the mock FITS header
        print('  --> Testing with incorrect bounds -- should raise ValueError and continue testing.')
        fit_parameters, fit_covariances = VR.utils.fit_gaussian_absorption_profile(wavelength, flux, initial_guess, bounds = [(10,10)])
    print(f'  --> ValueError raised: {excinfo.value}')

    # Let's test using not exactly 3 initial guess parameters
    with pytest.raises(ValueError) as excinfo:
        initial_guess = [5450.0, 0.5, 0.5, 0.5]
        # Call the function with the mock FITS header
        print('  --> Testing with not exactly 3 initial guess parameters -- should raise ValueError and continue testing.')
        fit_parameters, fit_covariances = VR.utils.fit_gaussian_absorption_profile(wavelength, flux, initial_guess)
    print(f'  --> ValueError raised: {excinfo.value}')

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
    barycentric_velocity_correction = VR.utils.calculate_barycentric_velocity_correction(fits_header_hip)
    # Print the barycentric velocity correction
    print(f"     Barycentric Velocity Correction: {barycentric_velocity_correction} km/s")

    print('\n  --> DONE Testing: calculate_barycentric_velocity_correction()')

def test_match_month_to_date():
    print('\n  --> Testing: match_month_to_date()')

    # Call the function with the mock FITS header
    for date in ['000122','000222','000322','000422','000522','000622','000722','000822','000922','001022','001122','001222']:
        month = VR.utils.match_month_to_date(date)
        # Print the month
        print(f"     Month for {date}: {month}")

    print('\n  --> DONE Testing: match_month_to_date()')
    
def test_polynomial_function():
    print('\n  --> Testing: polynomial_function()')

    # Call the function with mock data
    x = np.arange(0, 10, 1)
    coeffs = [320.0, 0.5, 0.1, 0.01, 0.001]
    y = VR.utils.polynomial_function(x, *coeffs)

    # Print the polynomial function
    print(f"  --> Polynomial Function: {coeffs[0]} + {coeffs[1]}*x + {coeffs[2]}*x^2 + {coeffs[3]}*x^3 + {coeffs[4]}*x^4")

    print('\n  --> DONE Testing: polynomial_function()')

def test_read_veloce_fits_image_and_metadata():
    print('\n  --> Testing: read_veloce_fits_image_and_metadata()')

    # Call the function with one of the provided FITS images
    image, metadata = VR.utils.read_veloce_fits_image_and_metadata(str(Path(__file__).resolve().parent)+'/../observations/001122/ccd_1/22nov10030.fits')
    # Print the image and header
    print(f"     Image shape: {np.shape(image)}")
    print(f"     Metadata: {metadata}")

    print('\n  --> DONE Testing: read_veloce_fits_image_and_metadata()')

def test_identify_calibration_and_science_runs():
    print('\n  --> Testing: identify_calibration_and_science_runs()')

    raw_data_dir = str(Path(__file__).resolve().parent)+'/../observations/'

    # Call the function with a non-existing date
    with pytest.raises(ValueError) as excinfo:
        date = '001322'
        VR.utils.identify_calibration_and_science_runs(date, raw_data_dir)
    print(f'  --> ValueError raised: {excinfo.value}')

    # Call the function with the provided raw data directory
    date = '001122'
    for each_science_run_separately in [False, True]:
        print(f"  --> Testin with each_science_run_separately: {each_science_run_separately}")
        VR.utils.identify_calibration_and_science_runs(date, raw_data_dir, each_science_run_separately)


    # Call the function with the provided raw data directory
    date = '001122'
    raw_data_dir = str(Path(__file__).resolve().parent)+'/../observations/'

    for each_science_run_separately in [False, True]:
        print(f"     Each Science Run Separately: {each_science_run_separately}")
        VR.utils.identify_calibration_and_science_runs(date, raw_data_dir, each_science_run_separately)

    print('\n  --> DONE Testing: identify_calibration_and_science_runs()')

def test_interpolate_spectrum():
    print('\n  --> Testing: interpolate_spectrum()')

    # Create a mock spectrum
    wavelength = np.arange(5000, 6000, 1)
    flux = 1.0 - np.exp(-0.5 * (wavelength - 5500.0)**2 / 100.0**2)
    target_wavelength = np.arange(5000, 6000, 0.5)

    # Call the function with the mock spectrum
    interpolated_flux = VR.utils.interpolate_spectrum(wavelength, flux, target_wavelength)
    # Print the interpolated flux
    print(f"     Interpolated flux with shape {np.shape(interpolated_flux)} from flux with shape {np.shape(flux)} onto target wavelength with shape {np.shape(target_wavelength)}.")

    print('\n  --> DONE Testing: interpolate_spectrum()')

def test_lasercomb_wavelength_from_numbers():
    print('\n  --> Testing: lasercomb_wavelength_from_numbers()')

    n = np.arange(18000, 19000, 250)
    repeat_frequency_ghz = 25.00000000
    offset_frequency_ghz = 9.56000000000
    wavelength = VR.utils.lasercomb_wavelength_from_numbers(n, repeat_frequency_ghz, offset_frequency_ghz)
    print(f"  --> n = {n}, repeat_frequency_ghz = {repeat_frequency_ghz}, offset_frequency_ghz = {offset_frequency_ghz} --> wavelength = {wavelength}")

    print('\n  --> DONE Testing: lasercomb_wavelength_from_numbers()')

def test_lasercomb_numbers_from_wavelength():
    print('\n  --> Testing: lasercomb_numbers_from_wavelength()')

    wavelength = np.arange(5000, 6000, 250)
    repeat_frequency_ghz = 25.00000000
    offset_frequency_ghz = 9.56000000000
    n = VR.utils.lasercomb_numbers_from_wavelength(wavelength, repeat_frequency_ghz, offset_frequency_ghz)
    print(f"     Wavelength = {wavelength}, repeat_frequency_ghz = {repeat_frequency_ghz}, offset_frequency_ghz = {offset_frequency_ghz} --> n = {n}")

    print('\n  --> DONE Testing: lasercomb_numbers_from_wavelength()')

def test_read_in_wavelength_solution_coefficients_tinney():
    print('\n  --> Testing: read_in_wavelength_solution_coefficients_tinney()')

    # Call the function with the provided wavelength solution coefficients
    coefficients = VR.utils.read_in_wavelength_solution_coefficients_tinney()
    # Print the coefficients
    print(f"  --> Found Tinney Wavelength Solution Coefficients for {len(coefficients)} orders.")
    expected_order = 'ccd_3_order_70'
    print(f"  --> First entry ({expected_order}): f{[format(value, '.3e') for value in coefficients[expected_order]]}")

    print('\n  --> DONE Testing: read_in_wavelength_solution_coefficients_tinney()')

def test_wavelength_vac_to_air():
    print('\n  --> Testing: wavelength_vac_to_air()')

    # Call the function with the provided wavelength solution coefficients
    for wavelength_vac in [6562.7970, np.arange(5000, 6000, 250)]:
        wavelength_air = VR.utils.wavelength_vac_to_air(wavelength_vac)
        # Print the coefficients
        print(f"  --> Vacuum Wavelength: {wavelength_vac} Angstroms --> Air Wavelength: {wavelength_air} Angstroms")

    print('\n  --> DONE Testing: wavelength_vac_to_air()')

def test_wavelength_air_to_vac():
    print('\n  --> Testing: wavelength_air_to_vac()')

    # Call the function with the provided wavelength solution coefficients
    for wavelength_air in [6563.2410, np.arange(5000, 6000, 250)]:
        wavelength_vac = VR.utils.wavelength_air_to_vac(wavelength_air)
        # Print the coefficients
        print(f"  --> Air Wavelength: {wavelength_air} Angstroms --> Vacuum Wavelength: {wavelength_vac} Angstroms")

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
        repeated_observations = VR.utils.check_repeated_observations(science_runs)
        print(f"  --> Repeated Observations: {repeated_observations}")

    print('\n  --> DONE Testing: check_repeated_observations()')

def test_monitor_vrad_for_repeat_observations():
    print('\n  --> Testing: monitor_vrad_for_repeat_observations()')

    date = '001122'
    repeated_observations = {'HIP69673': ['0150', '0151']}

    print('\n  --> Testing with repeated observations')
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'
    VR.utils.monitor_vrad_for_repeat_observations(date, repeated_observations)

    print('\n  --> Testing without repeated observations')
    repeated_observations = {}
    VR.utils.monitor_vrad_for_repeat_observations(date, repeated_observations)

    print('\n  --> DONE Testing: monitor_vrad_for_repeat_observations()')

def test_get_memory_usage():
    print('\n  --> Testing: get_memory_usage()')

    memory_usage = VR.utils.get_memory_usage()
    print(f"  --> Memory Usage: {memory_usage}")

    print('\n  --> DONE Testing: get_memory_usage()')

def test_degrade_spectral_resolution():
    print('\n  --> Testing: degrade_spectral_resolution()')

    # Create a mock spectrum
    wavelength = np.arange(5400, 5600, 1)
    flux = 1.0 - np.exp(-0.5 * (wavelength - 5500.0)**2 / 10.0**2)
    original_resolution = 300000.
    target_resolution = 80000.

    # Call the function with the provided wavelength solution coefficients
    degraded_flux = VR.utils.degrade_spectral_resolution(wavelength, flux, original_resolution, target_resolution)
    # Print the coefficients
    high_res_not_one = np.where(flux != 1.0)[0]
    low_res_not_one = np.where(degraded_flux != 1.0)[0]
    print(f"  --> Original Resolution: {original_resolution} --> Target Resolution: {target_resolution}")
    print(f"  --> Flux != 1.0 at {len(high_res_not_one)} indices in high resolution spectrum.")
    print(f"  --> Flux != 1.0 at {len(low_res_not_one)} indices in low resolution spectrum.")

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

    fits_header_18Sco = {
        'OBJECT': '18 Sco',
        'MEANRA': 243.905279,  # Right Ascension in decimal degrees
        'MEANDEC': -8.37164116155, # Declination in decimal degrees
        'UTMJD': 60359.7838614119 # Modified Julian Date at start of exposure
    }

    fits_header_other = {
        'OBJECT': '23_LZ_Gmag8',
        'MEANRA': 180.2385559,  # Right Ascension in decimal degrees
        'MEANDEC': -08.369341, # Declination in decimal degrees
        'UTMJD': 60359.7838614119 # Modified Julian Date at start of exposure
    }

    fits_header_18sco_with_different_name= {
        'OBJECT': 'Fake Star 18 Sco',
        'MEANRA': 243.905279,  # Right Ascension in decimal degrees
        'MEANDEC': -08.369341, # Declination in decimal degrees
        'UTMJD': 60359.7838614119 # Modified Julian Date at start of exposure
    }

    fits_header_fake_star = {
        'OBJECT': 'Fake Star',
        'MEANRA': 0.0,  # Right Ascension in decimal degrees
        'MEANDEC': 0.0, # Declination in decimal degrees
        'UTMJD': 60359.7838614119 # Modified Julian Date at start of exposure
    }

    for fits_header in [fits_header_hip, fits_header_gaia, fits_header_18Sco, fits_header_other, fits_header_18sco_with_different_name, fits_header_fake_star]:
        # Call the function with the mock FITS header
        updated_header = VR.utils.update_fits_header_via_crossmatch_with_simbad(fits_header)
        # Print the updated header to see the changes
        print("  --> Updated FITS Header for OBJECT "+fits_header['OBJECT']+":")
        for key, value in updated_header.items():
            print(f"      {key}: {value}")

    print('\n  --> DONE Testing: update_fits_header_via_crossmatch_with_simbad()')


def test_find_best_radial_velocity_from_fits_header():
    print('\n  --> Testing: find_best_radial_velocity_from_fits_header()')

    # Mock FITS headers for testing
    fits_header_vrad_both_veloce_better = {
        'VRAD': 234.0,
        'E_VRAD': 1.0,
        'VRAD_LIT': 233.0,
        'E_VRAD_LIT': 2.0,
        'VRAD_BIB': '2018A&A...616A...7S'
    }
    fits_header_vrad_both_veloce_worse = {
        'VRAD': 234.0,
        'E_VRAD': 1.0,
        'VRAD_LIT': 233.0,
        'E_VRAD_LIT': 0.1,
        'VRAD_BIB': '2018A&A...616A...7S'
    }
    fits_header_vrad_veloce_only = {
        'VRAD': 234.0,
        'E_VRAD': 1.0,
    }
    fits_header_vrad_literature_only = {
        'VRAD': 'None',
        'E_VRAD': 'None',
        'VRAD_LIT': 233.0,
        'E_VRAD_LIT': 0.1,
        'VRAD_BIB': '2018A&A...616A...7S'
    }

    for fits_header in [fits_header_vrad_both_veloce_better, fits_header_vrad_both_veloce_worse, fits_header_vrad_veloce_only, fits_header_vrad_literature_only]:
        # Call the function with the mock FITS header
        for key, value in fits_header.items():
            print(f"      {key}: {value}")
        best_vrad = VR.utils.find_best_radial_velocity_from_fits_header(fits_header)
        # Print the best radial velocity
        print(f"  --> Best Radial Velocity: {best_vrad} km/s"+'\n')

    # Now let's test a case that will raise a ValueError
    fits_header_vrad_none = {
        'VRAD': 'None',
        'E_VRAD': 'None',
    }
    expected_msg = 'No valid option for VRAD avaialble. Aborting calibration via synthesis.'

    for key, value in fits_header.items():
            print(f"      {key}: {value}")

    with pytest.raises(ValueError) as excinfo:
        best_vrad = VR.utils.find_best_radial_velocity_from_fits_header(fits_header_vrad_none)
    print(f'  --> ValueError raised: {excinfo.value}')
    
    print('\n  --> DONE Testing: find_best_radial_velocity_from_fits_header()')

def test_find_closest_korg_spectrum():
    print('\n  --> Testing: find_closest_korg_spectrum()')

    korg_spectra = VR.flux_comparison.read_available_korg_syntheses()

    # Mock FITS headers for testing
    fits_header_18sco = {
        'OBJECT': 'HIP79672'
    }
    fits_header_cool_giant = {
        'OBJECT': 'Cool Giant',
        'TEFF_LIT': 4500.0,
        'LOGG_LIT': 2.5,
        'FE_H_LIT': -0.5
    }
    fits_header_cool_dwarf = {
        'OBJECT': 'Cool Dwarf',
        'TEFF_LIT': 4500.0,
        'LOGG_LIT': 4.5,
        'FE_H_LIT': -0.5
    }
    fits_header_metal_poor_dwarf = {
        'OBJECT': 'Metal Poor Dwarf',
        'TEFF_LIT': 5500.0,
        'LOGG_LIT': 4.5,
        'FE_H_LIT': -0.5
    }
    fits_header_solar_dwarf = {
        'OBJECT': 'Solar Dwarf',
        'TEFF_LIT': 5500.0,
        'LOGG_LIT': 4.5,
        'FE_H_LIT': 0.0
    }
    fits_header_only_low_fe_h = {
        'OBJECT': 'Only [Fe/H], metal poor',
        'FE_H_LIT': -0.5
    }
    fits_header_only_solar_fe_h = {
        'OBJECT': 'Only [Fe/H], Solar',
        'FE_H_LIT': 0.0
    }
    fits_header_only_plx_bgr_cool_dwarf = {
        'OBJECT': 'Only CMD BGR cool dwarf',
        'PLX': 10.0,
        'B': 12.0,
        'G': 10.0,
        'R': 10.0
    }
    fits_header_only_plx_bgr_warm_dwarf = {
        'OBJECT': 'Only CMD BGR warm dwarf',
        'PLX': 10.0,
        'B': 10.0,
        'G': 10.0,
        'R': 10.0
    }
    fits_header_only_plx_vr_cool_dwarf = {
        'OBJECT': 'Only CMD VR cool dwarf',
        'PLX': 10.0,
        'V': 11.0,
        'R': 10.0
    }
    fits_header_only_plx_vr_warm_dwarf = {
        'OBJECT': 'Only CMD VR warm dwarf',
        'PLX': 10.0,
        'V': 10.0,
        'R': 10.0
    }
    fits_header_only_plx_bgr_cool_giant = {
        'OBJECT': 'Only CMD, cool giant',
        'PLX': 10.0,
        'B': 7.0,
        'G': 5.0,
        'R': 5.0
    }
    fits_header_negative_parallax = {
        'OBJECT': 'None parallax',
        'PLX': -0.1,
    }
    fits_header_none_parallax = {
        'OBJECT': 'None parallax',
        'PLX': 'None',
    }
    fits_header_not_even_parallax = {
        'OBJECT': 'Not even parallax'
    }

    # Let's loop through all cases
    for fits_header in [
        fits_header_18sco, fits_header_cool_giant, fits_header_cool_dwarf, fits_header_metal_poor_dwarf,
        fits_header_solar_dwarf, fits_header_only_low_fe_h, fits_header_only_solar_fe_h, fits_header_only_plx_bgr_cool_dwarf,
        fits_header_only_plx_bgr_warm_dwarf, fits_header_only_plx_vr_cool_dwarf, fits_header_only_plx_vr_warm_dwarf,
        fits_header_only_plx_bgr_cool_giant, fits_header_negative_parallax, fits_header_none_parallax, fits_header_not_even_parallax
    ]:
        print('  --> Testing '+fits_header['OBJECT'])
        closest_korg_spectrum = VR.utils.find_closest_korg_spectrum(korg_spectra,fits_header)
        print(f"  --> Closest Korg Spectrum for '{fits_header['OBJECT']}': {closest_korg_spectrum}")

    print('\n  --> DONE Testing: find_closest_korg_spectrum()')

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

    test_find_best_radial_velocity_from_fits_header()

    test_find_closest_korg_spectrum()

    print('\n  DONE Testing: VR.utils.py')