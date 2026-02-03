import velocereduction as VR
from pathlib import Path
import pytest
from astropy.io import fits

def test_calculate_wavelength_coefficients_with_korg_synthesis():
    print('\n  --> Testing: calculate_wavelength_coefficients_with_korg_synthesis()')

    VR.config.date = '001122'
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'
    science_object = 'HIP69673'

    korg_spectra = VR.flux_comparison.read_available_korg_syntheses()

    # We do not update, but just open the FITS file.
    # with fits.open(VR.config.working_directory+'reduced_data/'+VR.config.date+'/'+science_object+'/veloce_spectra_'+science_object+'_'+VR.config.date+'.fits', mode='update') as veloce_fits_file:
    with fits.open(VR.config.working_directory+'reduced_data/'+VR.config.date+'/'+science_object+'/veloce_spectra_'+science_object+'_'+VR.config.date+'.fits', mode='update') as veloce_fits_file:

        # Find the closest match based on (possibly available) literature TEFF/LOGG/FE_H
        closest_korg_spectrum = VR.utils.find_closest_korg_spectrum(
            available_korg_spectra = korg_spectra,
            fits_header = veloce_fits_file[0].header,
        )

        # Find the best RV or raise ValueError of none available.
        vrad_for_calibration = VR.utils.find_best_radial_velocity_from_fits_header(fits_header = veloce_fits_file[0].header)

        print('  --> Testing order ccd_3_order_71 with telluric form Bstar and debug=True')
        # Let's test this for one order with bstar
        for order in ['ccd_3_order_71']:
        
            VR.flux_comparison.calculate_wavelength_coefficients_with_korg_synthesis(
                veloce_fits_file,
                korg_wavelength_vac = korg_spectra['wavelength_vac'],
                korg_flux = korg_spectra['flux_'+closest_korg_spectrum],
                vrad_for_calibration = vrad_for_calibration,
                order_selection=[order],
                telluric_hinkle_or_bstar = 'bstar', # You can choose between 'hinkle' and 'bstar'
                debug=True
            )
    
        print('  --> Testing: ValueError raised if order does not have valid Korg flux, e.g. when the wavelength is not covered by the Korg spectra')
        with pytest.raises(ValueError) as excinfo:
            # Let's test this for all orders with hinkle
            VR.flux_comparison.calculate_wavelength_coefficients_with_korg_synthesis(
                veloce_fits_file,
                korg_wavelength_vac = korg_spectra['wavelength_vac']/10.,
                korg_flux = korg_spectra['flux_'+closest_korg_spectrum],
                vrad_for_calibration = vrad_for_calibration,
                order_selection=['ccd_1_order_167'],
                telluric_hinkle_or_bstar = 'hinkle', # You can choose between 'hinkle' and 'bstar'
                debug=False
            )
        print(f'  --> ValueError raised as expected: {excinfo.value}')

        print('  --> Testing: ValueError raised if order is not within our expected Veloce orders.')
        with pytest.raises(ValueError) as excinfo:
            # Let's test this for all orders with hinkle
            VR.flux_comparison.calculate_wavelength_coefficients_with_korg_synthesis(
                veloce_fits_file,
                korg_wavelength_vac = korg_spectra['wavelength_vac']/10.,
                korg_flux = korg_spectra['flux_'+closest_korg_spectrum],
                vrad_for_calibration = vrad_for_calibration,
                order_selection=['ccd_1_order_168'],
                telluric_hinkle_or_bstar = 'hinkle', # You can choose between 'hinkle' and 'bstar'
                debug=False
            )
        print(f'  --> ValueError raised as expected: {excinfo.value}')

    print('\n  --> DONE Testing: calculate_wavelength_coefficients_with_korg_synthesis()')
    
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
    test_calculate_wavelength_coefficients_with_korg_synthesis()

    test_find_closest_korg_spectrum()

