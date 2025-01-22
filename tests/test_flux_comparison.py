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

        # Let's test this for a few orders (or simply set order_selection=None to use all)
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

            VR.flux_comparison.calculate_wavelength_coefficients_with_korg_synthesis(
                veloce_fits_file,
                korg_wavelength_vac = korg_spectra['wavelength_vac'],
                korg_flux = korg_spectra['flux_'+closest_korg_spectrum],
                vrad_for_calibration = vrad_for_calibration,
                order_selection=[order],
                telluric_hinkle_or_bstar = 'hinkle', # You can choose between 'hinkle' and 'bstar'
                debug=True
            )

    print('\n  --> DONE Testing: calculate_wavelength_coefficients_with_korg_synthesis()')
    
# Run the test function
if __name__ == "__main__":
    test_calculate_wavelength_coefficients_with_korg_synthesis()