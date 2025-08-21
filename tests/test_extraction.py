import velocereduction as VR
from pathlib import Path
import pytest

def test_substract_overscan():
    print('\n  --> Testing: substract_overscan()')

    # Load the image to test
    test_file = str(Path(__file__).resolve().parent)+'/../observations/001122/ccd_1/22nov10030.fits'

    full_image, metadata = VR.utils.read_veloce_fits_image_and_metadata(test_file)

    # Test the function without debug
    print('      --> with debug_overscan=False')
    trimmed_image, _, _, _ = VR.extraction.substract_overscan(full_image, metadata)

    # Test the debug function
    print('      --> with debug_overscan=True')
    trimmed_image, _, _, _ = VR.extraction.substract_overscan(full_image, metadata, debug_overscan= True)

    # Load the image to test
    print('      --> with 4Amp')
    test_file = str(Path(__file__).resolve().parent)+'/../observations/001122/ccd_1/4amp_example.fits'
    full_image, metadata = VR.utils.read_veloce_fits_image_and_metadata(test_file)
    trimmed_image, _, _, _ = VR.extraction.substract_overscan(full_image, metadata)

    print('\n  --> DONE Testing: substract_overscan()')

def test_read_in_order_tramlines_tinney():
    print('\n  --> Testing: read_in_order_tramlines_tinney()')

    tinney_tramlines = VR.extraction.read_in_order_tramlines_tinney()

    print('\n  --> DONE Testing: read_in_order_tramlines_tinney()')

def test_read_in_order_tramlines():
    print('\n  --> Testing: read_in_order_tramlines()')

    test_order = 'ccd_3_order_70'
    print('  --> Testing with order '+test_order)

    # Test the function
    print('      with use_default=False')
    order_tramline_ranges, order_tramline_beginning_coefficients, order_tramline_ending_coefficients = VR.extraction.read_in_order_tramlines()
    print(f'  --> tramline ranges test entry: {order_tramline_ranges[test_order]}')
    print(f'  --> tramline beginning coefficients test entry: {[format(value, ".3e") for value in order_tramline_beginning_coefficients[test_order]]}')
    print(f'  --> tramline ending coefficients test entry: {[format(value, ".3e") for value in order_tramline_ending_coefficients[test_order]]}')

    print('      with use_default=True')
    order_tramline_ranges, order_tramline_beginning_coefficients, order_tramline_ending_coefficients = VR.extraction.read_in_order_tramlines(use_default=True)
    print(f'  --> tramline ranges test entry: {order_tramline_ranges[test_order]}')
    print(f'  --> tramline beginning coefficients test entry: {[format(value, ".3e") for value in order_tramline_beginning_coefficients[test_order]]}')
    print(f'  --> tramline ending coefficients test entry: {[format(value, ".3e") for value in order_tramline_ending_coefficients[test_order]]}')

    print('\n  --> DONE Testing: read_in_order_tramlines()')

def test_get_master_dark():
    print('\n  --> Testing: get_master_dark()')

    VR.config.date = '001122'
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'

    runs = ['0224']

    print('      with archival=False')
    master_dark = VR.extraction.get_master_dark(runs)
    print(f'  --> master dark test entry: {master_dark.keys()}')

    print('      with archival=True')
    master_dark = VR.extraction.get_master_dark(runs, archival=True)
    print(f'  --> master dark test entry: {master_dark.keys()}')

    print('\n  --> DONE Testing: get_master_dark()')

def test_get_tellurics_from_bstar():
    print('\n  --> Testing: get_tellurics_from_bstar() with debug=True')

    VR.config.date = '001122'
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'

    calibration_runs = {
        'Flat_60.0': ['0030'],
        'Flat_1.0': ['0016'],
        'Flat_0.1': ['0009'],
        'Bstar': {'18:57:01': ['127972', '0154', '18:57:01']}
    }

    master_flat, master_flat_images = VR.extraction.extract_orders(
        ccd1_runs = calibration_runs['Flat_60.0'],
        ccd2_runs = calibration_runs['Flat_1.0'],
        ccd3_runs = calibration_runs['Flat_0.1'],
        Flat = True,
        update_tramlines_based_on_flat = False,
        debug_overscan = False,
        debug_rows = False,
        debug_tramlines = False
    )

    for bstar_exposure in calibration_runs['Bstar'].keys():
        telluric_flux, telluric_mjd = VR.extraction.get_tellurics_from_bstar(
            calibration_runs['Bstar'][bstar_exposure], master_flat_images,
            debug = True
        )

    print('\n  --> DONE Testing: get_tellurics_from_bstar()')


def test_extract_orders_Flat():
    print('\n  --> Testing: extract_orders() with Flat')

    VR.config.date = '001122'
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'

    calibration_runs = {
        'Flat_60.0': ['0030'],
        'Flat_1.0': ['0016'],
        'Flat_0.1': ['0009']
    }

    print('\n      --> Testing with update_tramlines_based_on_flat=True & debug_rows=True')
    master_flat, master_flat_images = VR.extraction.extract_orders(
        ccd1_runs = calibration_runs['Flat_60.0'],
        ccd2_runs = calibration_runs['Flat_1.0'],
        ccd3_runs = calibration_runs['Flat_0.1'],
        Flat = True,
        update_tramlines_based_on_flat = True,
        debug_overscan = False,
        debug_rows = True,
        debug_tramlines = False
    )

    print('\n      --> Testing with debug_overscan=True')
    master_flat, master_flat_images = VR.extraction.extract_orders(
        ccd1_runs = calibration_runs['Flat_60.0'],
        ccd2_runs = calibration_runs['Flat_1.0'],
        ccd3_runs = calibration_runs['Flat_0.1'],
        Flat = True,
        update_tramlines_based_on_flat = False,
        debug_overscan = True,
        debug_rows = False,
        debug_tramlines = False
    )

    print('\n     --> Testing with debug_rows=True')
    master_flat, master_flat_images = VR.extraction.extract_orders(
        ccd1_runs = calibration_runs['Flat_60.0'],
        ccd2_runs = calibration_runs['Flat_1.0'],
        ccd3_runs = calibration_runs['Flat_0.1'],
        Flat = True,
        update_tramlines_based_on_flat = False,
        debug_overscan = False,
        debug_rows = True,
        debug_tramlines = False
    )

    print('\n     --> Testing with debug_tramlines=True')
    master_flat, master_flat_images = VR.extraction.extract_orders(
        ccd1_runs = calibration_runs['Flat_60.0'],
        ccd2_runs = calibration_runs['Flat_1.0'],
        ccd3_runs = calibration_runs['Flat_0.1'],
        Flat = True,
        update_tramlines_based_on_flat = False,
        debug_overscan = False,
        debug_rows = False,
        debug_tramlines = True
    )

    print('\n  --> DONE Testing: extract_orders() with Flat')

def test_extract_orders_ThXe():
    print('\n  --> Testing: extract_orders() with ThXe')

    VR.config.date = '001122'
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'

    calibration_runs = {
        'FibTh_180.0': ['0047'],
        'FibTh_60.0': ['0042'],
        'FibTh_15.0': ['0037']
    }

    print('\n     --> Testing with debug_tramlines=True')
    master_thxe = VR.extraction.extract_orders(
        ccd1_runs = calibration_runs['FibTh_180.0'],
        ccd2_runs = calibration_runs['FibTh_60.0'],
        ccd3_runs = calibration_runs['FibTh_15.0'],
        ThXe = True,
        debug_tramlines = True
    )

    print('\n  --> DONE Testing: extract_orders() with ThXe')

def test_extract_orders_LC():
    print('\n  --> Testing: extract_orders() with LC')

    VR.config.date = '001122'
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'

    calibration_runs = {
        'SimLC': ['0159'],
        'SimLC': ['0159'],
        'SimLC': ['0159']
    }

    print('\n     --> Testing with debug_tramlines=True')
    master_lc = VR.extraction.extract_orders(
        ccd1_runs = calibration_runs['SimLC'],
        ccd2_runs = calibration_runs['SimLC'],
        ccd3_runs = calibration_runs['SimLC'],
        LC = True,
        debug_tramlines = True
    )

    print('\n  --> DONE Testing: extract_orders() with LC')

def test_extract_order_Science():
    print('\n  --> Testing: extract_orders() with Science')

    VR.config.date = '001122'
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'

    science_runs = {
        'HIP69673': ['0150','0151']
    }
    
    print('\n     --> Testing with master_darks = None & debug_tramlines=True')
    science, science_noise, science_header = VR.extraction.extract_orders(
        ccd1_runs = science_runs['HIP69673'],
        ccd2_runs = science_runs['HIP69673'],
        ccd3_runs = science_runs['HIP69673'],
        Science = True,
        debug_tramlines = True
    )

    print('\n     --> Testing with master_darks = 1800.0 & exposure_time_threshold_darks = 5.0 & debug_tramlines=True')
    master_darks = dict()
    master_darks['1800.0'] = VR.extraction.get_master_dark(None, archival=True)
    science, science_noise, science_header = VR.extraction.extract_orders(
        ccd1_runs = science_runs['HIP69673'],
        ccd2_runs = science_runs['HIP69673'],
        ccd3_runs = science_runs['HIP69673'],
        Science = True,
        master_darks = master_darks,
        exposure_time_threshold_darks = 5.0,
        debug_tramlines = True
    )

    print('\n  --> DONE Testing: extract_orders() with Science')

def test_extract_orders_ValueErrors():
    print('\n  --> Testing: extract_orders() to raise ValueErrors')

    # Let's use Flat=False and update_tramlines_based_on_flat=True

    VR.config.date = '001122'
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'

    calibration_runs = {
        'FibTh_180.0': ['0047'],
        'FibTh_60.0': ['0042'],
        'FibTh_15.0': ['0037']
    }

    print('     --> Testing ValueError with neither Flat nor LC nor Bstar nor Science nor ThXe')
    with pytest.raises(ValueError) as excinfo:
        master_thxe = VR.extraction.extract_orders(
            ccd1_runs = calibration_runs['FibTh_180.0'],
            ccd2_runs = calibration_runs['FibTh_60.0'],
            ccd3_runs = calibration_runs['FibTh_15.0']
        )
    print(f'     --> ValueError raised: {excinfo.value}')

    print('     --> Testing ValueError with Flat=False and update_tramlines_based_on_flat, wrongly using ThXe=True.')
    with pytest.raises(ValueError) as excinfo:
        master_thxe = VR.extraction.extract_orders(
            ccd1_runs = calibration_runs['FibTh_180.0'],
            ccd2_runs = calibration_runs['FibTh_60.0'],
            ccd3_runs = calibration_runs['FibTh_15.0'],
            ThXe = True,
            update_tramlines_based_on_flat = True
        )
    print(f'     --> ValueError raised: {excinfo.value}')

    print('\n  --> DONE Testing: extract_orders() to raise ValueErrors')

# Run the test function
if __name__ == "__main__":

    test_substract_overscan()

    test_read_in_order_tramlines_tinney()

    test_read_in_order_tramlines()

    test_get_master_dark()

    test_get_tellurics_from_bstar()

    test_extract_orders_Flat()

    test_extract_orders_ThXe()

    test_extract_orders_LC()

    test_extract_order_Science()

    test_extract_orders_ValueErrors()