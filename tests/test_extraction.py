import velocereduction as VR
from pathlib import Path

def test_substract_overscan():
    print('\n  --> Testing: substract_overscan()')

    # Load the image to test
    test_file = str(Path(__file__).resolve().parent)+'/../observations/001122/ccd_1/22nov10030.fits'

    full_image, metadata = VR.utils.read_veloce_fits_image_and_metadata(test_file)

    # Test the function without debug
    print('      with debug_overscan=False')
    trimmed_image, _, _, _ = VR.extraction.substract_overscan(full_image, metadata)

    # Test the debug function
    print('      with debug_overscan=True')
    trimmed_image, _, _, _ = VR.extraction.substract_overscan(full_image, metadata, debug_overscan= True)

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

def test_extract_orders_flats():
    print('\n  --> Testing: extraction()')

    VR.config.date = '001122'
    VR.config.working_directory = str(Path(__file__).resolve().parent)+'/../'

    calibration_runs = {
        'Flat_60.0': ['0030'],
        'Flat_1.0': ['0016'],
        'Flat_0.1': ['0009']
    }

    print('\n  --> Extracting Master Flat')

    print('\n  --> Testing with update_tramlines_based_on_flat=True')
    master_flat, _ = VR.extraction.extract_orders(
        ccd1_runs = calibration_runs['Flat_60.0'],
        ccd2_runs = calibration_runs['Flat_1.0'],
        ccd3_runs = calibration_runs['Flat_0.1'],
        Flat = True,
        update_tramlines_based_on_flat = True,
        debug_overscan = False,
        debug_rows = False,
        debug_tramlines = False
    )

    print('\n  --> Testing with debug_overscan=True')
    master_flat, _ = VR.extraction.extract_orders(
        ccd1_runs = calibration_runs['Flat_60.0'],
        ccd2_runs = calibration_runs['Flat_1.0'],
        ccd3_runs = calibration_runs['Flat_0.1'],
        Flat = True,
        update_tramlines_based_on_flat = False,
        debug_overscan = True,
        debug_rows = False,
        debug_tramlines = False
    )

    print('\n  --> Testing with debug_rows=True')
    master_flat, _ = VR.extraction.extract_orders(
        ccd1_runs = calibration_runs['Flat_60.0'],
        ccd2_runs = calibration_runs['Flat_1.0'],
        ccd3_runs = calibration_runs['Flat_0.1'],
        Flat = True,
        update_tramlines_based_on_flat = False,
        debug_overscan = False,
        debug_rows = True,
        debug_tramlines = False
    )

    print('\n  --> Testing with debug_tramlines=True')
    master_flat, _ = VR.extraction.extract_orders(
        ccd1_runs = calibration_runs['Flat_60.0'],
        ccd2_runs = calibration_runs['Flat_1.0'],
        ccd3_runs = calibration_runs['Flat_0.1'],
        Flat = True,
        update_tramlines_based_on_flat = False,
        debug_overscan = False,
        debug_rows = False,
        debug_tramlines = True
    )
    

# Run the test function
if __name__ == "__main__":

    # test_substract_overscan()

    # test_read_in_order_tramlines_tinney()

    # test_read_in_order_tramlines()

    # test_get_master_dark()

    # test_extract_orders_flats()