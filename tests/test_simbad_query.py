import velocereduction
from velocereduction.utils import update_fits_header_via_crossmatch_with_simbad

# Define the FITS header as a dictionary
fits_header = {
    'OBJECT': 'HIP69673',
    'MEANRA': 213.907739365913,  # Right Ascension in decimal degrees
    'MEANDEC': 19.1682209854537, # Declination in decimal degrees
    'UTMJD': 60359.7838614119    # Modified Julian Date at start of exposure
}

def test_update_fits_header():
    # Call the function with the mock FITS header
    updated_header = update_fits_header_via_crossmatch_with_simbad(fits_header)
    # Print the updated header to see the changes
    print("Updated FITS Header:")
    for key, value in updated_header.items():
        print(f"{key}: {value}")

# Run the test function
if __name__ == "__main__":
    test_update_fits_header()
