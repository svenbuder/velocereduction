from astropy.io import fits

def read_fits_file(filepath):
    """Read a FITS file and return the data and header."""
    with fits.open(filepath) as hdulist:
        data = hdulist[0].data
        header = hdulist[0].header
    return data, header
