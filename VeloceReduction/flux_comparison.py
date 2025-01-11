import numpy as np
from astropy.table import Table
from pathlib import Path

def read_korg_syntheses():
    """
    Read synthetic spectra synthesised with the synthesis tool Korg (https://github.com/ajwheeler/Korg.jl).
    The code is run in julia and we thus use spectra precomputed and provided as part of this repository.

    Returns:
        korg_spectra (astropy.table.Table) with columns:
        - wavelength_air (Angstrom): wavelength in air,
        - wavelength_vac (Angstrom): wavelength in vacuum,
        - flux_sun (array):      normalised flux for the Sun,
        - flux_arcturus (array): normalised flux for Arcturus,
        - flux_61cyga (array):   normalised flux for 61 Cyg A,
        - flux_hd22879 (array):  normalised flux for HD 22879,
        - flux_18sco (array):    normalised flux for 18 Sco

    Details on how to recreate the spectra can be found in korg_flux/calculate_korg_flux.ipynb.    
    In short, spectra are synthesised:
    - for the Sun, Arcturus, 61 Cyg A, HD 22879 and 18 Sco with stellar parameters (Teff, logg, [Fe/H], vmic, vsini) from Jofre et al. (2017, http://adsabs.harvard.edu/abs/2017A%26A...601A..38J) and Soubiran et al. (2024, https://ui.adsabs.harvard.edu/abs/2024A&A...682A.145S) for 18 Sco,
    - on a wavelength grid 3590:0.01:9510 Angstrom and downgraded to resolution R=80,000,
    - the default linelist of Korg based on an extraction of lines for the Sun from the VALD database.
    
    Warning:
    The linelist is not including lines between 9000 and 9510Å.
    It is further not including lines that are not visible in the Sun, but possibly in the cooler stars.

    Stellar parameters:
    Sun:      Teff=5771, logg=4.44, [Fe/H]= 0.03, vmic=1.06, vsini=np.sqrt(1.6**2 +4.2**2)
    Acruturs: Teff=4286, logg=1.64, [Fe/H]=-0.52, vmic=1.58, vsini=np.sqrt(3.8**2 +5.0**2)
    61 Cyg A: Teff=4373, logg=4.63, [Fe/H]=-0.33, vmic=1.07, vsini=np.sqrt(0.0**2 +4.2**2)
    HD 22879: Teff=5868, logg=4.27, [Fe/H]=-0.86, vmic=1.05, vsini=np.sqrt(4.4**2 +5.4**2)
    18 Sco:   Teff=5824, logg=4.42, [Fe/H]= 0.03, vmic=1.00, vsini=np.sqrt(2.03**2+3.7**2)

    """
    korg_spectra = Table.read(Path(__file__).resolve().parent / 'korg_flux' / 'korg_flux_sun_arcturus_61cyga_hd22879_18sco_R80000_3590_0.01_9510AA.fits')
    return(korg_spectra)

def read_korg_normalisation_buffers():
    """
    Read the left and right pixel buffers that will be used for the order-by-order comparison of the synthetic spectra.

    Returns:
        buffer_dict (dict) with keys for orders in the form 'ccd_'+ccd+'_order_'+order.
        
    Each order has an 2-element array with the left and right buffer.
    Right buffers are negative and buffers are to be applied as: flux_before_buffering[buffer[0]:buffer[1]].
    """
    buffer_text = np.loadtxt(Path(__file__).resolve().parent / 'korg_flux' / 'korg_order_comparison_buffers.txt', dtype=str)
    normalisation_buffers = dict()
    for buffer in buffer_text:
        normalisation_buffers[buffer[0]] = [int(buffer[1]),int(buffer[2])]
    return(normalisation_buffers)