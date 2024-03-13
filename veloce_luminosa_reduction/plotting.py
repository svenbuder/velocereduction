import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_ccd_imshow(ax, image, panel_title):
    
    ax.set_title(panel_title)
    count_percentiles = np.percentile(image, q=[20,95])
    s = ax.imshow(image,cmap='OrRd',vmin=np.max([0,count_percentiles[0]]),vmax=count_percentiles[-1])
    cbar = plt.colorbar(s, ax=ax, extend='both', orientation='horizontal')
    cbar.set_label('Counts')

def wavelength_to_rgb(wavelength, gamma=1.0):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A = 1.
    if wavelength < 380:
        wavelength = 380.
#     if wavelength >750:
#         wavelength = 751.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    elif wavelength >= 750 and wavelength <= 800:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B,A)


def create_rainbow_colormap():
    clim=(350,780)
    cmap_norm = plt.Normalize(*clim)
    wl = np.arange(clim[0],clim[1]+1,1)
    colorlist = list(zip(cmap_norm(wl),[wavelength_to_rgb(w) for w in wl]))
    colormap = LinearSegmentedColormap.from_list("spectrum", colorlist)
    return(colormap)

def create_transparent_greyscale_colormap():
    colors = [(0, 0, 0, 1), (0.5, 0.5, 0.5, 0)]  # RGBA colors from transparent to black
    cmap_name = 'transparent'
    colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    return(colormap)


