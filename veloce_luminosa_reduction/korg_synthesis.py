import numpy as np
from juliacall import Main as jl
jl.seval("using Korg")
Korg = jl.Korg
from . import config

def calculate_synthetic_korg_spectrum(teff, logg, fe_h, vsini=6.25, epsilon=0.6):

    # Read in linelist (Korg default covers only 3900-9000AA)
    korg_lines = Korg.read_linelist(config.working_directory+'veloce_luminosa_reduction/literature_data/vald_linelists/3600_9500_0p01_3500_4.5_MH_0p5_long_withHFS.vald')

    # Calculate Korg wavelength arrays in vacuum and air
    wave_start = '3600'
    wave_end = '9500'
    wave_interval = '0.01'
    korg_wavelength_vac = jl.seval('['+wave_start+':'+wave_interval+':'+wave_end+']')
    wavelength_vac = np.arange(float(wave_start),float(wave_end)+float(wave_interval),float(wave_interval))
    korg_wavelength_air = np.array([Korg.vacuum_to_air(pixel) for pixel in wavelength_vac])

    # Create the A(X) dictionary for synthesising lines with Korg
    alpha_fe = -0.4 * fe_h
    if alpha_fe > 0.4: alpha_fe = 0.4
    if alpha_fe < 0.0: alpha_fe = 0.0
    alpha_h = alpha_fe + fe_h
    a_x_dictionary = Korg.format_A_X(fe_h, alpha_h)

    # Create parameter dictionary
    params = {'teff':teff, 'logg':logg, 'fe_h':fe_h, 'vsini':vsini, 'epsilon':epsilon}
    
    # Interpolate the atmosphere
    atmosphere = Korg.interpolate_marcs(teff, logg, a_x_dictionary)

    # Synthesise the Korg spectrum
    synthesis = Korg.synthesize(atmosphere, korg_lines, a_x_dictionary, korg_wavelength_vac)

    # Broaden spectrum, first with rotation, then for instrumental profile via line-spread-function
    F = np.array(synthesis.flux)/np.array(synthesis.cntm)
    F = Korg.apply_rotation(F, korg_wavelength_vac, params["vsini"], params["epsilon"])

    # Compute LSF matrix for Korg assuming R=80,000 everywhere and apply it onto flux
    # LSF_matrix = Korg.compute_LSF_matrix(
    #     korg_wavelength_vac,
    #     korg_wavelength_vac,
    #     80_000)
    # F = LSF_matrix_all * F
    
    return(synthesis, wavelength_vac, korg_wavelength_air, np.array(F))
