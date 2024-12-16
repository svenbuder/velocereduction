import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import config

from VeloceReduction.utils import polynomial_function

def calibrate_wavelength(science_object, create_overview_pdf=False):
    """
    The function to read in the FITS file of an extracted science_object
    and perform the wavelength calibration to overwrite placeholder wavelength arrays.
    """
    
    # Directory where the reduced data for this science_object can be found
    input_output_directory = config.working_directory+'reduced_data/'+config.date+'/'+science_object

    # Prepare to save overview PDF 
    with PdfPages(input_output_directory+'/veloce_spectra_'+science_object+'_'+config.date+'_overview.pdf') as pdf:

        # Read in FITS file and prepare it to be updated
        with fits.open(input_output_directory+'/veloce_spectra_'+science_object+'_'+config.date+'.fits', mode='update') as file:

            # Loop over the extension, i.e. order, of the fits file
            for order in range(1,len(file)):

                order_name = file[order].header['EXTNAME'].lower()[len(science_object)+1:]
                
                # Note: Some orders may either not exists or cannot be wavelength calibrated.
                if order_name not in [
                    'ccd_2_order_133',
                    'ccd_2_order_134',
                    'ccd_2_order_135',
                    'ccd_2_order_136',
                    'ccd_2_order_137',
                    'ccd_2_order_138',
                    'ccd_2_order_139',
                    'ccd_2_order_140',
                    'ccd_3_order_65',
                    'ccd_3_order_66',
                    'ccd_3_order_67',
                    'ccd_3_order_68',
                    'ccd_3_order_69',
                    'ccd_3_order_82',
                    'ccd_3_order_83',
                    'ccd_3_order_84',
                    'ccd_3_order_85',
                    'ccd_3_order_86',
                    'ccd_3_order_87',
                    'ccd_3_order_88',
                    'ccd_3_order_89',
                    'ccd_3_order_90',
                    'ccd_3_order_96',
                    'ccd_3_order_97',
                    'ccd_3_order_98',
                    'ccd_3_order_99',
                    'ccd_3_order_100',
                ]:
                
                    try:

                        # Use the initial pixel <-> wavelength information per order to fit a polynomial function to it.
                        # Here wavelength is reported in vacuum and in units of nm, not Å.
                        # So we will later multiply it with 10 to report wavelength in vacuum in Å.
                        # We will also use a published form to convert from vacuum to air wavelength and report that for ease.
                        thxe_pixels_and_wavelengths = np.array(np.loadtxt('./VeloceReduction/veloce_reference_data/thxe_pixels_and_positions/'+file[order].header['EXTNAME'].lower()[len(science_object)+1:]+'_px_wl.txt'))
                        wavelength_solution_vacuum_coefficients, x = curve_fit(polynomial_function,
                            thxe_pixels_and_wavelengths[:,0] - 2064,
                            thxe_pixels_and_wavelengths[:,1],
                            p0 = [np.median(thxe_pixels_and_wavelengths[:,1]), 0.05, 0.0, 0.0, 0.0]
                        )
                        wavelength_solution_vacuum = polynomial_function(np.arange(len(file[order].data['WAVE_VAC'])),*wavelength_solution_vacuum_coefficients)
                        file[order].data['WAVE_VAC'] = wavelength_solution_vacuum*10. # report Å, not nm.

                        # Using conversion from Birch, K. P., & Downs, M. J. 1994, Metro, 31, 315
                        # Consistent to the 2024 version of Korg (https://github.com/ajwheeler/Korg.jl)
                        file[order].data['WAVE_AIR'] = file[order].data['WAVE_VAC'] / (1 + 0.0000834254 + 0.02406147 / (130 - (1e4/file[order].data['WAVE_VAC'])**2) + 0.00015998 / (38.9 - (1e4/file[order].data['WAVE_VAC'])**2))

                        #print('Calibrated wavelength for '+file[order].header['EXTNAME'])

                        if create_overview_pdf:
                            f, gs = plt.subplots(4,1,figsize=(15,10),sharex=True)
                            f.suptitle(config.date+' '+science_object+' '+file[order].header['EXTNAME'][len(science_object)+1:])

                            ax = gs[0]
                            ax.plot(file[order].data['SCIENCE']/file[order].data['FLAT'], lw=1)
                            ax.set_yscale('log')
                            ax.set_ylabel('Science')

                            ticks = np.arange(0,4128,100)
                            ax.set_xticks(ticks,labels=ticks,rotation=90)
                            ax.set_xlim(ticks[0],ticks[-1])   
                            ax2 = ax.twiny()
                            ax2.set_xticks(ticks-ticks[0])
                            ax2.set_xticklabels(np.round(file[order].data['WAVE_AIR'][ticks],1),rotation=90)
                            ax2.set_xlabel(r'Air Wavelength $\lambda_\mathrm{air}~/~\mathrm{\AA}$')

                            ax = gs[1]
                            ax.plot(file[order].data['SCIENCE']/file[order].data['SCIENCE_NOISE'], lw=1)
                            ax.set_ylabel('Science Signal-to-Noise')

                            ax = gs[2]
                            ax.plot(file[order].data['THXE']/file[order].data['FLAT'], lw=1)
                            ax.set_yscale('log')
                            ax.set_ylabel('ThXe')

                            ax = gs[3]
                            ax.plot(file[order].data['LC']/file[order].data['FLAT'], lw=0.5)
                            ax.set_yscale('log')
                            ax.set_ylabel('Lc')
                            ax.set_xlabel('Pixel')
                            ax.set_xticks(ticks,labels=ticks,rotation=90)

                            wavelength = file[order].data['WAVE_AIR']
                            science = file[order].data['SCIENCE']/file[order].data['FLAT']
                            thxe = file[order].data['THXE']/file[order].data['FLAT']
                            lc = file[order].data['LC']/file[order].data['FLAT']

                            pdf.savefig()
                            plt.close()

                    except:
                        print('Could not calibrate wavelength for '+file[order].header['EXTNAME'])
                        
                #else:
                #    print('Skipped calibration of wavelength for '+file[order].header['EXTNAME'])
