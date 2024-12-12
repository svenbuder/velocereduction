# *Veloce Luminosa* Reduction Pipeline of Veloce Spectra

This package is designed for the reduction of spectroscopic data from the Veloce spectrograph. It encompasses all necessary steps from preprocessing raw data to producing calibrated and reduced spectra. Detailed visualization tools are also included for quality assessment and analysis of the spectroscopic data.

> :warning: **Warning:** THIS PACKAGE IS STILL UNDER DEVELOPMENT AND DOES NOT YET INCLUDE ALL NECESSARY FEATURES, AND WILL MOST LIKELY FAIL TO RUN.

## Author

Sven Buder (ANU, sven.buder@anu.edu.au)

## Installation Instructions

To install this package, we recommend cloning the repository and installing it in development mode to facilitate updates and customization. Please follow the steps below:

```shell
git clone https://github.com/svenbuder/veloce_luminosa_reduction.git
cd veloce_luminosa_reduction
pip install -e .
```

As this package is still in heavy development, this approach ensures that any modifications you make to the scripts or package will be immediately available without the need for reinstallation.

## Usage Instructions

### Raw Data

1. Each night's raw data as created by the Veloce software should be placed in a separate directory following the naming convention `raw_data/YYMMDD`, where `YYMMDD` is the date of observation.

### Running the Reduction

> :warning: **Warning:** THIS PACKAGE IS STILL UNDER DEVELOPMENT AND DOES NOT YET INCLUDE ALL NECESSARY FEATURES, AND WILL MOST LIKELY FAIL TO RUN.

2. With the data in place, proceed to the `scripts/` directory, where you will find Python scripts (`.py`) and Jupyter notebooks (`.ipynb`) designed for data reduction:

```shell
python veloce_luminosa_reduction_script.py 240219 HIP71683 --working_directory /Users/buder/git/veloce_luminosa_reduction/ --debug
```

These scripts perform a series of reduction steps including calibration, spectral extraction, normalisation, and merging. The process is configured to be flexible, catering to different data sets and objectives.

### Output

3. After running the reduction scripts, the output will be systematically organized in `reduced_data/YYMMDD`, corresponding to each night of observation.

## Workflow

### Identification of calibration and science runs + tramlines extractions
- Identify calibration runs and science runs from log files.
- For calibration frames, the following are hard-coded to be use:
   - Flat_60.0 (Azzurro), Flat_1.0 (Verde), Flat_0.1 (Rosso)
   - FibTh_180.0 (Azzurro), FibTh_60.0 (Verde), FibTh_15.0 (Rosso)
   - SimLC (Verde), SimLC (Rosso)
   - Dark not yet implemented
- Identify overscan region and its RMS, subtract overscan and trim image
- Join image of same type: calculate median image for calibration runs; co-add images for science runs
- Extract tramlines: use predefined regions from Chris Tinney to identify pixels for each order that will be co-added
- Calculate rough SNR: use sqrt(counts) + read noise, where read_noise is calculated from maximum overscan RMS * nr of pixels. For science runs also multiply read noise by nr of runs, since they are co-added.
- Safe files as minimal*.fits with one extension per order. Each order has a data table with 6x4128 entries for wave (at this stage only placeholder), science, science_noise, flat, thxe, and lc. ccd and orders can be identified from the FITS extension "EXTNAME" in the form of "CCD_3_ORDER_93" for CCD3's 93rd order.

### Wavelength calibration
- Read in the minimal*.fits
- Loop over each order and use the preidentified thxe pixel -> wavelength combinations to fit a 4th order polynomial wavelength solution.
- This is currently a static solution (assuming and hoping that the pixel -> wavelength positions do not change over the years)
- LC is currently not used to improve wavelength solution
- Overwrite wave placeholder and save minimal*.fits

## Dependencies

This package requires the following libraries:
- NumPy
- SciPy
- matplotlib
- Astropy

Before running the scripts, ensure these dependencies are installed via `pip` or `conda`.

If you want to use the spectrum normaliation feature with synthetic spectra from `Korg`, you also need to install the following:
- julia (programming language)
- Korg in Julia
- juliacall (python package) to use Korg in python

For further information on julia and Korg, take a look at the [Korg Installation Instructions](https://ajwheeler.github.io/Korg.jl/stable/).

## Contributing

Contributions to enhance and expand this package are highly encouraged. Please feel free to fork the repository, make your improvements, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Taste the Rainbow!

Below is are two reduced spectra of the solar-like star alpha Centauri A (HIP71683, [Fe/H] = 0.20 dex) on the left and the metal-poor star HD 140283 (HIP76976, [Fe/H] = -2.48) on right right. You can also find PDF versions of these plots [here](https://github.com/svenbuder/veloce_luminosa_reduction/blob/main/reduced_data/240219/diagnostic_plots/0141/240219_0141_HIP71683_rainbow.pdf) and [here](https://github.com/svenbuder/veloce_luminosa_reduction/blob/main/reduced_data/240220/diagnostic_plots/0161/240220_0161_HIP76976_rainbow.pdf).

<p align="center">
  <img src="https://github.com/svenbuder/veloce_luminosa_reduction/blob/main/reduced_data/240219/diagnostic_plots/0141/240219_0141_HIP71683_rainbow.png" width="49%"/>
  <img src="https://github.com/svenbuder/veloce_luminosa_reduction/blob/main/reduced_data/240220/diagnostic_plots/0161/240220_0161_HIP76976_rainbow.png" width="49%"/>
</p>
