from setuptools import setup, find_packages
from VeloceReduction import __version__ as version

setup(
    name='VeloceReduction',
    version=version,
    packages=find_packages(),
    # Add other package dependencies as needed
    install_requires=[
        'numpy',
        'astropy',
        'scipy',
        'matplotlib',
        'astroquery>=0.4.7',
    ],
    scripts=['./VeloceReduction_tutorial.py'],
    # Metadata
    author='Sven buder',
    author_email='sven.buder@anu.edu.au',
    description='A package for reducing CCD images from the Veloce echelle spectrograph',
    license='MIT',
    url = "https://github.com/svenbuder/VeloceReduction",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='astronomy spectroscopy data-reduction',
)
