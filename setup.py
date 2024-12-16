from setuptools import setup, find_packages

setup(
    name='VeloceReduction',
    version='0.2.0',
    packages=find_packages(),
    # Add other package dependencies as needed
    install_requires=[
        'numpy',
        'astropy',
        'scipy',
        'matplotlib'
    ],
    # Metadata
    author='Sven buder',
    author_email='sven.buder@anu.edu.au',
    description='A package for reducing CCD images from the Veloce echelle spectrograph',
    license='MIT',
    keywords='astronomy spectroscopy data-reduction',
)
