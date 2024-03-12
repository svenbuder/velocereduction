from setuptools import setup, find_packages

setup(
    name='veloce_luminosa_reduction',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'veloce_luminosa_reduction=scripts.veloce_luminosa_reduction:main',
        ],
    },
    # Add other package dependencies as needed
    install_requires=[
        'numpy',
        'astropy'
    ],
    # Metadata
    author='Sven buder',
    author_email='sven.buder@anu.edu.au',
    description='A package for reducing CCD images from the Veloce echelle spectrograph',
    license='MIT',
    keywords='astronomy spectroscopy data-reduction',
)
