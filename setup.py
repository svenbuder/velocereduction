from setuptools import setup, find_packages

setup(
    name="velocereduction",
    version="0.5.4",
    install_requires=[
        "numpy",
        "astropy",
        "scipy",
        "matplotlib",
        "astroquery>=0.4.8",
        "scikit-image",
        "pytest",
        "pytest-cov",
    ],
    python_requires='>=3.9',
    author="Sven Buder",
    author_email="sven.buder@anu.edu.au",
    description="A package for reducing CCD images from the Veloce echelle spectrograph",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/svenbuder/velocereduction",
    packages=find_packages(),
    keywords=["astronomy", "spectroscopy", "data-reduction"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
