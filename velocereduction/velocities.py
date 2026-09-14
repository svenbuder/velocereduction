import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from .constants import C_KMS, SSO_HEIGHT_M, SSO_LAT_DEG, SSO_LON_DEG

SSO = EarthLocation.from_geodetic(
    lon=SSO_LON_DEG * u.deg, lat=SSO_LAT_DEG * u.deg, height=SSO_HEIGHT_M * u.m,
)


def barycentric_velocity_correction(ra_deg, dec_deg, mjd):
    target = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg)
    correction = target.radial_velocity_correction(
        obstime=Time(float(mjd), format="mjd", scale="utc"), location=SSO,
    )
    return float(correction.to_value(u.km / u.s))


def barycentric_wavelength(wavelength, berv_kms):
    """Optical-convention wavelength correction, including the RV cross term."""
    return np.asarray(wavelength, float) * (1.0 + float(berv_kms) / C_KMS)
