import numpy as np

N_DISPERSION = 4112
N_CROSS_DISPERSION = 4096
TRACE_DEGREE = 4
TRACE_HALF_WINDOW = 40
REFERENCE_NIGHT = "001122"
REFERENCE_PIXEL = 2048.0

CCD_ORDERS = {
    "1": np.arange(167, 137, -1),
    "2": np.arange(140, 102, -1),
    "3": np.arange(104, 64, -1),
}

SLIT_ORDER = (
    "ThXe", "S5", "S2", "Blank",
    7, 18, 17, 6, 16, 15, 5, 14, 13, 1, 12, 11, 4, 10, 9, 3, 8, 19, 2,
    "Blank", "S4", "S3", "S1", "LC",
)
SCIENCE_FIBRES = tuple(x for x in SLIT_ORDER if isinstance(x, int))
SKY_FIBRES = tuple(x for x in SLIT_ORDER if isinstance(x, str) and x.startswith("S"))
FIBRE_COMPONENTS = tuple(x for x in SLIT_ORDER if x in SCIENCE_FIBRES + SKY_FIBRES)
CENTRAL_SLIT_INDEX = SLIT_ORDER.index(1)
FIBRE_SLOTS = np.array([SLIT_ORDER.index(x) - CENTRAL_SLIT_INDEX for x in FIBRE_COMPONENTS], float)
FIBRE_SLOT = dict(zip(FIBRE_COMPONENTS, FIBRE_SLOTS))

CALIBRATION_EXPTIMES = {
    "Flat": {"1": 60.0, "2": 1.0, "3": 0.1},
    "SimTh": {"1": 180.0, "2": 60.0, "3": 15.0},
    "FibTh": {"1": 180.0, "2": 60.0, "3": 15.0},
}
EXPECTED_CCDS = {
    "Bias": ("1", "2", "3"), "Dark": ("1", "2", "3"),
    "Flat": ("1", "2", "3"), "SimTh": ("1", "2", "3"),
    "FibTh": ("1", "2", "3"), "Science": ("1", "2", "3"),
    "SimLC": ("2", "3"),
}
CALIBRATION_TYPES = ("Flat", "SimTh", "SimLC", "FibTh", "Bias", "Dark")

LC_REPEAT_GHZ = 25.0
LC_OFFSET_GHZ = 9.56
C_ANGSTROM_GHZ = 2.9979246e9
C_KMS = 299792.458

SSO_LAT_DEG = -31.2749453
SSO_LON_DEG = 149.0684588
SSO_HEIGHT_M = 1164.0
