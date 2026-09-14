import numpy as np


def relative_difference(reference, comparison):
    reference, comparison = np.asarray(reference, float), np.asarray(comparison, float)
    return np.divide(
        comparison - reference, reference, out=np.full_like(reference, np.nan),
        where=np.isfinite(reference) & (reference != 0),
    )
