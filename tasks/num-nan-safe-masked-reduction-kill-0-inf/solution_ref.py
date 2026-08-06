import numpy as np

def masked_sum(data, mask):
    """Return the sum of data values where mask is True.

    Uses np.where to avoid the 0*inf NaN trap that data*mask creates.
    """
    data = np.asarray(data, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    return float(np.sum(np.where(mask, data, 0.0)))
