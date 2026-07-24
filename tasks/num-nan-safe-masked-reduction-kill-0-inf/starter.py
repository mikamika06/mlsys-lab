import numpy as np

def masked_sum(data, mask):
    """Return the sum of data values where mask is True, handling special floats."""
    data = np.asarray(data, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    # Zero out unmasked positions, then reduce
    masked = data * mask
    return float(np.sum(masked))
