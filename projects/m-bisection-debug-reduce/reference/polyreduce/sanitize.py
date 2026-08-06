import numpy as np

def sanitize_tensor(arr, denormal_threshold=1e-7, zero_nans=False):
    """Sanitize float subnormals and NaNs in a numpy array."""
    out = np.array(arr, copy=True)
    if np.issubdtype(out.dtype, np.floating):
        mask = np.abs(out) < denormal_threshold
        out[mask] = 0.0
        if zero_nans:
            out[np.isnan(out)] = 0.0
    return out
