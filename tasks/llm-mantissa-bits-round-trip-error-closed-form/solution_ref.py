import numpy as np

def round_trip_error(mantissa_bits):
    """
    Compute the maximum relative round‑trip error for given mantissa bit widths.
    
    Parameters
    ----------
    mantissa_bits : int or array-like of ints
        Number of mantissa bits. Can be a scalar or a one‑dimensional sequence.
        
    Returns
    -------
    np.ndarray
        Array of float64 values containing 2**(-(m+1)) for each input m.
    """
    m = np.asarray(mantissa_bits, dtype=int)
    res = np.power(2.0, -(m + 1))
    return np.atleast_1d(res).astype(np.float64)
