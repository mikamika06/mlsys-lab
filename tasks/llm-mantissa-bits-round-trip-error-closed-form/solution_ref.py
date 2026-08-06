import numpy as np
import math

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
    
    if m.ndim == 0:
        val = math.pow(2.0, -(int(m) + 1))
        res = np.array([val], dtype=np.float64)
    else:
        out_list = []
        for x in m:
            val = math.pow(2.0, -(int(x) + 1))
            out_list.append(val)
        res = np.array(out_list, dtype=np.float64)
        
    return np.atleast_1d(res).astype(np.float64)
