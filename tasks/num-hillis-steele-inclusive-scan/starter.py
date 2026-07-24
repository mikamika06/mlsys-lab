import numpy as np


def hillis_steele_scan(x: np.ndarray) -> np.ndarray:
    """Inclusive prefix-sum scan via the Hillis-Steele distance-doubling recurrence.

    Parameters
    ----------
    x : ndarray, shape (N,), integer
        Input sequence.

    Returns
    -------
    ndarray, shape (N,), integer
        y[i] = sum(x[0..i]), computed in ceil(log2(N)) rounds where round k
        combines each element with the one 2**k positions to its left.
    """
    raise NotImplementedError('your code here')
