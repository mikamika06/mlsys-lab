import math

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
    y = np.asarray(x, dtype=np.int64).copy()
    n = y.shape[0]
    if n <= 1:
        return y

    n_rounds = math.ceil(math.log2(n))
    shift = 1
    for _ in range(n_rounds):
        prev = y.copy()  # read the previous round's values only
        if shift < n:
            y[shift:] = prev[shift:] + prev[:-shift]
        shift *= 2
    return y
