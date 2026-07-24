import numpy as np


def welford_mean_var(x: np.ndarray) -> tuple:
    """Single-pass mean and population variance via Welford's recurrence.

    Visit each element of `x` exactly once, updating a running mean and
    running M2 (sum of squared deviations from the running mean). Do not
    call np.mean/np.var or make a second pass over x.
    """
    raise NotImplementedError('your code here')
