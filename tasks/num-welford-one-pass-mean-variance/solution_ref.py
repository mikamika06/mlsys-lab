import numpy as np


def welford_mean_var(x: np.ndarray) -> tuple:
    """Single-pass mean and population variance via Welford's recurrence.

    Visits each element of `x` exactly once, updating a running mean and
    running M2 (sum of squared deviations from the running mean). Never
    forms sum(x**2) or makes a second pass.
    """
    n = 0
    mean = 0.0
    m2 = 0.0
    for xi in x:
        xi = float(xi)
        n += 1
        delta = xi - mean
        mean += delta / n
        delta2 = xi - mean
        m2 += delta * delta2
    var = m2 / n
    return mean, var
