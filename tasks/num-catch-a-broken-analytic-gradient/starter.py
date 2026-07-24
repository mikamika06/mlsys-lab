import numpy as np


def fixed_gradient(x):
    # TODO: fix the broken analytic gradient.
    # The cubic term is differentiated correctly, but the quadratic term
    # incorrectly uses 2*x instead of 4*x.
    x = np.asarray(x, dtype=np.float64)
    i = np.arange(x.size, dtype=np.float64)
    return 3 * (i + 1) * x**2 + 2 * x - 5
