import numpy as np


def fixed_gradient(x):
    x = np.asarray(x, dtype=np.float64)
    i = np.arange(x.size, dtype=np.float64)
    return 3 * (i + 1) * x**2 + 4 * x - 5
