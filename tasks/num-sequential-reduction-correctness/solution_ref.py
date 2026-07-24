import numpy as np


def sequential_sum(values):
    total = np.float64(0.0)
    for value in values:
        total = np.float64(total + np.float64(value))
    return total
