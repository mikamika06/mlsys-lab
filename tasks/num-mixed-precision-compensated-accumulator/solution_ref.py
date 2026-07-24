import numpy as np


def compensated_sum(x):
    s = np.float32(0.0)
    c = np.float32(0.0)

    for value in x:
        y = np.float32(value) - c
        t = s + y
        c = (t - s) - y
        s = t

    return float(s)
