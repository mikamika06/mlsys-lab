import numpy as np


def alibi_slopes(n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i] = 2.0 ** (-8.0 * float(i) / float(n))
    return out
