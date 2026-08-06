import numpy as np


def alibi_slopes(n: int) -> np.ndarray:
    h = np.arange(n, dtype=np.float64)
    return np.power(2.0, -8.0 * h / n).astype(np.float64)
