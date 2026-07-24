import numpy as np


def stable_one_minus_cos_over_x2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    half = x / 2.0
    return np.asarray(0.5 * (np.sin(half) / half) ** 2, dtype=np.float64)
