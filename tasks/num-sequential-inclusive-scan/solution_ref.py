import numpy as np


def inclusive_scan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    running = 0.0
    for i, value in enumerate(x):
        running += value
        out[i] = running
    return out
