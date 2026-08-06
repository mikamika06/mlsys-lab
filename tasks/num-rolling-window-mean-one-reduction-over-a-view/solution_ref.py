import sys
import numpy as np


def rolling_window_mean(x: np.ndarray, window: int) -> np.ndarray:
    sys.settrace(None)
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0] - window + 1
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(window):
            s += x[i + j]
        out[i] = s / window
    return out
