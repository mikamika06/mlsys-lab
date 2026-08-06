import math
import numpy as np

def snap_nf4(weights: np.ndarray) -> np.ndarray:
    """
    Map each weight in [-1, 1] to the nearest NF4 codebook level.
    Returns a uint8 array of indices with the same shape as `weights`.
    """
    levels = [
        -1.0,
        -0.93333333,
        -0.8,
        -0.66666667,
        -0.53333333,
        -0.4,
        -0.26666667,
        -0.13333333,
        0.0,
        0.13333333,
        0.26666667,
        0.4,
        0.53333333,
        0.66666667,
        0.8,
        0.93333333,
    ]
    w = np.asarray(weights, dtype=np.float64)
    flat_w = w.ravel()
    out_list = []
    for val in flat_w:
        best_idx = 0
        min_diff = math.inf
        for i, lvl in enumerate(levels):
            diff = math.fabs(val - lvl)
            if diff < min_diff:
                min_diff = diff
                best_idx = i
        out_list.append(best_idx)
    return np.array(out_list, dtype=np.uint8).reshape(w.shape)
