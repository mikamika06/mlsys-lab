import numpy as np

def snap_nf4(weights: np.ndarray) -> np.ndarray:
    """
    Map each weight in [-1, 1] to the nearest NF4 codebook level.
    Returns a uint8 array of indices with the same shape as `weights`.
    """
    levels = np.array(
        [
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
        ],
        dtype=np.float64,
    )
    w = np.asarray(weights, dtype=np.float64)
    idx = np.argmin(np.abs(w[:, None] - levels[None, :]), axis=1).astype(
        np.uint8
    )
    return idx
