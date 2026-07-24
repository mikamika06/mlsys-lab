import numpy as np


def q4_0_dequantize(x: np.ndarray) -> np.ndarray:
    # TODO: This incorrectly uses one scale for the whole tensor.
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x)) / 7.0
    if scale == 0:
        return np.zeros_like(x)
    q = np.clip(np.round(x / scale), -8, 7)
    return q * scale
