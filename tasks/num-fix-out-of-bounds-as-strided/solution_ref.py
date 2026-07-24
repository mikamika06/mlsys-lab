import numpy as np


def fixed_windows(x: np.ndarray, width: int) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if width <= 0 or width > len(x):
        raise ValueError("invalid width")
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(len(x) - width + 1, width),
        strides=(x.strides[0], x.strides[0]),
        writeable=False,
    )
