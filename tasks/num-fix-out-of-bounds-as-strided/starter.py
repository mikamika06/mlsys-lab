import numpy as np


def fixed_windows(x: np.ndarray, width: int) -> np.ndarray:
    # TODO: the shape keeps one extra row, causing the final window to read
    # beyond the end of the original array buffer for many inputs.
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(len(x) - width + 2, width),
        strides=(x.strides[0], x.strides[0]),
        writeable=False,
    )
