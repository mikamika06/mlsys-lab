import numpy as np


def rolling_window_mean(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    view = np.lib.stride_tricks.as_strided(
        x,
        shape=(x.shape[0] - window + 1, window),
        strides=(x.strides[0], x.strides[0]),
        writeable=False,
    )
    return np.mean(view, axis=1, dtype=np.float64)
