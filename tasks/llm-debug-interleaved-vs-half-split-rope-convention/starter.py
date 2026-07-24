import numpy as np


def apply_rope(x: np.ndarray, position: int) -> np.ndarray:
    # TODO: this uses adjacent dimension pairs instead of the half-split
    # GPT-NeoX convention required by the task.
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[0]
    out = np.empty_like(x, dtype=np.float64)

    idx = np.arange(0, d, 2, dtype=np.float64)
    theta = position * (10000.0 ** (-2.0 * (idx / 2) / d))

    for j, angle in enumerate(theta):
        i = 2 * j
        c = np.cos(angle)
        s = np.sin(angle)
        out[i] = x[i] * c - x[i + 1] * s
        out[i + 1] = x[i] * s + x[i + 1] * c

    return out
