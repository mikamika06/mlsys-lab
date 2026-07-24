import numpy as np


def obs_update(W: np.ndarray, Hinv: np.ndarray, q: int) -> np.ndarray:
    out = np.array(W, dtype=np.float64, copy=True)
    column = out[:, q].copy()
    scale = Hinv[q, q]
    for j in range(out.shape[1]):
        if j != q:
            out[:, j] -= column * Hinv[q, j] / scale
    out[:, q] = 0.0
    return out
