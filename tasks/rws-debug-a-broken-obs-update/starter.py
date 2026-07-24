import numpy as np


def obs_update(W: np.ndarray, Hinv: np.ndarray, q: int) -> np.ndarray:
    # TODO: missing the OBS curvature divisor Hinv[q, q].
    # This produces a correction with the wrong magnitude.
    out = np.array(W, dtype=np.float64, copy=True)
    column = out[:, q].copy()
    for j in range(out.shape[1]):
        if j != q:
            out[:, j] -= column * Hinv[q, j]
    out[:, q] = 0.0
    return out
