import numpy as np


def householder_fixed(x: np.ndarray) -> np.ndarray:
    # TODO: this uses the cancellation-prone sign choice.
    x = np.asarray(x, dtype=np.float64)
    m = x.shape[0]
    norm = np.linalg.norm(x)
    if norm == 0:
        return np.eye(m, dtype=np.float64)
    v = x - norm * np.eye(m, dtype=np.float64)[:, 0]
    denom = np.dot(v, v)
    if denom == 0:
        return np.eye(m, dtype=np.float64)
    return np.eye(m, dtype=np.float64) - 2.0 * np.outer(v, v) / denom
