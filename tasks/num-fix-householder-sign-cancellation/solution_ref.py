import numpy as np


def householder_fixed(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.shape[0]
    norm = np.linalg.norm(x)
    if norm == 0:
        return np.eye(m, dtype=np.float64)
    alpha = -np.sign(x[0]) * norm
    if x[0] == 0:
        alpha = -norm
    v = x.copy()
    v[0] -= alpha
    denom = np.dot(v, v)
    if denom == 0:
        return np.eye(m, dtype=np.float64)
    return np.eye(m, dtype=np.float64) - 2.0 * np.outer(v, v) / denom
