import numpy as np


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-free logistic sigmoid, evaluated branchlessly."""
    x = np.asarray(x, dtype=np.float64)
    # exp of a non-positive argument can only underflow, never overflow
    z = np.exp(-np.abs(x))
    denom = 1.0 + z
    return np.where(x >= 0.0, 1.0 / denom, z / denom)
