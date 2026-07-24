import numpy as np

_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def snap_to_e2m1(x):
    """Snap each element of x to the nearest signed E2M1 FP4 value."""
    x = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x)
    diffs = np.abs(abs_x[:, np.newaxis] - _MAG[np.newaxis, :])
    idx = np.argmin(diffs, axis=1)
    return np.sign(x) * _MAG[idx]
