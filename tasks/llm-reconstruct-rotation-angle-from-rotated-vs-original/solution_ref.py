import numpy as np

def recover_angles(orig: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """
    Compute the signed rotation angle (in radians) that maps each row of `orig`
    to the corresponding row of `rot` using vectorised NumPy operations.
    Both inputs must be 2‑D arrays of shape (n, 2). The output is a 1‑D array
    of length n with dtype float64.
    """
    cross = orig[:, 0] * rot[:, 1] - orig[:, 1] * rot[:, 0]
    dot   = orig[:, 0] * rot[:, 0] + orig[:, 1] * rot[:, 1]
    return np.arctan2(cross, dot)
