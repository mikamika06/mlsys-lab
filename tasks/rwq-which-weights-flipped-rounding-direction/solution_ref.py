import numpy as np

def classify_rounding(W: np.ndarray, V: np.ndarray, s: float) -> np.ndarray:
    """
    Classify each weight according to how its rounded value changes when a
    correction vector is added.

    Parameters
    ----------
    W : np.ndarray
        Original weights.
    V : np.ndarray
        Correction vector of the same shape as `W`.
    s : float
        Positive scaling factor used before rounding.

    Returns
    -------
    np.ndarray
        Integer array with values -1 (rounded down), 0 (no change),
        or +1 (rounded up).
    """
    # Compute rounded values before and after adding V
    r0 = np.round(W / s)
    r1 = np.round((W + V) / s)

    # Classify differences: >0 -> up, <0 -> down, ==0 -> nearest
    return (r1 > r0).astype(np.int8) - (r1 < r0).astype(np.int8)
