import numpy as np

def classify_accept(u: float, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return a boolean array indicating acceptance for each token.

    Parameters
    ----------
    u : float
        Acceptance threshold.
    p, q : np.ndarray
        1‑D arrays of the same length.  ``q`` may contain zeros; in that case
        the corresponding ratio is treated as zero.

    Returns
    -------
    np.ndarray[bool]
        Boolean array where element *i* is ``True`` iff
        ``u <= min(p[i]/q[i], 1)``.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    ratio = np.where(q == 0.0, 0.0, p / q)
    ratio_clamped = np.minimum(ratio, 1.0)
    return u <= ratio_clamped
