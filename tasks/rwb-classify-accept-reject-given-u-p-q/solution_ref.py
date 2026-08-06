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
    n = p.shape[0]
    out = np.empty(n, dtype=bool)
    for i in range(n):
        qi = q[i]
        if qi == 0.0:
            ratio = 0.0
        else:
            ratio = p[i] / qi
        if ratio < 1.0:
            ratio_clamped = ratio
        else:
            ratio_clamped = 1.0
        out[i] = u <= ratio_clamped
    return out
