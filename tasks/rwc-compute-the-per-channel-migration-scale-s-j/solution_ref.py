import math

def _flatten(nested):
    """Recursively flatten an arbitrarily nested list of numbers."""
    for item in nested:
        if isinstance(item, list):
            yield from _flatten(item)
        else:
            yield float(item)

def compute_migration_scales(W: list, X: list, alpha: float) -> list[float]:
    """
    Compute per‑channel migration scales.

    Parameters
    ----------
    W : list
        Weight tensor of shape (C_out, *).
    X : list
        Activation tensor of shape (N, C_out, *).
    alpha : float
        Hyper‑parameter in [0, 1].

    Returns
    -------
    s : list[float]
        List of length C_out containing the scales.
    """
    out_c = len(W)
    N = len(X)

    s = []

    for c in range(out_c):
        m_w = -1.0
        for val in _flatten(W[c]):
            abs_val = val if val >= 0 else -val
            if abs_val > m_w:
                m_w = abs_val

        m_x = -1.0
        for n in range(N):
            for val in _flatten(X[n][c]):
                abs_val = val if val >= 0 else -val
                if abs_val > m_x:
                    m_x = abs_val

        s.append((m_x ** alpha) / (m_w ** (1 - alpha)))

    return s
