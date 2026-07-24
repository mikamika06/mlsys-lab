import numpy as np


def ot_cost_1d(positions: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """1-D optimal transport cost with ground cost |x - y|, via the
    closed-form cumulative-CDF-difference formula (exact for convex
    1-D ground costs -- no LP needed).
    """
    positions = np.asarray(positions, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    order = np.argsort(positions)
    x = positions[order]
    p_sorted = p[order]
    q_sorted = q[order]

    P = np.cumsum(p_sorted)
    Q = np.cumsum(q_sorted)

    gaps = x[1:] - x[:-1]
    return float(np.sum(np.abs(P[:-1] - Q[:-1]) * gaps))
