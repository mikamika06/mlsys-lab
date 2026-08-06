import numpy as np


def ot_cost_1d(positions: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """1-D optimal transport cost with ground cost |x - y|, via the
    closed-form cumulative-CDF-difference formula (exact for convex
    1-D ground costs -- no LP needed).
    """
    positions = np.asarray(positions, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    n = positions.shape[0]
    order = sorted(range(n), key=lambda i: positions[i])

    x = np.empty(n, dtype=np.float64)
    p_sorted = np.empty(n, dtype=np.float64)
    q_sorted = np.empty(n, dtype=np.float64)

    for i in range(n):
        idx = order[i]
        x[i] = positions[idx]
        p_sorted[i] = p[idx]
        q_sorted[i] = q[idx]

    P = np.empty(n, dtype=np.float64)
    Q = np.empty(n, dtype=np.float64)
    curr_p = 0.0
    curr_q = 0.0
    for i in range(n):
        curr_p += p_sorted[i]
        curr_q += q_sorted[i]
        P[i] = curr_p
        Q[i] = curr_q

    total = 0.0
    for i in range(n - 1):
        gap = x[i + 1] - x[i]
        diff = P[i] - Q[i]
        total += abs(diff) * gap

    return float(total)
