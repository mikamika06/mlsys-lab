import numpy as np


def expected_acceptance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Expected single-token speculative-decoding acceptance probability:

        E[accept] = sum_v min(p_v, q_v) = 1 - TV(p, q) = 1 - 0.5 * sum_v |p_v - q_v|

    p is the target distribution, q is the draft distribution, both 1-D
    probability vectors over the same support (summing to 1).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    total = 0.0
    for i in range(len(p)):
        diff = p[i] - q[i]
        if diff < 0.0:
            abs_diff = -diff
        else:
            abs_diff = diff
        total += abs_diff
    return float(1.0 - 0.5 * total)
