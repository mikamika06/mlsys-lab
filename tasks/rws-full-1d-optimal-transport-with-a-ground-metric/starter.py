import numpy as np


def ot_cost_1d(positions: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """1-D optimal transport cost with ground cost |x - y|.

    positions: 1-D float64 array of n distinct support point locations
        (not necessarily sorted).
    p, q: 1-D nonnegative float64 arrays of length n, masses at those
        positions. sum(p) == sum(q).

    1. Sort the support points ascending, reorder p and q to match.
    2. Cumulative sums P, Q of the reordered p, q.
    3. Return sum_{k=0}^{n-2} |P[k] - Q[k]| * (x_sorted[k+1] - x_sorted[k]).
    """
    raise NotImplementedError('your code here')
