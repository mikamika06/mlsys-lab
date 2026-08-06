import math
import numpy as np

def iterations_to_tolerance(A: np.ndarray, tol: float = 1e-6) -> int:
    """
    Return the number of power‑iteration steps needed for successive iterates
    to differ by less than ``tol`` in Euclidean norm.
    The initial vector is all ones normalised to unit length.
    If convergence is not reached within 10 000 iterations, return 10000.
    """
    n = A.shape[0]
    v = np.ones(n, dtype=np.float64)
    sum_sq_v = 0.0
    for j in range(n):
        sum_sq_v += v[j] * v[j]
    norm_v = math.sqrt(sum_sq_v)
    for j in range(n):
        v[j] /= norm_v

    for i in range(1, 10001):
        w = np.empty(n, dtype=np.float64)
        for r in range(n):
            s = 0.0
            for c in range(n):
                s += A[r, c] * v[c]
            w[r] = s

        sum_sq_w = 0.0
        for r in range(n):
            sum_sq_w += w[r] * w[r]
        norm_w = math.sqrt(sum_sq_w)

        if norm_w == 0:
            return i

        v_next = np.empty(n, dtype=np.float64)
        for r in range(n):
            v_next[r] = w[r] / norm_w

        sum_sq_diff = 0.0
        for r in range(n):
            d = v_next[r] - v[r]
            sum_sq_diff += d * d
        diff = math.sqrt(sum_sq_diff)

        if diff < tol:
            return i
        v = v_next

    return 10000
