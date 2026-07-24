import numpy as np

def iterations_to_tolerance(A: np.ndarray, tol: float = 1e-6) -> int:
    """
    Return the number of power‑iteration steps needed for successive iterates
    to differ by less than ``tol`` in Euclidean norm.
    The initial vector is all ones normalised to unit length.
    If convergence is not reached within 10 000 iterations, return 10000.
    """
    n = A.shape[0]
    v = np.ones(n, dtype=np.float64)
    v /= np.linalg.norm(v)
    for i in range(1, 10001):
        w = A @ v
        norm_w = np.linalg.norm(w)
        if norm_w == 0:
            return i
        v_next = w / norm_w
        diff = np.linalg.norm(v_next - v)
        if diff < tol:
            return i
        v = v_next
    return 10000
