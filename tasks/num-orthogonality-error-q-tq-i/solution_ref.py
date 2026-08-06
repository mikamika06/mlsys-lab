import numpy as np

def orthogonality_error(Q: np.ndarray) -> float:
    """Return the max-abs element of (Q.T @ Q - I)."""
    n = Q.shape[0]
    max_val = 0.0
    first = True
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += Q[k, i] * Q[k, j]
            val = s - 1.0 if i == j else s
            abs_val = abs(val)
            if first or abs_val > max_val:
                max_val = abs_val
                first = False
    return float(max_val)
