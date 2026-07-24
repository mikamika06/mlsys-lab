import numpy as np

def pairwise_sq_dists(A):
    """Vectorised computation of squared Euclidean distances."""
    A = np.asarray(A, dtype=np.float64)
    sq_norms = np.sum(A*A, axis=1).reshape(-1, 1)
    return sq_norms + sq_norms.T - 2 * A.dot(A.T)
