import numpy as np

def topk_deflation(A: np.ndarray, k: int):
    """
    Return the k largest eigenvalues and corresponding orthonormal eigenvectors
    of a real symmetric matrix A using NumPy's eigh routine.
    """
    A = np.asarray(A, dtype=np.float64)
    vals, vecs = np.linalg.eigh(A)          # full spectrum sorted ascending
    idx = np.argsort(vals)[::-1]            # descending order
    return vals[idx][:k], vecs[:, idx][:, :k]
