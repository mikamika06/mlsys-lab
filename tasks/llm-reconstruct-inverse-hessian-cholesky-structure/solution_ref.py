import numpy as np

def reconstruct_inverse_hessian(A: np.ndarray, lambda_reg: float) -> np.ndarray:
    """
    Compute the inverse of H = A @ A.T + λ I using Cholesky factorisation.
    Returns an (n,n) array of dtype float64.
    """
    n = A.shape[0]
    H = A @ A.T + lambda_reg * np.eye(n, dtype=np.float64)
    L = np.linalg.cholesky(H)
    I = np.eye(n, dtype=np.float64)
    X = np.linalg.solve(L, I)
    inv_H = np.linalg.solve(L.T, X)
    return inv_H
