import numpy as np

def hessian_and_inverse(X: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    H = 2 * X @ X.T + lam * np.eye(n, dtype=np.float64)

    # Cholesky factorisation for numerical stability
    L = np.linalg.cholesky(H)
    inv_L = np.linalg.solve(L, np.eye(n, dtype=np.float64))
    H_inv = inv_L.T @ inv_L

    return H, H_inv
