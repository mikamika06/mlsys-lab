import numpy as np


def _oracle_reconstruct(X, k):
    n = X.shape[0]
    C = X.T @ X / n
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending eigenvalue order
    V = eigvecs[:, -k:]  # LARGEST-eigenvalue directions -> min reconstruction error
    P = X @ V
    return P @ V.T


def grade(sol, fx) -> dict:
    """
    Builds several seeded random data matrices, computes the minimum-error
    rank-k reconstruction (top-k eigenvectors of X^T X / n by DESCENDING
    eigenvalue, i.e. the Eckart-Young-optimal subspace) with a NumPy
    oracle, and compares it (max abs error) to the submission's
    reconstruction. Reconstruction is invariant to eigenvector sign, so a
    correct fix matches the oracle exactly up to float noise.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(4):
        n = int(rng.integers(30, 60))
        d = int(rng.integers(5, 10))
        k = int(rng.integers(2, d - 1))
        X = rng.normal(size=(n, d)) @ np.diag(rng.uniform(0.3, 3.0, size=d))

        Xhat_exp = _oracle_reconstruct(X, k)

        try:
            Xhat_got = np.asarray(sol.pca_reconstruct(X.copy(), k), dtype=np.float64)
        except Exception:
            return {"recon_max_abs_err": float("inf")}

        if Xhat_got.shape != Xhat_exp.shape:
            worst = float("inf")
        else:
            worst = max(worst, float(np.max(np.abs(Xhat_got - Xhat_exp))))

    return {"recon_max_abs_err": worst}
