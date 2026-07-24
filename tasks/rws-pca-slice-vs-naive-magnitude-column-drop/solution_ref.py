import numpy as np


def pca_vs_naive_mse(X: np.ndarray, k: int):
    """
    Compare two rank-k feature reductions of X by their reconstruction
    MSE against the original X:

    - PCA: mean-center, keep the top-k principal components (via SVD),
      project and reconstruct.
    - Naive: keep the k raw columns with the largest L2 norm, zero out
      every other column entirely.

    Returns (mse_pca, mse_naive).
    """
    X = np.asarray(X, dtype=np.float64)

    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k].T
    Z = Xc @ Vk
    Xhat_pca = Z @ Vk.T + mean
    mse_pca = float(np.mean((X - Xhat_pca) ** 2))

    norms = np.linalg.norm(X, axis=0)
    order = np.argsort(-norms, kind="stable")
    keep = order[:k]
    Xhat_naive = np.zeros_like(X)
    Xhat_naive[:, keep] = X[:, keep]
    mse_naive = float(np.mean((X - Xhat_naive) ** 2))

    return mse_pca, mse_naive
