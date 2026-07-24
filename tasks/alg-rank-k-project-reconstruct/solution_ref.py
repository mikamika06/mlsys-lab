import numpy as np

def rank_k_project_reconstruct(X: np.ndarray, k: int) -> np.ndarray:
    """
    Return the reconstruction of X using its top‑k principal components.
    The input is centred before computing the SVD and the mean is added back
    after reconstruction.  The output has dtype float64 and shape (n,d).
    """
    # Centre the data
    Xc = X - np.mean(X, axis=0)
    # Compute full SVD; we only need the first k right singular vectors
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k, :].T  # shape (d, k)
    # Project onto subspace and reconstruct
    Y = Xc @ Vk          # n x k
    X_hat = Y @ Vk.T + np.mean(X, axis=0)  # add mean back
    return X_hat.astype(np.float64)
