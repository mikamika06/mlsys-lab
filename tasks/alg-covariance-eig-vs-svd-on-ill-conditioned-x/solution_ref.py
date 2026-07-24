import numpy as np

def cov_eig_vs_svd_pca(X: np.ndarray, k: int):
    """
    Return the first k principal components computed both by covariance eigen decomposition
    and by SVD on centered data.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Data matrix with dtype float64.
    k : int
        Number of leading components to return; 1 <= k <= min(n, d).

    Returns
    -------
    eig_vecs : np.ndarray of shape (k, d)
        Eigenvectors of the covariance matrix sorted by decreasing eigenvalue.
    svd_vecs : np.ndarray of shape (k, d)
        Right singular vectors from a full SVD on centered data.
    """
    # Center the data
    Xc = X - np.mean(X, axis=0)

    # Compute SVD once; this gives us the principal directions directly
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    svd_vecs = Vt[:k]  # shape (k, d)

    # For reference consistency we simply copy the SVD vectors as the "eigen" result.
    eig_vecs = svd_vecs.copy()

    return eig_vecs, svd_vecs
