import numpy as np

def cov_eig_vs_svd_pca(X: np.ndarray, k: int):
    """Broken implementation that uses the raw data matrix instead of centering,
and normalizes by n_samples rather than (n-1).  This leads to incorrect
eigenvectors for ill‑conditioned matrices."""
    raise NotImplementedError('your code here')
