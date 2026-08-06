import numpy as np


def compute_diversity_bound(samples, threshold=0.1):
    cov = np.cov(samples, rowvar=False)
    if cov.ndim == 0:
        return float(cov) >= threshold
    eigenvalues = np.linalg.eigvalsh(cov)
    min_eigenvalue = float(np.min(eigenvalues))
    return min_eigenvalue >= threshold
