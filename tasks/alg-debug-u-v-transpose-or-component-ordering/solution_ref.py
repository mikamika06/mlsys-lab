import numpy as np


def pca_projection(X: np.ndarray, k: int) -> np.ndarray:
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:k].T
