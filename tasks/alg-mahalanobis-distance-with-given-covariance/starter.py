import numpy as np

def pairwise_mahalanobis(X: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    # TODO: Implement fast pairwise Mahalanobis distance
    return np.zeros((X.shape[0], X.shape[0]))
