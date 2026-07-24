import numpy as np

def compute_activation_norms(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X.astype(np.float64), axis=0)
