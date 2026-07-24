import numpy as np

def classify_k_for_variance_target(eigenvalues: np.ndarray, target: float) -> int:
    total = np.sum(eigenvalues)
    cum = np.cumsum(eigenvalues)
    ratio = cum / total
    idx = np.searchsorted(ratio, target, side='left')
    return int(idx + 1)
