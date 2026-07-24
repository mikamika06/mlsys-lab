import numpy as np

def sq_dist_expansion(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, a) + np.dot(b, b) - 2.0 * np.dot(a, b))
