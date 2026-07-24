import numpy as np


def dot_error_bound(n: int) -> float:
    eps = np.finfo(np.float32).eps
    return float((n * eps) / (1.0 - n * eps))
