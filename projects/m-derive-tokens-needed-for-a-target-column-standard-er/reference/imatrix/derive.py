import numpy as np


def compute_required_tokens(variance: float, target_se: float) -> int:
    if target_se <= 0:
        raise ValueError("target_se must be positive")
    if variance < 0:
        raise ValueError("variance cannot be negative")
    n = np.ceil(variance / (target_se ** 2))
    return int(n)
