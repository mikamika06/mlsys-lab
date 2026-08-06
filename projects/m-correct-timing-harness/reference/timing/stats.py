import numpy as np


def calculate_sample_size(mean, std, target_error=0.05, z_score=1.96):
    if mean <= 0:
        raise ValueError("mean must be positive")
    margin = target_error * mean
    n = (z_score * std / margin) ** 2
    return int(np.ceil(n))
