import numpy as np

def compute_group_params(weights: np.ndarray,
                         group_size: int = 64,
                         bits: int = 4) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=np.float64)
    n_groups = len(w) // group_size
    denom = (2 ** bits) - 1
    scales = []
    biases = []
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size
        min_val = w[start]
        max_val = w[start]
        for j in range(start + 1, end):
            val = w[j]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
        scales.append((max_val - min_val) / denom)
        biases.append(min_val)
    return np.asarray(scales, dtype=np.float64), np.asarray(biases, dtype=np.float64)
