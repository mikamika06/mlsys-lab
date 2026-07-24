import numpy as np


def combined_served_size(weight: np.ndarray, group_size: int) -> float:
    flat = np.asarray(weight).reshape(-1)
    groups = flat.size // group_size
    total = 0.0

    for i in range(groups):
        block = flat[i * group_size:(i + 1) * group_size]
        nnz = int(np.count_nonzero(block))
        total += nnz * 0.5
        total += nnz * 0.25
        total += 2.0

    return float(total)
