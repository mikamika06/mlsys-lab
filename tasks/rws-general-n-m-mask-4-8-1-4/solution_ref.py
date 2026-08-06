import numpy as np


def nm_mask(weights: np.ndarray, N: int, M: int) -> np.ndarray:
    weights = np.asarray(weights)
    mask = np.zeros(weights.shape, dtype=np.int8)
    for start in range(0, len(weights), M):
        group = weights[start:start + M]
        order = sorted(range(len(group)), key=lambda i: (-abs(group[i]), i))
        for idx in order[:N]:
            mask[start + idx] = 1
    return mask
