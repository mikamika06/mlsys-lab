import numpy as np


def nm_mask(weights: np.ndarray, N: int, M: int) -> np.ndarray:
    weights = np.asarray(weights)
    mask = np.zeros(weights.shape, dtype=np.int8)
    for start in range(0, len(weights), M):
        group = weights[start:start + M]
        order = np.argsort(-np.abs(group), kind="stable")
        mask[start + order[:N]] = 1
    return mask
