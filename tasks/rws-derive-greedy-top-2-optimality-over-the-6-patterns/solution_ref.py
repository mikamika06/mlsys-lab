import numpy as np


def greedy_24_prune(W: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(W, dtype=np.float64))
    two_smallest = np.partition(values, 1, axis=1)[:, :2]
    return np.sum(two_smallest, axis=1, dtype=np.float64)
