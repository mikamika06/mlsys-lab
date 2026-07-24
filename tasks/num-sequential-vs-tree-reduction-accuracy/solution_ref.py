import numpy as np


def tree_sum(values: np.ndarray) -> np.float32:
    current = np.asarray(values, dtype=np.float32).copy()

    while current.size > 1:
        if current.size % 2 == 1:
            current = np.concatenate(
                [current, np.zeros(1, dtype=np.float32)]
            )
        current = current.reshape(-1, 2).sum(axis=1, dtype=np.float32)

    if current.size == 0:
        return np.float32(0.0)
    return np.float32(current[0])
