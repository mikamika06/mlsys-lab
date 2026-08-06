import numpy as np


def tree_sum(values: np.ndarray) -> np.float32:
    current = np.asarray(values, dtype=np.float32).copy()

    while current.size > 1:
        if current.size % 2 == 1:
            padded = np.empty(current.size + 1, dtype=np.float32)
            for i in range(current.size):
                padded[i] = current[i]
            padded[current.size] = np.float32(0.0)
            current = padded
        
        new_size = current.size // 2
        next_current = np.empty(new_size, dtype=np.float32)
        for i in range(new_size):
            next_current[i] = np.float32(current[2 * i] + current[2 * i + 1])
        current = next_current

    if current.size == 0:
        return np.float32(0.0)
    return np.float32(current[0])
