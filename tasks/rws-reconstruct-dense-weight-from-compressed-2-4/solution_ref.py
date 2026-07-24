import numpy as np


def reconstruct_24(values: np.ndarray, indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    out = np.zeros(shape, dtype=np.float32)
    groups = cols // 4
    for r in range(rows):
        for g in range(groups):
            base = g * 4
            out[r, base + int(indices[r, 2 * g])] = np.float32(values[r, 2 * g])
            out[r, base + int(indices[r, 2 * g + 1])] = np.float32(values[r, 2 * g + 1])
    return out
