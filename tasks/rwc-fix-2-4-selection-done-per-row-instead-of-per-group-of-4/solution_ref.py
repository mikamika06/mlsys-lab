import numpy as np


def select_2_4_mask(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W)
    rows, cols = W.shape
    out = np.zeros((rows, cols), dtype=np.int64)
    for r in range(rows):
        for start in range(0, cols, 4):
            group = W[r, start:start + 4]
            order = sorted(range(4), key=lambda i: (-abs(float(group[i])), i))
            for idx in order[:2]:
                out[r, start + idx] = 1
    return out
