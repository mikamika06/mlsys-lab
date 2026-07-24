import numpy as np


def fused_tile_pipeline(A: np.ndarray, B: np.ndarray, C: np.ndarray, tile_size: int):
    """
    D = (A + B) * C; E = relu(D) - A; F = sum(E), computed tile-by-tile:
    each contiguous chunk of `tile_size` elements is fully processed
    (add, relu, subtract, partial-sum) before moving to the next chunk.
    """
    n = A.shape[0]
    E = np.empty(n, dtype=np.float64)
    F = 0.0
    for start in range(0, n, tile_size):
        end = min(start + tile_size, n)
        a = A[start:end]
        b = B[start:end]
        c = C[start:end]
        d = (a + b) * c
        e = np.maximum(d, 0.0) - a
        E[start:end] = e
        F += float(np.sum(e))
    return E, F
