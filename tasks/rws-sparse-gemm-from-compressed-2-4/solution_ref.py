import numpy as np


def compressed_matmul(values: np.ndarray, idx: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Reconstruct the dense weight matrix from NVIDIA-style compressed 2:4
    storage (2 nonzero values per group of 4 columns, plus a 2-bit index
    per value giving its position 0..3 within the group), then compute
    the dense matmul W @ X.
    """
    values = np.asarray(values, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    X = np.asarray(X, dtype=np.float64)

    d_out, half = values.shape
    d_in = half * 2

    row_idx = np.broadcast_to(np.arange(d_out)[:, None], (d_out, half))
    group_base = (np.arange(half) // 2) * 4
    cols = group_base[None, :] + idx

    W = np.zeros((d_out, d_in), dtype=np.float64)
    W[row_idx, cols] = values

    return W @ X
