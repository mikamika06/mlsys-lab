import numpy as np


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """Split (T, D) into (num_heads, T, D // num_heads) with the head axis first.

    out[h, t, j] = x[t, h * head_dim + j]

    Uses explicit index arithmetic only -- no reshape / transpose / swapaxes.
    """
    x = np.asarray(x, dtype=np.float64)
    seq_len, dim = x.shape
    head_dim = dim // num_heads
    out = np.empty((num_heads, seq_len, head_dim), dtype=np.float64)
    for h in range(num_heads):
        base = h * head_dim
        for t in range(seq_len):
            for j in range(head_dim):
                out[h, t, j] = x[t, base + j]
    return out


def merge_heads(heads: np.ndarray) -> np.ndarray:
    """Inverse of split_heads: (num_heads, T, head_dim) -> (T, num_heads * head_dim).

    out[t, h * head_dim + j] = heads[h, t, j]

    Uses explicit index arithmetic only -- no reshape / transpose / swapaxes.
    """
    heads = np.asarray(heads, dtype=np.float64)
    num_heads, seq_len, head_dim = heads.shape
    out = np.empty((seq_len, num_heads * head_dim), dtype=np.float64)
    for h in range(num_heads):
        base = h * head_dim
        for t in range(seq_len):
            for j in range(head_dim):
                out[t, base + j] = heads[h, t, j]
    return out
