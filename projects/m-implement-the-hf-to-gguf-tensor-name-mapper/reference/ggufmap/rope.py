import numpy as np


def undo_rope_permutation(w: np.ndarray, n_heads: int) -> np.ndarray:
    """Undo GGUF RoPE permutation on tensor w to restore HuggingFace layout."""
    shape = w.shape
    total_dim = shape[0]
    if total_dim % n_heads != 0:
        raise ValueError(f"Total dimension {total_dim} not divisible by n_heads {n_heads}")
    head_dim = total_dim // n_heads
    w_3d = w.reshape(n_heads, head_dim, -1)
    half_dim = head_dim // 2
    out_3d = np.zeros_like(w_3d)
    out_3d[:, 0::2, :] = w_3d[:, :half_dim, :]
    out_3d[:, 1::2, :] = w_3d[:, half_dim:, :]
    return out_3d.reshape(shape)
