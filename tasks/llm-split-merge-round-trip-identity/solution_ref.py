import numpy as np

def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Split the last dimension of x into `num_heads` heads.
    Input shape: (B, T, D)
    Output shape: (B, T, num_heads, D // num_heads)
    """
    batch, seq_len, dim = x.shape
    head_dim = dim // num_heads
    return x.reshape(batch, seq_len, num_heads, head_dim)

def merge_heads(heads: np.ndarray) -> np.ndarray:
    """
    Merge the heads back into a single last dimension.
    Input shape: (B, T, num_heads, D // num_heads)
    Output shape: (B, T, D)
    """
    batch, seq_len, num_heads, head_dim = heads.shape
    return heads.reshape(batch, seq_len, num_heads * head_dim)
