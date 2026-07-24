import numpy as np

def gqa_attention(Q, K, V):
    """Compute grouped-query attention.

    Args:
        Q: np.ndarray, shape (batch, seq_q, n_heads, head_dim)
        K: np.ndarray, shape (batch, seq_k, n_kv_heads, head_dim)
        V: np.ndarray, shape (batch, seq_k, n_kv_heads, head_dim)

    Returns:
        np.ndarray, shape (batch, seq_q, n_heads, head_dim), dtype float64
    """
    batch, seq_q, n_heads, head_dim = Q.shape
    n_kv_heads = K.shape[2]
    group_size = n_heads // n_kv_heads
    scale = head_dim ** -0.5

    # Expand K, V: repeat each KV head group_size times along axis 2
    K_exp = np.repeat(K, group_size, axis=2)   # (batch, seq_k, n_heads, head_dim)
    V_exp = np.repeat(V, group_size, axis=2)

    # Transpose to (batch, n_heads, seq, head_dim) for batched matmul
    Q_h = Q.transpose(0, 2, 1, 3)
    K_h = K_exp.transpose(0, 2, 1, 3)
    V_h = V_exp.transpose(0, 2, 1, 3)

    # Scaled dot-product scores: (batch, n_heads, seq_q, seq_k)
    scores = (Q_h @ K_h.swapaxes(-2, -1)) * scale

    # Numerically stable softmax over key dimension
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted sum: (batch, n_heads, seq_q, head_dim)
    out = weights @ V_h

    # Back to (batch, seq_q, n_heads, head_dim)
    return out.transpose(0, 2, 1, 3)
