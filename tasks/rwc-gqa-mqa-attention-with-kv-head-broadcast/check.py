import numpy as np

def _softmax(x, axis=-1):
    """Numerically stable softmax along *axis*."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def _oracle_gqa_attention(Q, K, V):
    """Oracle: expand KV heads via np.repeat, then standard attention."""
    batch, seq_q, n_heads, head_dim = Q.shape
    n_kv_heads = K.shape[2]
    group_size = n_heads // n_kv_heads
    scale = head_dim ** -0.5

    # Expand K, V: each of the n_kv heads is repeated group_size times
    K_exp = np.repeat(K, group_size, axis=2)   # (B, T_k, n_h, d_h)
    V_exp = np.repeat(V, group_size, axis=2)

    # Rearrange to (batch, n_heads, seq, head_dim) for batched matmul
    Q_h = Q.transpose(0, 2, 1, 3)
    K_h = K_exp.transpose(0, 2, 1, 3)
    V_h = V_exp.transpose(0, 2, 1, 3)

    # Scaled attention scores: (batch, n_heads, seq_q, seq_k)
    scores = (Q_h @ K_h.swapaxes(-2, -1)) * scale

    # Softmax over key dimension
    weights = _softmax(scores, axis=-1)

    # Weighted sum of values: (batch, n_heads, seq_q, head_dim)
    out = weights @ V_h

    # Back to (batch, seq_q, n_heads, head_dim)
    return out.transpose(0, 2, 1, 3)

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(42)

    # (batch, seq_q, seq_k, n_heads, n_kv_heads, head_dim)
    configs = [
        (2, 8, 8, 8, 2, 16),    # GQA, group_size = 4
        (1, 4, 4, 4, 1, 8),     # MQA
        (2, 6, 10, 4, 4, 16),   # MHA, cross-attention (T_q != T_k)
        (1, 3, 3, 6, 3, 32),    # GQA, group_size = 2
        (3, 5, 5, 8, 1, 8),     # MQA, larger batch
    ]

    worst_err = 0.0

    for batch, seq_q, seq_k, n_heads, n_kv_heads, head_dim in configs:
        Q = rng.randn(batch, seq_q, n_heads, head_dim)
        K = rng.randn(batch, seq_k, n_kv_heads, head_dim)
        V = rng.randn(batch, seq_k, n_kv_heads, head_dim)

        try:
            student_out = sol.gqa_attention(Q, K, V)
        except Exception:
            return {"max_abs_err": 1e10}

        oracle_out = _oracle_gqa_attention(Q, K, V)

        if student_out.shape != oracle_out.shape:
            return {"max_abs_err": 1e10}

        err = float(np.max(np.abs(student_out.astype(np.float64) - oracle_out)))
        worst_err = max(worst_err, err)

    return {"max_abs_err": worst_err}
