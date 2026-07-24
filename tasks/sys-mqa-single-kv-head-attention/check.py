import numpy as np

def _softmax_stable(x, axis=-1):
    """Numerically stable softmax along *axis*."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def _standard_mha(Q, K, V):
    """Standard scaled dot-product attention (all head dims matched)."""
    head_dim = Q.shape[-1]
    scale = head_dim ** -0.5
    # Q: (B, H, S, D)  K: (B, H, S, D) -> scores: (B, H, S, S)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) * scale
    weights = _softmax_stable(scores, axis=-1)
    return np.matmul(weights, V)

def _reference(Q, K_single, V_single):
    """Expand a single KV head to h heads, then run standard MHA."""
    n_heads = Q.shape[1]
    K_exp = np.repeat(K_single, n_heads, axis=1)
    V_exp = np.repeat(V_single, n_heads, axis=1)
    return _standard_mha(Q, K_exp, V_exp)

def grade(sol, fx) -> dict:
    """Grade mha_single_kv_head against the single-head-expanded reference."""
    cases = [
        # (batch, n_heads, seq_len, head_dim)
        (2, 4, 8, 16),
        (1, 8, 16, 32),
        (3, 2, 6, 8),
        (1, 1, 4, 4),      # h=1 edge case
        (2, 6, 12, 24),    # non-power-of-2 head_dim
        (1, 3, 1, 64),     # seq_len=1
    ]

    max_err = 0.0
    for batch, n_heads, seq_len, head_dim in cases:
        rng = np.random.RandomState(42)
        Q = rng.randn(batch, n_heads, seq_len, head_dim).astype(np.float64)
        K = rng.randn(batch, 1, seq_len, head_dim).astype(np.float64)
        V = rng.randn(batch, 1, seq_len, head_dim).astype(np.float64)

        try:
            got = sol.mha_single_kv_head(Q, K, V)
            ref = _reference(Q, K, V)
            err = float(np.max(np.abs(got - ref)))
            max_err = max(max_err, err)
        except Exception:
            max_err = float("inf")
            break

    return {"max_abs_err": max_err}
