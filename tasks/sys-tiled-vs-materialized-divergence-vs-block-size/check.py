import numpy as np

# Reference implementations used by the grader
def _dense_attention(Q, K, V):
    d = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d)
    maxs = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - maxs)
    attn = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return attn @ V

def _tiled_attention(Q, K, V, block_size):
    d = Q.shape[1]
    out = np.zeros_like(V)
    for start in range(0, Q.shape[0], block_size):
        end = min(start + block_size, Q.shape[0])
        Qb = Q[start:end]
        scores = Qb @ K.T / np.sqrt(d)
        maxs = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - maxs)
        attn = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        out[start:end] = attn @ V
    return out

def _compute_errors(block_sizes):
    np.random.seed(0)
    seq_len = 128
    d_model = 64
    Q = np.random.randn(seq_len, d_model).astype(np.float64)
    K = np.random.randn(seq_len, d_model).astype(np.float64)
    V = np.random.randn(seq_len, d_model).astype(np.float64)

    dense_out = _dense_attention(Q, K, V)
    errors = []
    for bs in block_sizes:
        tiled_out = _tiled_attention(Q, K, V, bs)
        err = np.max(np.abs(tiled_out - dense_out))
        errors.append(err)
    return np.array(errors, dtype=np.float64)

# Block sizes used by the grader
BLOCK_SIZES = [8, 16, 32, 64]

def grade(sol, fx) -> dict:
    try:
        cand = sol.attention_divergence(BLOCK_SIZES)
    except Exception:
        return {"max_abs_err": float("inf")}
    if not isinstance(cand, np.ndarray):
        return {"max_abs_err": float("inf")}
    ref = _compute_errors(BLOCK_SIZES)
    diff = np.max(np.abs(cand - ref))
    return {"max_abs_err": diff}
