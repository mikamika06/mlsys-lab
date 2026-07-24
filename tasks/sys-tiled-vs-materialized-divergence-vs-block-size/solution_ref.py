import numpy as np

def attention_divergence(
    block_sizes: list[int],
    seq_len: int = 128,
    d_model: int = 64
) -> np.ndarray:
    """
    Compute the maximum absolute difference between tiled and dense
    scaled dot‑product attention for each block size in `block_sizes`.
    The random tensors are fixed by seeding with 0.
    """
    np.random.seed(0)
    Q = np.random.randn(seq_len, d_model).astype(np.float64)
    K = np.random.randn(seq_len, d_model).astype(np.float64)
    V = np.random.randn(seq_len, d_model).astype(np.float64)

    def dense_attention(Q_, K_, V_):
        d = Q_.shape[1]
        scores = Q_ @ K_.T / np.sqrt(d)
        maxs = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - maxs)
        attn = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return attn @ V_

    def tiled_attention(Q_, K_, V_, block_size):
        d = Q_.shape[1]
        out = np.zeros_like(V_)
        for start in range(0, Q_.shape[0], block_size):
            end = min(start + block_size, Q_.shape[0])
            Qb = Q_[start:end]
            scores = Qb @ K_.T / np.sqrt(d)
            maxs = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores - maxs)
            attn = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            out[start:end] = attn @ V_
        return out

    dense_out = dense_attention(Q, K, V)

    errors = []
    for bs in block_sizes:
        tiled_out = tiled_attention(Q, K, V, bs)
        err = np.max(np.abs(tiled_out - dense_out))
        errors.append(err)

    return np.array(errors, dtype=np.float64)
