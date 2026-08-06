import math
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
        scale = 1.0 / math.sqrt(d)
        n_q = Q_.shape[0]
        n_k = K_.shape[0]
        v_cols = V_.shape[1]

        scores = np.empty((n_q, n_k), dtype=np.float64)
        for i in range(n_q):
            for j in range(n_k):
                acc = 0.0
                for k_idx in range(d):
                    acc += Q_[i, k_idx] * K_[j, k_idx]
                scores[i, j] = acc * scale

        maxs = np.empty((n_q, 1), dtype=np.float64)
        for i in range(n_q):
            m = scores[i, 0]
            for j in range(1, n_k):
                if scores[i, j] > m:
                    m = scores[i, j]
            maxs[i, 0] = m

        exp_scores = np.empty((n_q, n_k), dtype=np.float64)
        for i in range(n_q):
            m = maxs[i, 0]
            for j in range(n_k):
                exp_scores[i, j] = math.exp(scores[i, j] - m)

        attn = np.empty((n_q, n_k), dtype=np.float64)
        for i in range(n_q):
            row_sum = 0.0
            for j in range(n_k):
                row_sum += exp_scores[i, j]
            for j in range(n_k):
                attn[i, j] = exp_scores[i, j] / row_sum

        out = np.empty((n_q, v_cols), dtype=np.float64)
        for i in range(n_q):
            for j in range(v_cols):
                acc = 0.0
                for k_idx in range(n_k):
                    acc += attn[i, k_idx] * V_[k_idx, j]
                out[i, j] = acc
        return out

    def tiled_attention(Q_, K_, V_, block_size):
        d = Q_.shape[1]
        scale = 1.0 / math.sqrt(d)
        n_q = Q_.shape[0]
        n_k = K_.shape[0]
        v_cols = V_.shape[1]
        out = np.zeros_like(V_)

        for start in range(0, n_q, block_size):
            end = min(start + block_size, n_q)
            qb_len = end - start

            scores = np.empty((qb_len, n_k), dtype=np.float64)
            for i in range(qb_len):
                q_idx = start + i
                for j in range(n_k):
                    acc = 0.0
                    for k_idx in range(d):
                        acc += Q_[q_idx, k_idx] * K_[j, k_idx]
                    scores[i, j] = acc * scale

            maxs = np.empty((qb_len, 1), dtype=np.float64)
            for i in range(qb_len):
                m = scores[i, 0]
                for j in range(1, n_k):
                    if scores[i, j] > m:
                        m = scores[i, j]
                maxs[i, 0] = m

            exp_scores = np.empty((qb_len, n_k), dtype=np.float64)
            for i in range(qb_len):
                m = maxs[i, 0]
                for j in range(n_k):
                    exp_scores[i, j] = math.exp(scores[i, j] - m)

            attn = np.empty((qb_len, n_k), dtype=np.float64)
            for i in range(qb_len):
                row_sum = 0.0
                for j in range(n_k):
                    row_sum += exp_scores[i, j]
                for j in range(n_k):
                    attn[i, j] = exp_scores[i, j] / row_sum

            for i in range(qb_len):
                q_idx = start + i
                for j in range(v_cols):
                    acc = 0.0
                    for k_idx in range(n_k):
                        acc += attn[i, k_idx] * V_[k_idx, j]
                    out[q_idx, j] = acc
        return out

    dense_out = dense_attention(Q, K, V)

    errors = []
    for bs in block_sizes:
        tiled_out = tiled_attention(Q, K, V, bs)
        max_err = 0.0
        for i in range(tiled_out.shape[0]):
            for j in range(tiled_out.shape[1]):
                diff = tiled_out[i, j] - dense_out[i, j]
                abs_diff = diff if diff >= 0.0 else -diff
                if abs_diff > max_err:
                    max_err = abs_diff
        errors.append(max_err)

    return np.array(errors, dtype=np.float64)
