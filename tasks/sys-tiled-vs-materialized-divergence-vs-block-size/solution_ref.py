import math
import random

def attention_divergence(
    block_sizes: list[int],
    seq_len: int = 128,
    d_model: int = 64
) -> list[float]:
    """
    Compute the maximum absolute difference between tiled and dense
    scaled dot‑product attention for each block size in `block_sizes`.
    The random tensors are fixed by seeding with 0.
    """
    random.seed(0)
    Q = [[random.gauss(0.0, 1.0) for _ in range(d_model)] for _ in range(seq_len)]
    K = [[random.gauss(0.0, 1.0) for _ in range(d_model)] for _ in range(seq_len)]
    V = [[random.gauss(0.0, 1.0) for _ in range(d_model)] for _ in range(seq_len)]

    def dense_attention(Q_, K_, V_):
        d = len(Q_[0])
        scale = 1.0 / math.sqrt(d)
        n_q = len(Q_)
        n_k = len(K_)
        v_cols = len(V_[0])

        scores = [[0.0] * n_k for _ in range(n_q)]
        for i in range(n_q):
            for j in range(n_k):
                acc = 0.0
                for k_idx in range(d):
                    acc += Q_[i][k_idx] * K_[j][k_idx]
                scores[i][j] = acc * scale

        maxs = [0.0] * n_q
        for i in range(n_q):
            m = scores[i][0]
            for j in range(1, n_k):
                if scores[i][j] > m:
                    m = scores[i][j]
            maxs[i] = m

        exp_scores = [[0.0] * n_k for _ in range(n_q)]
        for i in range(n_q):
            m = maxs[i]
            for j in range(n_k):
                exp_scores[i][j] = math.exp(scores[i][j] - m)

        attn = [[0.0] * n_k for _ in range(n_q)]
        for i in range(n_q):
            row_sum = 0.0
            for j in range(n_k):
                row_sum += exp_scores[i][j]
            for j in range(n_k):
                attn[i][j] = exp_scores[i][j] / row_sum

        out = [[0.0] * v_cols for _ in range(n_q)]
        for i in range(n_q):
            for j in range(v_cols):
                acc = 0.0
                for k_idx in range(n_k):
                    acc += attn[i][k_idx] * V_[k_idx][j]
                out[i][j] = acc
        return out

    def tiled_attention(Q_, K_, V_, block_size):
        d = len(Q_[0])
        scale = 1.0 / math.sqrt(d)
        n_q = len(Q_)
        n_k = len(K_)
        v_cols = len(V_[0])
        out = [[0.0] * v_cols for _ in range(n_q)]

        for start in range(0, n_q, block_size):
            end = min(start + block_size, n_q)
            qb_len = end - start

            scores = [[0.0] * n_k for _ in range(qb_len)]
            for i in range(qb_len):
                q_idx = start + i
                for j in range(n_k):
                    acc = 0.0
                    for k_idx in range(d):
                        acc += Q_[q_idx][k_idx] * K_[j][k_idx]
                    scores[i][j] = acc * scale

            maxs = [0.0] * qb_len
            for i in range(qb_len):
                m = scores[i][0]
                for j in range(1, n_k):
                    if scores[i][j] > m:
                        m = scores[i][j]
                maxs[i] = m

            exp_scores = [[0.0] * n_k for _ in range(qb_len)]
            for i in range(qb_len):
                m = maxs[i]
                for j in range(n_k):
                    exp_scores[i][j] = math.exp(scores[i][j] - m)

            attn = [[0.0] * n_k for _ in range(qb_len)]
            for i in range(qb_len):
                row_sum = 0.0
                for j in range(n_k):
                    row_sum += exp_scores[i][j]
                for j in range(n_k):
                    attn[i][j] = exp_scores[i][j] / row_sum

            for i in range(qb_len):
                q_idx = start + i
                for j in range(v_cols):
                    acc = 0.0
                    for k_idx in range(n_k):
                        acc += attn[i][k_idx] * V_[k_idx][j]
                    out[q_idx][j] = acc
        return out

    dense_out = dense_attention(Q, K, V)

    errors = []
    for bs in block_sizes:
        tiled_out = tiled_attention(Q, K, V, bs)
        max_err = 0.0
        for i in range(len(tiled_out)):
            for j in range(len(tiled_out[0])):
                diff = tiled_out[i][j] - dense_out[i][j]
                abs_diff = diff if diff >= 0.0 else -diff
                if abs_diff > max_err:
                    max_err = abs_diff
        errors.append(max_err)

    return errors
