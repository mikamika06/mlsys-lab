import math
import numpy as np


def tiled_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    """
    Non-causal full attention O = softmax(Q K^T) V, computed by streaming
    K/V in blocks of `block_size` rows while maintaining a per-query running
    max `m`, running denominator `l`, and running numerator accumulator `acc`
    (the standard online-softmax / FlashAttention forward recurrence).

    Q : (n_q, d)
    K : (n_k, d)
    V : (n_k, d_v)
    returns O : (n_q, d_v), float64
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n_q = Q.shape[0]
    d = Q.shape[1]
    d_v = V.shape[1]
    n_k = K.shape[0]

    m = np.full(n_q, -float("inf"), dtype=np.float64)
    l = np.zeros(n_q, dtype=np.float64)
    acc = np.zeros((n_q, d_v), dtype=np.float64)

    for start in range(0, n_k, block_size):
        end = min(start + block_size, n_k)
        K_blk = K[start:end]
        V_blk = V[start:end]
        bs = end - start

        scores = np.empty((n_q, bs), dtype=np.float64)
        for i in range(n_q):
            for j in range(bs):
                s = 0.0
                for c in range(d):
                    s += Q[i, c] * K_blk[j, c]
                scores[i, j] = s

        block_max = np.empty(n_q, dtype=np.float64)
        for i in range(n_q):
            mx = -float("inf")
            for j in range(bs):
                if scores[i, j] > mx:
                    mx = scores[i, j]
            block_max[i] = mx

        new_m = np.empty(n_q, dtype=np.float64)
        for i in range(n_q):
            if m[i] > block_max[i]:
                new_m[i] = m[i]
            else:
                new_m[i] = block_max[i]

        alpha = np.empty(n_q, dtype=np.float64)
        for i in range(n_q):
            alpha[i] = math.exp(m[i] - new_m[i])

        p = np.empty((n_q, bs), dtype=np.float64)
        for i in range(n_q):
            for j in range(bs):
                p[i, j] = math.exp(scores[i, j] - new_m[i])

        p_sum = np.empty(n_q, dtype=np.float64)
        for i in range(n_q):
            s = 0.0
            for j in range(bs):
                s += p[i, j]
            p_sum[i] = s

        for i in range(n_q):
            l[i] = l[i] * alpha[i] + p_sum[i]

        p_matmul_v = np.zeros((n_q, d_v), dtype=np.float64)
        for i in range(n_q):
            for j in range(d_v):
                s = 0.0
                for k in range(bs):
                    s += p[i, k] * V_blk[k, j]
                p_matmul_v[i, j] = s

        for i in range(n_q):
            for j in range(d_v):
                acc[i, j] = acc[i, j] * alpha[i] + p_matmul_v[i, j]

        for i in range(n_q):
            m[i] = new_m[i]

    out = np.empty((n_q, d_v), dtype=np.float64)
    for i in range(n_q):
        inv_l = 1.0 / l[i]
        for j in range(d_v):
            out[i, j] = acc[i, j] * inv_l

    return out
