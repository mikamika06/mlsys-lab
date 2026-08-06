import math
import numpy as np


def sliding_window_attention_tiled(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window: int, block_size: int) -> np.ndarray:
    """
    Sliding-window causal attention, computed tile by tile over the query
    axis. For each query tile [qs, qe), only the key/value slice
    [max(0, qs - window + 1), qe) is ever touched -- the full (n, n) mask
    or score matrix is never materialized.

    Q, K, V: (n, d) float64.
    window: query i attends to keys max(0, i-window+1) .. i.
    block_size: number of query rows processed per tile.

    Returns: (n, d) float64 attention output.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    out = np.empty((n, d), dtype=np.float64)

    for qs in range(0, n, block_size):
        qe = min(n, qs + block_size)
        k_lo = max(0, qs - window + 1)
        k_hi = qe

        Q_tile = Q[qs:qe]
        K_tile = K[k_lo:k_hi]
        V_tile = V[k_lo:k_hi]

        num_q = qe - qs
        num_k = k_hi - k_lo

        scores = np.empty((num_q, num_k), dtype=np.float64)
        for i in range(num_q):
            for j in range(num_k):
                dot = 0.0
                for k_idx in range(d):
                    dot += Q_tile[i, k_idx] * K_tile[j, k_idx]
                scores[i, j] = dot * scale

        masked = np.empty((num_q, num_k), dtype=np.float64)
        for i in range(num_q):
            row_idx = qs + i
            for j in range(num_k):
                col_idx = k_lo + j
                allowed = (col_idx <= row_idx) and ((row_idx - col_idx) < window)
                if allowed:
                    masked[i, j] = scores[i, j]
                else:
                    masked[i, j] = -float('inf')

        for i in range(num_q):
            max_val = masked[i, 0]
            for j in range(1, num_k):
                if masked[i, j] > max_val:
                    max_val = masked[i, j]
            for j in range(num_k):
                masked[i, j] = masked[i, j] - max_val

        e = np.empty((num_q, num_k), dtype=np.float64)
        for i in range(num_q):
            for j in range(num_k):
                e[i, j] = math.exp(masked[i, j])

        p = np.empty((num_q, num_k), dtype=np.float64)
        for i in range(num_q):
            sum_e = 0.0
            for j in range(num_k):
                sum_e += e[i, j]
            for j in range(num_k):
                p[i, j] = e[i, j] / sum_e

        res_tile = np.empty((num_q, d), dtype=np.float64)
        for i in range(num_q):
            for j in range(d):
                val = 0.0
                for k_idx in range(num_k):
                    val += p[i, k_idx] * V_tile[k_idx, j]
                res_tile[i, j] = val

        out[qs:qe] = res_tile

    return out
