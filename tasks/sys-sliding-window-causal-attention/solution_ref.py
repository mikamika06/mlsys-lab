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
    scale = 1.0 / np.sqrt(d)

    out = np.empty((n, d), dtype=np.float64)

    for qs in range(0, n, block_size):
        qe = min(n, qs + block_size)
        k_lo = max(0, qs - window + 1)
        k_hi = qe

        Q_tile = Q[qs:qe]
        K_tile = K[k_lo:k_hi]
        V_tile = V[k_lo:k_hi]

        scores = (Q_tile @ K_tile.T) * scale

        rows = np.arange(qs, qe).reshape(-1, 1)
        cols = np.arange(k_lo, k_hi).reshape(1, -1)
        allowed = (cols <= rows) & (rows - cols < window)

        masked = np.where(allowed, scores, -np.inf)
        masked = masked - np.max(masked, axis=-1, keepdims=True)
        e = np.exp(masked)
        p = e / np.sum(e, axis=-1, keepdims=True)

        out[qs:qe] = p @ V_tile

    return out
