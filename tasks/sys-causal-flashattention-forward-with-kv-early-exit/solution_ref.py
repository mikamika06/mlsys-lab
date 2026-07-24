import numpy as np


def _score_kv_tile(q_block, k_block, v_block, q_start, k_start, tile_size):
    return q_block @ k_block.T / np.sqrt(float(q_block.shape[1])), v_block


def causal_flash_attention_forward(Q, K, V, tile_size=2):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    dv = V.shape[1]
    out = np.zeros((n, dv), dtype=np.float64)
    lse = np.zeros(n, dtype=np.float64)

    for q_start in range(0, n, tile_size):
        q_end = min(n, q_start + tile_size)
        q_block = Q[q_start:q_end]
        scores_parts = []
        values_parts = []

        for k_start in range(0, n, tile_size):
            if k_start > q_end - 1:
                break
            k_end = min(n, k_start + tile_size)
            s, v = _score_kv_tile(
                q_block,
                K[k_start:k_end],
                V[k_start:k_end],
                q_start,
                k_start,
                tile_size,
            )
            rows = np.arange(q_start, q_end)[:, None]
            cols = np.arange(k_start, k_end)[None, :]
            s = np.where(cols <= rows, s, -np.inf)
            scores_parts.append(s)
            values_parts.append(v)

        scores = np.concatenate(scores_parts, axis=1)
        vals = np.concatenate(values_parts, axis=0)
        m = np.max(scores, axis=1)
        e = np.exp(scores - m[:, None])
        p = e / np.sum(e, axis=1, keepdims=True)
        out[q_start:q_end] = p @ vals
        lse[q_start:q_end] = np.log(np.sum(e, axis=1)) + m

    return out, lse
