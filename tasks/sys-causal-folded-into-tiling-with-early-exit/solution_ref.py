import numpy as np


def tiled_causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                            tile_q: int, tile_kv: int, on_tile=None) -> np.ndarray:
    """Tiled causal attention that skips fully-future KV tiles entirely.

    See task.md for the exact skip rule and masking. Calls
    on_tile(qi, kj) exactly once per visited (non-skipped) tile pair.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    S, d = Q.shape
    n_q_tiles = S // tile_q
    n_kv_tiles = S // tile_kv

    scores = np.full((S, S), -np.inf)
    for qi in range(n_q_tiles):
        q0, q1 = qi * tile_q, (qi + 1) * tile_q
        for kj in range(n_kv_tiles):
            k0, k1 = kj * tile_kv, (kj + 1) * tile_kv
            if k0 > q1 - 1:
                continue  # fully-future KV tile: skip entirely
            if on_tile is not None:
                on_tile(qi, kj)
            block = (Q[q0:q1] @ K[k0:k1].T) / np.sqrt(d)
            q_idx = np.arange(q0, q1)[:, None]
            k_idx = np.arange(k0, k1)[None, :]
            block = np.where(k_idx > q_idx, -np.inf, block)
            scores[q0:q1, k0:k1] = block

    m = np.max(scores, axis=1, keepdims=True)
    e = np.exp(scores - m)
    probs = e / np.sum(e, axis=1, keepdims=True)
    return probs @ V
