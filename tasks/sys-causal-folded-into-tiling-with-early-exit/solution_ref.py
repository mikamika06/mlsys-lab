import math
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

    scores = [[float('-inf')] * S for _ in range(S)]
    sqrt_d = math.sqrt(d)

    for qi in range(n_q_tiles):
        q0, q1 = qi * tile_q, (qi + 1) * tile_q
        for kj in range(n_kv_tiles):
            k0, k1 = kj * tile_kv, (kj + 1) * tile_kv
            if k0 > q1 - 1:
                continue
            if on_tile is not None:
                on_tile(qi, kj)
            for i_local in range(tile_q):
                i = q0 + i_local
                for j_local in range(tile_kv):
                    j = k0 + j_local
                    if j > i:
                        scores[i][j] = float('-inf')
                    else:
                        dot = 0.0
                        for dim in range(d):
                            dot += Q[i, dim] * K[j, dim]
                        scores[i][j] = dot / sqrt_d

    m = []
    for i in range(S):
        row_max = float('-inf')
        for j in range(S):
            if scores[i][j] > row_max:
                row_max = scores[i][j]
        m.append(row_max)

    e = [[0.0] * S for _ in range(S)]
    for i in range(S):
        row_m = m[i]
        for j in range(S):
            val = scores[i][j]
            if val == float('-inf'):
                e[i][j] = 0.0
            else:
                e[i][j] = math.exp(val - row_m)

    sum_e = []
    for i in range(S):
        s_val = 0.0
        for j in range(S):
            s_val += e[i][j]
        sum_e.append(s_val)

    probs = [[0.0] * S for _ in range(S)]
    for i in range(S):
        s_val = sum_e[i]
        for j in range(S):
            if s_val == 0.0:
                probs[i][j] = 0.0
            else:
                probs[i][j] = e[i][j] / s_val

    result = [[0.0] * d for _ in range(S)]
    for i in range(S):
        for col in range(d):
            val = 0.0
            for k in range(S):
                val += probs[i][k] * V[k, col]
            result[i][col] = val

    return np.array(result, dtype=np.float64)
