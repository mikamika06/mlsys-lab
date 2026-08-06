import math
import numpy as np


def slot_attention(K_cache, V_cache, Q, positions, B):
    K_cache = np.asarray(K_cache, dtype=np.float64)
    V_cache = np.asarray(V_cache, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)

    block_idx = positions // B
    block_offset = positions % B

    K = K_cache[block_idx, block_offset]
    V = V_cache[block_idx, block_offset]

    T, H, D = Q.shape
    L = K.shape[0]
    out = np.empty((T, H, D), dtype=np.float64)
    scale = 1.0 / math.sqrt(D)

    for i in range(T):
        scores = [[0.0] * L for _ in range(H)]
        for h in range(H):
            for l in range(L):
                s = 0.0
                for d in range(D):
                    s += Q[i, h, d] * K[l, h, d]
                scores[h][l] = s * scale

        for h in range(H):
            max_val = scores[h][0]
            for l in range(1, L):
                if scores[h][l] > max_val:
                    max_val = scores[h][l]
            for l in range(L):
                scores[h][l] -= max_val

        probs = [[0.0] * L for _ in range(H)]
        for h in range(H):
            for l in range(L):
                probs[h][l] = math.exp(scores[h][l])

        for h in range(H):
            sum_val = 0.0
            for l in range(L):
                sum_val += probs[h][l]
            for l in range(L):
                probs[h][l] /= sum_val

        for h in range(H):
            for d in range(D):
                val = 0.0
                for l in range(L):
                    val += probs[h][l] * V[l, h, d]
                out[i, h, d] = val

    return out
