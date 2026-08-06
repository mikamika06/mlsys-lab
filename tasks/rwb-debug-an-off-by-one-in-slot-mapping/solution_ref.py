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
    out = np.empty((T, H, D), dtype=np.float64)
    scale = 1.0 / np.sqrt(D)

    for i in range(T):
        scores = np.einsum("hd,lhd->hl", Q[i], K) * scale
        scores -= np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=1, keepdims=True)
        out[i] = np.einsum("hl,lhd->hd", probs, V)
    return out
