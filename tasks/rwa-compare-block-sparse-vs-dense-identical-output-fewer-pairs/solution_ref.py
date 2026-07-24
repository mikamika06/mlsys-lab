import numpy as np


def block_sparse_attention(Q, K, V, mask, block_size):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    n = Q.shape[0]
    m = V.shape[1]
    scores = Q @ K.T / np.sqrt(Q.shape[1])

    row_max = np.full(n, -np.inf, dtype=np.float64)
    attended_pairs = 0

    for bi in range(0, n, block_size):
        for bj in range(0, n, block_size):
            block_mask = mask[bi:bi + block_size, bj:bj + block_size]
            if np.any(block_mask):
                attended_pairs += block_mask.shape[0] * block_mask.shape[1]
                block_scores = scores[bi:bi + block_size, bj:bj + block_size]
                for li in range(block_scores.shape[0]):
                    vals = block_scores[li][block_mask[li]]
                    if vals.size:
                        row_max[bi + li] = max(row_max[bi + li], np.max(vals))

    denom = np.zeros(n, dtype=np.float64)
    numer = np.zeros((n, m), dtype=np.float64)

    for bi in range(0, n, block_size):
        for bj in range(0, n, block_size):
            block_mask = mask[bi:bi + block_size, bj:bj + block_size]
            if not np.any(block_mask):
                continue
            block_scores = scores[bi:bi + block_size, bj:bj + block_size]
            block_values = V[bj:bj + block_size]
            for li in range(block_scores.shape[0]):
                allowed = block_mask[li]
                if np.any(allowed):
                    w = np.exp(block_scores[li, allowed] - row_max[bi + li])
                    denom[bi + li] += np.sum(w)
                    numer[bi + li] += w @ block_values[allowed]

    out = np.zeros_like(numer)
    for i in range(n):
        if denom[i] != 0:
            out[i] = numer[i] / denom[i]

    return out, attended_pairs
