import numpy as np


def block_sparse_attention(Q, K, V, block_mask, block_size):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    blocks = block_mask.shape[0]

    allowed = np.zeros((n, n), dtype=bool)
    for bi in range(blocks):
        for bj in range(blocks):
            if block_mask[bi, bj]:
                r0 = bi * block_size
                c0 = bj * block_size
                allowed[
                    r0:min(n, r0 + block_size),
                    c0:min(n, c0 + block_size),
                ] = True

    scores = (Q @ K.T) / np.sqrt(float(d))
    scores = np.where(allowed, scores, -np.inf)
    scores -= np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)
    output = weights @ V

    active = int(np.sum(block_mask))
    ratio = float((n * n * d) / (active * block_size * block_size * d))
    return output.astype(np.float64), ratio
