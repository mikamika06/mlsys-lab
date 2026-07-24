import numpy as np


def chunked_attention(q, chunks):
    # TODO: this incorrectly normalizes every chunk independently.
    q = np.asarray(q, dtype=np.float64)
    outputs = []
    for K, V in chunks:
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        scores = K @ q
        weights = np.exp(scores - np.max(scores))
        weights = weights / np.sum(weights)
        outputs.append(weights @ V)

    return np.mean(outputs, axis=0)
