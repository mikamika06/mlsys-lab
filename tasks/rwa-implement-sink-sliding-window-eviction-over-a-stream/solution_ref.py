import numpy as np


def sink_attention_stream(Q, K, V, k, w):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    outputs = []
    lengths = []

    cache = []
    for t in range(n):
        cache.append(t)

        min_recent = max(0, t - w + 1)
        cache = [
            idx for idx in cache
            if idx < k or idx >= min_recent
        ]

        lengths.append(len(cache))

        logits = (K[cache] @ Q[t]) / np.sqrt(d)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        outputs.append(probs @ V[cache])

    return np.asarray(outputs, dtype=np.float64), lengths
