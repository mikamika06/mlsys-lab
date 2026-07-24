import numpy as np


def _attention(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    scores = q @ k.T / np.sqrt(q.shape[-1])
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights @ v


def scheduled_attention(layers, Qs, Ks, Vs):
    outputs = []
    cache = [None, None]

    for i, q in enumerate(Qs):
        slot = i % 2
        cache[slot] = layers[i]
        k, v = cache[slot]
        outputs.append(_attention(q, k, v))

    return outputs
