import numpy as np


def decode_step(k_cache, v_cache, q, k_new, v_new):
    """One autoregressive decode step with an O(1) KV-cache append.

    The new key/value are appended as a single row each. The row copy runs in
    C inside ``np.vstack``, so the Python-level work is a small constant that is
    independent of the cache length S -- never an O(S) row-by-row rebuild.
    """
    k_cache = np.asarray(k_cache, dtype=np.float64)
    v_cache = np.asarray(v_cache, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d = q.shape[0]

    # append exactly one new row to each cache
    k_cache = np.vstack((k_cache, np.asarray(k_new, dtype=np.float64).reshape(1, d)))
    v_cache = np.vstack((v_cache, np.asarray(v_new, dtype=np.float64).reshape(1, d)))

    # single-query attention over the whole (grown) cache
    scores = (q @ k_cache.T) / np.sqrt(d)
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / np.sum(weights)
    out = weights @ v_cache

    return out, k_cache, v_cache
