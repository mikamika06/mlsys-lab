import numpy as np


def run_with_cache(Wq, Wk, Wv, X, kv_cache=None):
    """Single-head causal self-attention over new tokens X, using (and
    extending) an optional KV cache built from an already-processed prefix.

    Wq, Wk, Wv : (d, d) float64 projection matrices (no output projection).
    X : (L, d) float64 -- new tokens' input embeddings to process.
    kv_cache : None, or {'K': (P, d), 'V': (P, d)} -- already-projected
        keys/values of P earlier ("prefix") tokens, or None to start fresh.

    Each new token i (0-indexed within X, global position P+i) attends
    causally over ALL keys/values at global positions 0..P+i inclusive:
    the cached prefix plus every new token up to and including itself.
    Queries are always computed fresh (never cached); keys/values for the
    new tokens are computed once and appended to the cache.

    Returns
    -------
    out : (L, d) float64 -- attention output for the L new tokens.
    new_cache : {'K': (P+L, d), 'V': (P+L, d)} -- the extended cache. The
        first P rows are exactly the input kv_cache's rows (untouched).
    """
    Wq = np.asarray(Wq, dtype=np.float64)
    Wk = np.asarray(Wk, dtype=np.float64)
    Wv = np.asarray(Wv, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]

    if kv_cache is None:
        K_prefix = np.zeros((0, d), dtype=np.float64)
        V_prefix = np.zeros((0, d), dtype=np.float64)
    else:
        K_prefix = np.asarray(kv_cache["K"], dtype=np.float64)
        V_prefix = np.asarray(kv_cache["V"], dtype=np.float64)
    P = K_prefix.shape[0]

    Q_new = X @ Wq
    K_new = X @ Wk
    V_new = X @ Wv

    K_all = np.concatenate([K_prefix, K_new], axis=0)
    V_all = np.concatenate([V_prefix, V_new], axis=0)

    L = X.shape[0]
    out = np.zeros((L, d), dtype=np.float64)
    for i in range(L):
        end = P + i + 1  # attend over global positions [0, P+i]
        scores = (Q_new[i] @ K_all[:end].T) / np.sqrt(d)
        scores = scores - np.max(scores)
        w = np.exp(scores)
        w = w / np.sum(w)
        out[i] = w @ V_all[:end]

    new_cache = {"K": K_all, "V": V_all}
    return out, new_cache
