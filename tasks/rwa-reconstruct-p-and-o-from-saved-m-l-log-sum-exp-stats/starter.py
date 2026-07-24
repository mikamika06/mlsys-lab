import numpy as np


def flash_forward_reconstruct(Q, K, V, m, l):
    """
    Q: (n, d), K: (k, d), V: (k, d_v).
    m: (n,) saved per-row max used for the original exponentiation.
    l: (n,) saved per-row normalizer, consistent with that same m:
       l[i] = sum_j exp(S[i, j] - m[i]) for S = Q @ K.T / sqrt(d).

    Reconstruct P[i, j] = exp(S[i, j] - m[i]) / l[i] using the SUPPLIED
    m and l (do not recompute your own row max/sum from S), and
    O = P @ V.

    Returns (P, O) as float64 arrays of shape (n, k) and (n, d_v).
    """
    raise NotImplementedError('your code here')
