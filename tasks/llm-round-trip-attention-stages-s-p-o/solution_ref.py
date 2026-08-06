import numpy as np
import math

def attention_roundtrip(Q: np.ndarray,
                        K: np.ndarray,
                        V: np.ndarray):
    """
    Correct implementation of scaled dot‑product attention.
    Returns the raw scores S, the softmax probabilities P,
    and the weighted output O.
    """
    n_q, d = Q.shape
    n_k = K.shape[0]
    n_v, d_v = V.shape

    S = np.empty((n_q, n_k), dtype=Q.dtype)
    sqrt_d = math.sqrt(d)
    for i in range(n_q):
        for j in range(n_k):
            acc = 0.0
            for k_idx in range(d):
                acc += Q[i, k_idx] * K[j, k_idx]
            S[i, j] = acc / sqrt_d

    P = np.empty((n_q, n_k), dtype=Q.dtype)
    for i in range(n_q):
        max_val = S[i, 0]
        for j in range(1, n_k):
            if S[i, j] > max_val:
                max_val = S[i, j]

        row_sum = 0.0
        exp_vals = []
        for j in range(n_k):
            val = math.exp(S[i, j] - max_val)
            exp_vals.append(val)
            row_sum += val

        for j in range(n_k):
            P[i, j] = exp_vals[j] / row_sum

    O = np.empty((n_q, d_v), dtype=Q.dtype)
    for i in range(n_q):
        for j in range(d_v):
            acc = 0.0
            for k_idx in range(n_k):
                acc += P[i, k_idx] * V[k_idx, j]
            O[i, j] = acc

    return S, P, O
