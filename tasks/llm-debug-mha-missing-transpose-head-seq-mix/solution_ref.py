import itertools
import math
import numpy as np


def _softmax(x):
    shape = x.shape
    out = np.empty(shape, dtype=x.dtype)
    n = shape[-1]
    batch_shape = shape[:-1]
    for idx in itertools.product(*(range(d) for d in batch_shape)):
        max_val = x[idx + (0,)]
        for i in range(1, n):
            val = x[idx + (i,)]
            if val > max_val:
                max_val = val
        exps = [math.exp(x[idx + (i,)] - max_val) for i in range(n)]
        sum_e = 0.0
        for e in exps:
            sum_e += e
        for i in range(n):
            out[idx + (i,)] = exps[i] / sum_e
    return out


def mha_forward(X, Wq, Wk, Wv, Wo, num_heads):
    B, S, E = X.shape
    d = E // num_heads

    q = np.empty((B, num_heads, S, d), dtype=X.dtype)
    k = np.empty((B, num_heads, S, d), dtype=X.dtype)
    v = np.empty((B, num_heads, S, d), dtype=X.dtype)

    for b in range(B):
        for h in range(num_heads):
            for s in range(S):
                for i in range(d):
                    e_idx = h * d + i
                    val_q = 0.0
                    val_k = 0.0
                    val_v = 0.0
                    for e in range(E):
                        x_val = X[b, s, e]
                        val_q += x_val * Wq[e, e_idx]
                        val_k += x_val * Wk[e, e_idx]
                        val_v += x_val * Wv[e, e_idx]
                    q[b, h, s, i] = val_q
                    k[b, h, s, i] = val_k
                    v[b, h, s, i] = val_v

    scale = math.sqrt(d)
    scores = np.empty((B, num_heads, S, S), dtype=X.dtype)
    for b in range(B):
        for h in range(num_heads):
            for s1 in range(S):
                for s2 in range(S):
                    val = 0.0
                    for i in range(d):
                        val += q[b, h, s1, i] * k[b, h, s2, i]
                    scores[b, h, s1, s2] = val / scale

    weights = _softmax(scores)

    out_concat = np.empty((B, S, E), dtype=X.dtype)
    for b in range(B):
        for s1 in range(S):
            for h in range(num_heads):
                for i in range(d):
                    val = 0.0
                    for s2 in range(S):
                        val += weights[b, h, s1, s2] * v[b, h, s2, i]
                    out_concat[b, s1, h * d + i] = val

    out = np.empty((B, S, E), dtype=X.dtype)
    for b in range(B):
        for s in range(S):
            for e2 in range(E):
                val = 0.0
                for e in range(E):
                    val += out_concat[b, s, e] * Wo[e, e2]
                out[b, s, e2] = val

    return out
