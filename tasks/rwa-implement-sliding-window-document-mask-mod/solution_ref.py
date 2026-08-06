import math
import numpy as np


def sliding_window_document_attention(Q, K, V, doc_ids, window):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    doc_ids = np.asarray(doc_ids)

    n, d = Q.shape
    _, d_v = V.shape

    mask = np.zeros((n, n), dtype=bool)
    logits = np.zeros((n, n), dtype=np.float64)
    scaled_d = math.sqrt(d)

    for i in range(n):
        for j in range(n):
            cond = (j <= i) and ((i - j) < window) and (doc_ids[i] == doc_ids[j])
            mask[i, j] = cond
            if cond:
                dot_val = 0.0
                for k_dim in range(d):
                    dot_val += Q[i, k_dim] * K[j, k_dim]
                logits[i, j] = dot_val / scaled_d
            else:
                logits[i, j] = -float("inf")

    probs = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        max_val = -float("inf")
        for j in range(n):
            if mask[i, j]:
                if logits[i, j] > max_val:
                    max_val = logits[i, j]

        row_sum = 0.0
        for j in range(n):
            if mask[i, j]:
                val = math.exp(logits[i, j] - max_val)
                probs[i, j] = val
                row_sum += val

        if row_sum > 0.0:
            for j in range(n):
                if mask[i, j]:
                    probs[i, j] /= row_sum

    out = np.zeros((n, d_v), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            p = probs[i, j]
            if p != 0.0:
                for k_dim in range(d_v):
                    out[i, k_dim] += p * V[j, k_dim]

    return out, mask
