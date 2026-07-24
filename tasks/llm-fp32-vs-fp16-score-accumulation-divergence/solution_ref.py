import numpy as np


def _score_matrix_fp16(q, k):
    n, d = q.shape
    m = k.shape[0]
    scores = np.empty((n, m), dtype=np.float16)
    for i in range(n):
        for j in range(m):
            acc = np.float16(0)
            for t in range(d):
                acc = np.float16(acc + np.float16(q[i, t] * k[j, t]))
            scores[i, j] = acc
    return scores


def attention_fp16_scores(q, k, v):
    d = q.shape[1]
    scores = _score_matrix_fp16(q, k).astype(np.float64)
    scores = scores / np.sqrt(d)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ v.astype(np.float64)
