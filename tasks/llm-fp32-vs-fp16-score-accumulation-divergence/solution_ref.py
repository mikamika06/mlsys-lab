import math
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
    n, d = q.shape
    m = k.shape[0]
    dv = v.shape[1]
    scores_fp16 = _score_matrix_fp16(q, k)
    sqrt_d = math.sqrt(d)
    out = np.empty((n, dv), dtype=np.float64)
    for i in range(n):
        row_scores = [0.0] * m
        max_val = -float("inf")
        for j in range(m):
            val = float(scores_fp16[i, j]) / sqrt_d
            row_scores[j] = val
            if val > max_val:
                max_val = val
        weights = [0.0] * m
        weight_sum = 0.0
        for j in range(m):
            w = math.exp(row_scores[j] - max_val)
            weights[j] = w
            weight_sum += w
        for j in range(m):
            weights[j] /= weight_sum
        for l in range(dv):
            acc = 0.0
            for j in range(m):
                acc += weights[j] * float(v[j, l])
            out[i, l] = acc
    return out
