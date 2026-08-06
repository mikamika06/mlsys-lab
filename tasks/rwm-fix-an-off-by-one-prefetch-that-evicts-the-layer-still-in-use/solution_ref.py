import math
import numpy as np


def _attention(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    d_k = q.shape[-1]
    scale = math.sqrt(d_k)
    
    q_rows = q.shape[0]
    k_rows = k.shape[0]
    v_cols = v.shape[1]
    
    scores = np.empty((q_rows, k_rows), dtype=np.float64)
    for i in range(q_rows):
        for j in range(k_rows):
            acc = 0.0
            for d in range(d_k):
                acc += q[i, d] * k[j, d]
            scores[i, j] = acc / scale
            
    max_scores = np.empty((q_rows, 1), dtype=np.float64)
    for i in range(q_rows):
        m = scores[i, 0]
        for j in range(1, k_rows):
            if scores[i, j] > m:
                m = scores[i, j]
        max_scores[i, 0] = m
        
    weights = np.empty((q_rows, k_rows), dtype=np.float64)
    for i in range(q_rows):
        for j in range(k_rows):
            weights[i, j] = math.exp(scores[i, j] - max_scores[i, 0])
            
    sum_weights = np.empty((q_rows, 1), dtype=np.float64)
    for i in range(q_rows):
        s = 0.0
        for j in range(k_rows):
            s += weights[i, j]
        sum_weights[i, 0] = s
        
    for i in range(q_rows):
        for j in range(k_rows):
            weights[i, j] /= sum_weights[i, 0]
            
    out = np.empty((q_rows, v_cols), dtype=np.float64)
    for i in range(q_rows):
        for c in range(v_cols):
            acc = 0.0
            for j in range(k_rows):
                acc += weights[i, j] * v[j, c]
            out[i, c] = acc
            
    return out


def scheduled_attention(layers, Qs, Ks, Vs):
    outputs = []
    cache = [None, None]

    for i, q in enumerate(Qs):
        slot = i % 2
        cache[slot] = layers[i]
        k, v = cache[slot]
        outputs.append(_attention(q, k, v))

    return outputs
