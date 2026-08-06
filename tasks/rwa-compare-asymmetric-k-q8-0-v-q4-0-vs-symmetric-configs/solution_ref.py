import math
import numpy as np


def _quantize_symmetric(x, bits):
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1
    
    max_val = 0.0
    for val in x.flat:
        abs_val = val if val >= 0.0 else -val
        if abs_val > max_val:
            max_val = abs_val
            
    scale = max_val / qmax
    if scale == 0:
        return np.zeros_like(x)
        
    out = np.empty_like(x)
    for i in range(x.size):
        out.flat[i] = round(x.flat[i] / scale) * scale
    return out


def _attention(K, V, q):
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    n_rows = K.shape[0]
    n_cols = K.shape[1]
    
    logits = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        s = 0.0
        for j in range(n_cols):
            s += K[i, j] * q[j]
        logits[i] = s / math.sqrt(n_cols)
        
    max_logit = logits[0]
    for i in range(1, logits.size):
        if logits[i] > max_logit:
            max_logit = logits[i]
            
    p = np.empty(n_rows, dtype=np.float64)
    sum_p = 0.0
    for i in range(n_rows):
        val = math.exp(logits[i] - max_logit)
        p[i] = val
        sum_p += val
        
    for i in range(n_rows):
        p[i] /= sum_p
        
    v_cols = V.shape[1]
    result = np.zeros(v_cols, dtype=np.float64)
    for j in range(v_cols):
        s = 0.0
        for i in range(n_rows):
            s += p[i] * V[i, j]
        result[j] = s
        
    return result


def kv_config_attention_errors(K, V, q):
    base = _attention(np.asarray(K, dtype=np.float64), np.asarray(V, dtype=np.float64), q)
    result = []
    for kb, vb in [(8, 8), (4, 4), (8, 4)]:
        kq = _quantize_symmetric(K, kb)
        vq = _quantize_symmetric(V, vb)
        att = _attention(kq, vq, q)
        max_diff = 0.0
        for i in range(att.size):
            diff = att.flat[i] - base.flat[i]
            abs_diff = diff if diff >= 0.0 else -diff
            if abs_diff > max_diff:
                max_diff = abs_diff
        result.append(float(max_diff))
    return np.asarray(result, dtype=np.float64)
