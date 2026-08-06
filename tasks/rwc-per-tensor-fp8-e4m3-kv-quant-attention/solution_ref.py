import math
import numpy as np

def _quantize_dequant(x):
    amax = 0.0
    flat_x = x.ravel()
    for val in flat_x:
        abs_val = val if val >= 0.0 else -val
        if abs_val > amax:
            amax = abs_val
    scale = amax / 448.0 if amax != 0.0 else 1.0
    
    q = np.empty_like(x, dtype=np.int8)
    res = np.empty_like(x, dtype=np.float32)
    
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        val = x[idx]
        rounded = round(val / scale)
        clipped = rounded
        if clipped < -127:
            clipped = -127
        elif clipped > 127:
            clipped = 127
        q[idx] = int(clipped)
        res[idx] = float(q[idx]) * scale
        it.iternext()
    return res

def quantized_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    K_dq = _quantize_dequant(K)
    V_dq = _quantize_dequant(V)
    d_k = Q.shape[-1]
    sqrt_d_k = math.sqrt(float(d_k))
    
    n_q = Q.shape[0]
    n_k = K_dq.shape[0]
    d_v = V_dq.shape[1]
    
    scores = np.empty((n_q, n_k), dtype=Q.dtype)
    for i in range(n_q):
        for j in range(n_k):
            s = 0.0
            for k_idx in range(d_k):
                s += Q[i, k_idx] * K_dq[j, k_idx]
            scores[i, j] = s / sqrt_d_k
            
    e = np.empty_like(scores)
    for i in range(n_q):
        max_val = scores[i, 0]
        for j in range(1, n_k):
            if scores[i, j] > max_val:
                max_val = scores[i, j]
        
        row_sum = 0.0
        for j in range(n_k):
            val = math.exp(scores[i, j] - max_val)
            e[i, j] = val
            row_sum += val
            
        for j in range(n_k):
            e[i, j] /= row_sum
            
    attn = np.empty((n_q, d_v), dtype=Q.dtype)
    for i in range(n_q):
        for j in range(d_v):
            s = 0.0
            for k_idx in range(n_k):
                s += e[i, k_idx] * V_dq[k_idx, j]
            attn[i, j] = s
            
    return attn
