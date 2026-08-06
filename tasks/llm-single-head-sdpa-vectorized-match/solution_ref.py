import math
import numpy as np

def sdpa_single_head(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    
    seq_len, d_head = Q.shape
    d_v = V.shape[1]
    
    scale = math.sqrt(d_head)
    
    scores = np.zeros((seq_len, seq_len), dtype=np.float64)
    for i in range(seq_len):
        for j in range(seq_len):
            acc = 0.0
            for k in range(d_head):
                acc += Q[i, k] * K[j, k]
            scores[i, j] = acc / scale
            
    softmax = np.zeros((seq_len, seq_len), dtype=np.float64)
    for i in range(seq_len):
        max_val = float('-inf')
        for j in range(seq_len):
            if scores[i, j] > max_val:
                max_val = scores[i, j]
        
        sum_exp = 0.0
        for j in range(seq_len):
            e_val = math.exp(scores[i, j] - max_val)
            softmax[i, j] = e_val
            sum_exp += e_val
            
        for j in range(seq_len):
            softmax[i, j] = softmax[i, j] / sum_exp
            
    out = np.zeros((seq_len, d_v), dtype=np.float64)
    for i in range(seq_len):
        for j in range(d_v):
            acc = 0.0
            for k in range(seq_len):
                acc += softmax[i, k] * V[k, j]
            out[i, j] = acc
            
    return out
