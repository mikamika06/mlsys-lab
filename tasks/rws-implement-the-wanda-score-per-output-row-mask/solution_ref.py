import math
import numpy as np


def wanda_score_mask(W: np.ndarray, X: np.ndarray, sparsity: float) -> np.ndarray:
    d_out, d_in = W.shape
    
    col_norm = np.zeros(d_in, dtype=X.dtype)
    for j in range(d_in):
        acc = 0.0
        for i in range(X.shape[0]):
            val = X[i, j]
            acc += val * val
        col_norm[j] = math.sqrt(acc)
    
    S = np.zeros((d_out, d_in), dtype=W.dtype)
    for i in range(d_out):
        for j in range(d_in):
            w_val = W[i, j]
            abs_w = w_val if w_val >= 0.0 else -w_val
            S[i, j] = abs_w * col_norm[j]
            
    k = max(1, int(round((1.0 - sparsity) * d_in)))
    
    mask = np.zeros((d_out, d_in), dtype=bool)
    for i in range(d_out):
        row_vals = []
        for j in range(d_in):
            row_vals.append((-S[i, j], j))
        
        indexed_vals = []
        for idx, item in enumerate(row_vals):
            indexed_vals.append((item[0], item[1], idx))
            
        def sort_key(x):
            return (x[0], x[2])
            
        sorted_vals = sorted(indexed_vals, key=sort_key)
        
        for rank in range(k):
            original_col = sorted_vals[rank][1]
            mask[i, original_col] = True
            
    return mask
