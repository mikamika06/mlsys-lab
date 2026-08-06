import numpy as np
import math

def masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked softmax that is safe against fully padded query rows."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    
    n, m = scores_arr.shape
    out = np.zeros((n, m), dtype=np.float64)
    
    for i in range(n):
        has_kept = False
        row_max = float('-inf')
        
        for j in range(m):
            if mask_arr[i, j]:
                has_kept = True
                val = float(scores_arr[i, j])
                if val > row_max:
                    row_max = val
                    
        if not has_kept:
            continue
            
        row_sum = 0.0
        for j in range(m):
            if mask_arr[i, j]:
                val = math.exp(float(scores_arr[i, j]) - row_max)
                out[i, j] = val
                row_sum += val
                
        if row_sum > 0.0:
            for j in range(m):
                if mask_arr[i, j]:
                    out[i, j] /= row_sum
                    
    return out
