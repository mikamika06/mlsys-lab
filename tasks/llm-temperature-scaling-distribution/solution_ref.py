import math
import numpy as np

def temperature_scale(logits, T):
    logits = np.asarray(logits, dtype=np.float64)
    shape = logits.shape
    N = shape[-1]
    
    total_elements = 1
    for s in shape:
        total_elements *= s
    num_slices = total_elements // N
    
    probs = np.empty(shape, dtype=np.float64)
    logits_flat = np.reshape(logits, -1)
    probs_flat = np.reshape(probs, -1)
    
    for slice_idx in range(num_slices):
        offset = slice_idx * N
        
        max_val = logits_flat[offset] / T
        for j in range(1, N):
            val = logits_flat[offset + j] / T
            if val > max_val:
                max_val = val
                
        sum_exp = 0.0
        for j in range(N):
            val = logits_flat[offset + j] / T
            exp_val = math.exp(val - max_val)
            probs_flat[offset + j] = exp_val
            sum_exp += exp_val
            
        for j in range(N):
            probs_flat[offset + j] /= sum_exp
            
    return probs
