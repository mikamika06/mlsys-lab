import math
import numpy as np

def soft_expected_mask(logits: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    patterns = np.asarray(patterns, dtype=np.float64)
    
    shape_logits = logits.shape
    shape_patterns = patterns.shape
    
    batch_dims = shape_logits[:-1]
    d_in = shape_logits[-1]
    d_out = shape_patterns[-1]
    
    total_batches = 1
    for dim in batch_dims:
        total_batches *= dim
        
    logits_flat = logits.reshape(total_batches, d_in)
    
    out_flat = np.zeros((total_batches, d_out), dtype=np.float64)
    
    for i in range(total_batches):
        row = logits_flat[i]
        
        max_val = row[0]
        for j in range(1, d_in):
            if row[j] > max_val:
                max_val = row[j]
                
        exp_row = np.empty(d_in, dtype=np.float64)
        sum_exp = 0.0
        for j in range(d_in):
            val = math.exp(row[j] - max_val)
            exp_row[j] = val
            sum_exp += val
            
        probs = np.empty(d_in, dtype=np.float64)
        for j in range(d_in):
            probs[j] = exp_row[j] / sum_exp
            
        for k in range(d_out):
            acc = 0.0
            for j in range(d_in):
                acc += probs[j] * patterns[j, k]
            out_flat[i, k] = acc
            
    return out_flat.reshape((*batch_dims, d_out))
