import math
import numpy as np

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, bool)
    
    shape = logits.shape
    out = np.empty(shape, dtype=np.float64)
    
    if len(shape) == 1:
        n = shape[0]
        max_val = -float('inf')
        for i in range(n):
            val = -float('inf') if mask[i] else logits[i]
            if val > max_val:
                max_val = val
        
        sum_val = 0.0
        for i in range(n):
            val = -float('inf') if mask[i] else logits[i]
            out[i] = math.exp(val - max_val)
            sum_val += out[i]
            
        for i in range(n):
            out[i] /= sum_val
    else:
        batch_size = shape[0]
        seq_len = shape[1]
        for b in range(batch_size):
            max_val = -float('inf')
            for i in range(seq_len):
                val = -float('inf') if mask[b, i] else logits[b, i]
                if val > max_val:
                    max_val = val
            
            sum_val = 0.0
            for i in range(seq_len):
                val = -float('inf') if mask[b, i] else logits[b, i]
                out[b, i] = math.exp(val - max_val)
                sum_val += out[b, i]
                
            for i in range(seq_len):
                out[b, i] /= sum_val
                
    return out
