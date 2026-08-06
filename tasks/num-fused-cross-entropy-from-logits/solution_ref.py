import math
import numpy as np


def fused_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-example cross-entropy loss ell_i = logsumexp(logits[i]) - logits[i, targets[i]],
    computed via the numerically-stable log-sum-exp trick (fully vectorised)."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    
    num_rows = logits.shape[0]
    num_cols = logits.shape[1]
    
    result = np.zeros(num_rows, dtype=np.float64)
    
    for i in range(num_rows):
        max_val = logits[i, 0]
        for j in range(1, num_cols):
            if logits[i, j] > max_val:
                max_val = logits[i, j]
                
        sum_exp = 0.0
        for j in range(num_cols):
            sum_exp += math.exp(logits[i, j] - max_val)
            
        lse = max_val + math.log(sum_exp)
        tgt_logit = logits[i, targets[i]]
        result[i] = lse - tgt_logit
        
    return result
