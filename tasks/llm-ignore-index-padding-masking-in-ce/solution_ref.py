import math
import numpy as np

def masked_cross_entropy(logits: np.ndarray,
                         targets: np.ndarray,
                         ignore_index: int = -100) -> float:
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    
    n_samples = logits.shape[0]
    n_classes = logits.shape[1]
    
    total_loss = 0.0
    valid_count = 0
    
    for i in range(n_samples):
        target = targets[i]
        if target == ignore_index:
            continue
            
        max_val = float(logits[i, 0])
        for j in range(1, n_classes):
            val = float(logits[i, j])
            if val > max_val:
                max_val = val
                
        sum_exp = 0.0
        for j in range(n_classes):
            sum_exp += math.exp(float(logits[i, j]) - max_val)
            
        log_prob = float(logits[i, target]) - max_val - math.log(sum_exp)
        total_loss += -log_prob
        valid_count += 1
        
    if valid_count == 0:
        return 0.0
        
    return float(total_loss / valid_count)
