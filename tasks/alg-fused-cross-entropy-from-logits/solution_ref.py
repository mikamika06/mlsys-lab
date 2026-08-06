def fused_cross_entropy(logits, targets):
    import numpy as np
    import math
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    
    N = logits.shape[0]
    C = logits.shape[1]
    
    total_ce = 0.0
    
    for i in range(N):
        row_max = logits[i, 0]
        for j in range(1, C):
            if logits[i, j] > row_max:
                row_max = logits[i, j]
                
        exp_sum = 0.0
        for j in range(C):
            exp_sum += math.exp(logits[i, j] - row_max)
            
        logsumexp = math.log(exp_sum) + row_max
        target_val = logits[i, targets[i]]
        
        ce = - (target_val - logsumexp)
        total_ce += ce
        
    return float(total_ce / N)
