import numpy as np

def acceptance_rate(target, draft):
    target = np.asarray(target, dtype=np.float64)
    draft = np.asarray(draft, dtype=np.float64)
    
    n_rows, n_cols = target.shape
    result = np.zeros(n_rows, dtype=np.float64)
    
    for i in range(n_rows):
        row_sum = 0.0
        for j in range(n_cols):
            t_val = target[i, j]
            d_val = draft[i, j]
            if t_val < d_val:
                row_sum += t_val
            else:
                row_sum += d_val
        result[i] = row_sum
        
    return result
