import numpy as np
import math

def compute_activation_norms(X: np.ndarray) -> np.ndarray:
    X_f64 = X.astype(np.float64)
    rows = X_f64.shape[0]
    cols = X_f64.shape[1]
    
    result = np.zeros(cols, dtype=np.float64)
    
    for j in range(cols):
        acc = 0.0
        for i in range(rows):
            val = X_f64[i, j]
            acc += val * val
        result[j] = math.sqrt(acc)
        
    return result
