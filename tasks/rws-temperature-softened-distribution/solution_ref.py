import math
import numpy as np

def softmax_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    n = logits.shape[0]
    
    z = np.empty(n, dtype=np.float64)
    for i in range(n):
        z[i] = logits[i] / T
        
    shift = z[0]
    for i in range(1, n):
        if z[i] > shift:
            shift = z[i]
            
    exp_z = np.empty(n, dtype=np.float64)
    for i in range(n):
        exp_z[i] = math.exp(z[i] - shift)
        
    total_sum = 0.0
    for i in range(n):
        total_sum += exp_z[i]
        
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = exp_z[i] / total_sum
        
    return result
