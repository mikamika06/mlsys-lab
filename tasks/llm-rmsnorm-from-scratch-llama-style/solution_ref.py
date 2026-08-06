import math
import numpy as np

def rmsnorm(x, weight, eps=1e-6):
    x_arr = np.asarray(x, dtype=np.float64)
    w_arr = np.asarray(weight, dtype=x_arr.dtype)
    n = len(x_arr)
    
    sq_sum = 0.0
    for i in range(n):
        val = float(x_arr[i])
        sq_sum += val * val
        
    mean_sq = sq_sum / n
    denom = math.sqrt(mean_sq + eps)
    
    out = np.empty(x_arr.shape, dtype=x_arr.dtype)
    for i in range(n):
        out[i] = float(w_arr[i]) * (float(x_arr[i]) / denom)
        
    return out
