import numpy as np
import math

def stable_softmax(x):
    rows, cols = x.shape
    out = np.zeros((rows, cols), dtype=float)
    
    for i in range(rows):
        max_x = x[i, 0]
        for j in range(1, cols):
            if x[i, j] > max_x:
                max_x = x[i, j]
                
        sum_exp = 0.0
        for j in range(cols):
            exp_val = math.exp(x[i, j] - max_x)
            out[i, j] = exp_val
            sum_exp += exp_val
            
        for j in range(cols):
            out[i, j] = out[i, j] / sum_exp
            
    return out
