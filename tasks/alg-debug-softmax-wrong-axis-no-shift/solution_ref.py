import math
import numpy as np
from itertools import product

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shape = x.shape
    ndim = len(shape)
    axis = axis % ndim
    
    out = np.empty(shape, dtype=x.dtype)
    
    ranges_pre = [range(shape[j]) for j in range(axis)]
    ranges_post = [range(shape[j]) for j in range(axis + 1, ndim)]
    axis_len = shape[axis]
    
    for idx_pre in product(*ranges_pre):
        for idx_post in product(*ranges_post):
            max_val = x[idx_pre + (0,) + idx_post]
            for i in range(1, axis_len):
                val = x[idx_pre + (i,) + idx_post]
                if val > max_val:
                    max_val = val
                    
            exp_sum = 0.0
            for i in range(axis_len):
                idx = idx_pre + (i,) + idx_post
                exp_val = math.exp(x[idx] - max_val)
                out[idx] = exp_val
                exp_sum += exp_val
                
            for i in range(axis_len):
                idx = idx_pre + (i,) + idx_post
                out[idx] = out[idx] / exp_sum
                
    return out
