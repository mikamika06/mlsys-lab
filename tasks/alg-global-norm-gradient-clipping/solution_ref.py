import math
import numpy as np
from typing import List

def clip_global_norm(grads: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
    total_norm_sq = 0.0
    for g in grads:
        for i in range(g.size):
            val = g.flat[i]
            total_norm_sq += val * val
            
    total_norm = math.sqrt(total_norm_sq)
    coef = min(1.0, max_norm / (total_norm + 1e-6))
    
    clipped_grads = []
    for g in grads:
        new_g = np.empty(g.shape, dtype=g.dtype)
        for i in range(g.size):
            new_g.flat[i] = g.flat[i] * coef
        clipped_grads.append(new_g)
        
    return clipped_grads
