import math
import numpy as np


def select_min_angle_block(query: np.ndarray, candidates: np.ndarray) -> int:
    q = np.asarray(query, dtype=np.float64)
    C = np.asarray(candidates, dtype=np.float64)
    num_candidates = C.shape[0]
    dim = C.shape[1]
    
    sum_q_sq = 0.0
    for j in range(dim):
        sum_q_sq += q[j] * q[j]
    norm_q = math.sqrt(sum_q_sq)
    
    best_idx = 0
    min_dist = float('inf')
    
    for i in range(num_candidates):
        dots_i = 0.0
        norm_c_i_sq = 0.0
        for j in range(dim):
            val_c = C[i, j]
            val_q = q[j]
            dots_i += val_c * val_q
            norm_c_i_sq += val_c * val_c
            
        denom_i = norm_q * math.sqrt(norm_c_i_sq)
        cos_i = dots_i / denom_i
        
        if cos_i > 1.0:
            cos_i = 1.0
        elif cos_i < -1.0:
            cos_i = -1.0
            
        dist_i = math.acos(cos_i)
        
        if dist_i < min_dist:
            min_dist = dist_i
            best_idx = i
            
    return int(best_idx)
