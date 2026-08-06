import numpy as np
import math

def mean_cosine_similarity(in_states: np.ndarray,
                           out_states: np.ndarray) -> float:
    n_rows = in_states.shape[0]
    n_cols = in_states.shape[1]
    
    cos_sum = 0.0
    for i in range(n_rows):
        in_norm_sq = 0.0
        out_norm_sq = 0.0
        dot_prod = 0.0
        for j in range(n_cols):
            in_val = in_states[i, j]
            out_val = out_states[i, j]
            in_norm_sq += in_val * in_val
            out_norm_sq += out_val * out_val
            dot_prod += in_val * out_val
            
        in_norm = math.sqrt(in_norm_sq)
        out_norm = math.sqrt(out_norm_sq)
        cos_sum += dot_prod / (in_norm * out_norm)
        
    return float(cos_sum / n_rows)
