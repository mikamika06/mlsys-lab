import numpy as np

NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float64)


def nf4_quantize_dequantize(w: np.ndarray, block_size: int = 64):
    w = np.asarray(w, dtype=np.float64)
    n = w.shape[0]
    nb = n // block_size
    
    idx_list = []
    deq_list = []
    
    for i in range(nb):
        block = w[i * block_size : (i + 1) * block_size]
        
        max_val = 0.0
        for val in block:
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_val:
                max_val = abs_val
                
        absmax_safe = 1.0 if max_val == 0.0 else max_val
        
        for val in block:
            normalized = val / absmax_safe
            
            best_idx = 0
            min_diff = float("inf")
            for j in range(16):
                diff = normalized - NF4_LEVELS[j]
                abs_diff = diff if diff >= 0.0 else -diff
                if abs_diff < min_diff:
                    min_diff = abs_diff
                    best_idx = j
                    
            idx_list.append(best_idx)
            deq_list.append(NF4_LEVELS[best_idx] * absmax_safe)
            
    return np.array(idx_list, dtype=np.uint8), np.array(deq_list, dtype=np.float64)
