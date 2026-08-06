import numpy as np

def reduce_range_accumulator_safety(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per‑column accumulations for full and reduced activation ranges,
    and the maximum intermediate partial sum when using the reduced range.
    All outputs are int32 arrays of shape (C,).
    """
    R, C = X.shape
    
    full_accum = np.zeros(C, dtype=np.int32)
    reduced_accum = np.zeros(C, dtype=np.int32)
    peak_per_col = np.zeros(C, dtype=np.int32)
    
    for c in range(C):
        full_sum = 0
        red_sum = 0
        max_peak = -2147483648
        current_cumsum = 0
        
        for r in range(R):
            val = X[r, c]
            full_sum += int(val)
            
            if val < 63:
                red_val = val
            else:
                red_val = 63
                
            red_sum += int(red_val)
            current_cumsum += int(red_val)
            if current_cumsum > max_peak:
                max_peak = current_cumsum
                
        full_accum[c] = full_sum
        reduced_accum[c] = red_sum
        peak_per_col[c] = max_peak
        
    return full_accum, reduced_accum, peak_per_col
