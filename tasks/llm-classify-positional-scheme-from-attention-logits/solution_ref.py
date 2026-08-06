import numpy as np
import math

def classify_positional_scheme(S: np.ndarray) -> str:
    n = S.shape[0]
    
    max_val = S[0, 0]
    min_val = S[0, 0]
    for i in range(n):
        for j in range(n):
            val = S[i, j]
            if val > max_val:
                max_val = val
            if val < min_val:
                min_val = val
                
    if max_val - min_val < 1e-4:
        return "none"
        
    is_toeplitz = True
    for i in range(n - 1):
        for j in range(n - 1):
            if abs(S[i, j] - S[i+1, j+1]) > 1e-4:
                is_toeplitz = False
                break
        if not is_toeplitz:
            break
            
    if not is_toeplitz:
        return "sinusoidal"
        
    def is_linear(arr):
        length = len(arr)
        if length < 3:
            return True
        diffs = [arr[k+1] - arr[k] for k in range(length - 1)]
        d0 = diffs[0]
        max_abs_diff = 0.0
        for k in range(len(diffs)):
            val = abs(diffs[k] - d0)
            if val > max_abs_diff:
                max_abs_diff = val
        return max_abs_diff < 1e-4
        
    if is_linear(S[0, :]) and is_linear(S[:, 0]):
        return "alibi"
    else:
        return "rope"
