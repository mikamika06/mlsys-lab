import math
import numpy as np


def _quantize_col(col, bits):
    qmax = (1 << (bits - 1)) - 1
    rows = col.shape[0]
    
    max_abs = 0.0
    for i in range(rows):
        val = col[i]
        abs_val = val if val >= 0.0 else -val
        if abs_val > max_abs:
            max_abs = abs_val
            
    scale = max_abs / qmax
    if scale == 0.0:
        return np.zeros_like(col)
        
    codes = np.empty_like(col, dtype=np.float64)
    for i in range(rows):
        val = col[i] / scale
        rounded = math.floor(val + 0.5)
        if rounded > qmax:
            rounded = float(qmax)
        elif rounded < -qmax:
            rounded = float(-qmax)
        codes[i] = rounded
        
    result = np.empty_like(col, dtype=np.float64)
    for i in range(rows):
        result[i] = codes[i] * scale
    return result


def gptq_quantize(W: np.ndarray, H_inv: np.ndarray, bits: int = 4) -> np.ndarray:
    work = np.array(W, dtype=np.float64, copy=True)
    result = np.zeros_like(work)
    rows = work.shape[0]
    cols = work.shape[1]

    for j in range(cols):
        current = np.empty(rows, dtype=np.float64)
        for i in range(rows):
            current[i] = work[i, j]
            
        quantized = _quantize_col(current, bits)
        
        for i in range(rows):
            result[i, j] = quantized[i]
            
        error = np.empty(rows, dtype=np.float64)
        for i in range(rows):
            error[i] = quantized[i] - current[i]

        inv_val = H_inv[j, j]
        for k in range(j + 1, cols):
            factor = H_inv[j, k] / inv_val
            for i in range(rows):
                work[i, k] += error[i] * factor

    return result
