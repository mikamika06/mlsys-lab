import math
import numpy as np


def affine_quant_dequant(x: np.ndarray, bits: int) -> np.ndarray:
    """Correct affine quantize-dequantize using (2^bits - 1) denominator."""
    x = np.asarray(x, dtype=np.float64)
    n_levels = (1 << bits) - 1
    
    x_min = x.flat[0]
    x_max = x.flat[0]
    for val in x.flat:
        if val < x_min:
            x_min = val
        if val > x_max:
            x_max = val

    scale = (x_max - x_min) / n_levels if x_max != x_min else 1.0
    
    out = np.empty_asarray(x.shape, dtype=np.float64) if hasattr(np, 'empty_asarray') else np.empty(x.shape, dtype=np.float64)
    
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        val = it[0]
        q = round((val - x_min) / scale)
        if q < 0.0:
            q = 0.0
        elif q > float(n_levels):
            q = float(n_levels)
        out[it.multi_index] = q * scale + x_min
        it.iternext()
        
    return out
