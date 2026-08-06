import math
import numpy as np

FP4_MAX = 6.0


def mxfp4_block_exponent(x: np.ndarray, block_size: int = 32) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    nb = n // block_size
    
    out = np.empty(nb, dtype=np.int64)
    
    for i in range(nb):
        start = i * block_size
        amax = 0.0
        for j in range(block_size):
            val = x[start + j]
            if val < 0.0:
                val = -val
            if val > amax:
                amax = val
                
        if amax == 0.0:
            exp = 0.0
        else:
            exp = math.floor(math.log2(amax / FP4_MAX))
            
        out[i] = int(exp)

    return out
