import math
import numpy as np


def online_softmax_stream(x: np.ndarray, B: int) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]

    m = -float('inf')
    s = 0.0
    peak_elements = 2

    for start in range(0, n, B):
        block = x[start:start + B]
        block_size = block.shape[0]
        peak_elements = max(peak_elements, block_size + 2)

        block_max = -float('inf')
        for i in range(block_size):
            val = block[i]
            if val > block_max:
                block_max = val

        new_m = m if m > block_max else block_max
        
        s_term = 0.0
        for i in range(block_size):
            s_term += math.exp(block[i] - new_m)
            
        s = s * math.exp(m - new_m) + s_term
        m = new_m

    out = np.empty_like(x)
    peak_elements = max(peak_elements, B + 2)

    for start in range(0, n, B):
        block = x[start:start + B]
        block_size = block.shape[0]
        for i in range(block_size):
            out[start + i] = math.exp(block[i] - m) / s

    return out, peak_elements
