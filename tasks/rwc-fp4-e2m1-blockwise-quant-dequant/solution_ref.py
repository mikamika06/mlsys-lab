import numpy as np

def fp4_quant_dequant(x: np.ndarray, block_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize `x` to FP4 e2m1 format in a blockwise manner and return both
    the integer codes (dtype=int8) and the dequantized float64 array.
    """
    flat = x.ravel()
    n = flat.size
    codes = np.empty_like(flat, dtype=np.int8)
    deq = np.empty_like(flat, dtype=np.float64)

    for start in range(0, n, block_size):
        end = start + block_size
        if end > n:
            end = n
        
        max_abs = 0.0
        for i in range(start, end):
            val = flat[i]
            if val < 0.0:
                abs_val = -val
            else:
                abs_val = val
            if abs_val > max_abs:
                max_abs = abs_val
        
        alpha = max_abs
        s = alpha / 7.0 if alpha != 0 else 1.0
        
        for i in range(start, end):
            val = flat[i]
            scaled = val / s
            rounded = round(scaled)
            if rounded < -8.0:
                clipped = -8
            elif rounded > 7.0:
                clipped = 7
            else:
                clipped = int(rounded)
            
            codes[i] = clipped
            deq[i] = float(clipped) * s

    return codes.reshape(x.shape), deq.reshape(x.shape)
