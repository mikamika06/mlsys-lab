import numpy as np


def q4_0_dequantize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for start in range(0, x.size, 32):
        block = x[start:start + 32]
        max_abs = 0.0
        for i in range(block.size):
            val = block[i]
            abs_val = -val if val < 0.0 else val
            if abs_val > max_abs:
                max_abs = abs_val
        scale = max_abs / 7.0
        if scale == 0:
            for i in range(block.size):
                out[start + i] = 0.0
        else:
            for i in range(block.size):
                val = block[i] / scale
                rounded = round(val)
                clipped = -8.0 if rounded < -8.0 else (7.0 if rounded > 7.0 else rounded)
                out[start + i] = clipped * scale
    return out
