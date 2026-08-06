import numpy as np


def blockwise_quantize_dequantize(x: np.ndarray, block_size: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)

    for start in range(0, len(x), block_size):
        end = min(start + block_size, len(x))
        block = x[start:end]
        max_abs = 0.0
        for val in block:
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_abs:
                max_abs = abs_val
        scale = max_abs / 127.0
        if scale == 0.0:
            for i in range(start, end):
                out[i] = 0.0
        else:
            for i in range(start, end):
                q = round(x[i] / scale)
                if q < -127:
                    q = -127
                elif q > 127:
                    q = 127
                out[i] = float(q) * scale

    return out
