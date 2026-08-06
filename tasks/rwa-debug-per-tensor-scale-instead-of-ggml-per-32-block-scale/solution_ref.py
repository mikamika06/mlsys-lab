import numpy as np


def q4_0_dequantize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for start in range(0, x.size, 32):
        block = x[start:start + 32]
        scale = np.max(np.abs(block)) / 7.0
        if scale == 0:
            out[start:start + 32] = 0.0
        else:
            q = np.clip(np.round(block / scale), -8, 7)
            out[start:start + 32] = q * scale
    return out
