import numpy as np


def blockwise_quantize_dequantize(x: np.ndarray, block_size: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)

    for start in range(0, len(x), block_size):
        end = min(start + block_size, len(x))
        block = x[start:end]
        scale = np.max(np.abs(block)) / 127.0
        if scale == 0:
            out[start:end] = 0.0
        else:
            q = np.clip(np.rint(block / scale), -127, 127).astype(np.int8)
            out[start:end] = q.astype(np.float64) * scale

    return out
