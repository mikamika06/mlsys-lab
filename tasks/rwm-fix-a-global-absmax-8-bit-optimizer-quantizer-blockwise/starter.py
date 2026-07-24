import numpy as np


def blockwise_quantize_dequantize(x: np.ndarray, block_size: int) -> np.ndarray:
    # TODO: incorrectly uses one scale for the whole tensor.
    # This loses precision for values near zero when a local spike exists.
    x = np.asarray(x, dtype=np.float64)

    scale = np.max(np.abs(x)) / 127.0
    if scale == 0:
        return np.zeros_like(x, dtype=np.float64)

    q = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return q.astype(np.float64) * scale
