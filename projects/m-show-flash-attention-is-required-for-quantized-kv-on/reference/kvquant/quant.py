import numpy as np


def quantize_q8_0(x: np.ndarray, block_size: int = 32) -> dict:
    """Symmetric Q8_0 quantization in blocks along the trailing dimension."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, block_size)
    max_vals = np.max(np.abs(x_flat), axis=-1, keepdims=True)
    scales = max_vals / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    qdata = np.round(x_flat / scales).astype(np.int8)
    return {"qdata": qdata, "scales": scales, "orig_shape": orig_shape, "block_size": block_size}


def dequantize_q8_0(qdict: dict) -> np.ndarray:
    """Dequantize Q8_0 block dictionary back to float32."""
    qdata = qdict["qdata"].astype(np.float32)
    scales = qdict["scales"].astype(np.float32)
    x_flat = qdata * scales
    return x_flat.reshape(qdict["orig_shape"])


def max_abs_error_bound(x: np.ndarray, block_size: int = 32) -> float:
    """Returns upper bound on maximum absolute quantization error per element."""
    x_flat = x.reshape(-1, block_size)
    max_vals = np.max(np.abs(x_flat), axis=-1)
    return float(np.max(max_vals / 254.0))
