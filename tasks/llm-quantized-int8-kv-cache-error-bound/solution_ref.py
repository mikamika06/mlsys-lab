import numpy as np

def kv_cache_quantize(keys_fp16: np.ndarray, values_fp16: np.ndarray):
    """
    Quantizes the key tensor to int8 using a per-row symmetric scale.
    
    Parameters
    ----------
    keys_fp16 : np.ndarray of shape (n, d)
        The original key matrix in float16.
    values_fp16 : np.ndarray of shape (n, d)
        The value matrix in float16.  It is passed for API compatibility but
        not used by the quantization routine.

    Returns
    -------
    keys_int8 : np.ndarray of shape (n, d), dtype=int8
        Quantized key values.
    scales   : np.ndarray of shape (n,), dtype=float32
        The per-row scale factors used for reconstruction.
    """
    if keys_fp16.dtype != np.float16:
        raise TypeError("keys_fp16 must be float16")
    if keys_fp16.ndim != 2:
        raise ValueError("keys_fp16 must be a 2-D array")

    n, d = keys_fp16.shape

    abs_max = np.zeros(n, dtype=np.float32)
    for i in range(n):
        row_max = 0.0
        for j in range(d):
            val = keys_fp16[i, j]
            if val < 0.0:
                val = -val
            if val > row_max:
                row_max = float(val)
        abs_max[i] = row_max

    eps = 1e-12
    scales = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if abs_max[i] > eps:
            scales[i] = abs_max[i] / 127.0
        else:
            scales[i] = 1.0

    clipped = np.zeros((n, d), dtype=np.int8)
    for i in range(n):
        scale = scales[i]
        for j in range(d):
            val = float(keys_fp16[i, j]) / scale
            rounded = round(val)
            if rounded < -127:
                rounded = -127
            elif rounded > 127:
                rounded = 127
            clipped[i, j] = int(rounded)

    return clipped, scales
