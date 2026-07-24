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
    # Ensure input is float16 and has two dimensions
    if keys_fp16.dtype != np.float16:
        raise TypeError("keys_fp16 must be float16")
    if keys_fp16.ndim != 2:
        raise ValueError("keys_fp16 must be a 2-D array")

    n, d = keys_fp16.shape

    # Compute per-row max absolute value
    abs_max = np.max(np.abs(keys_fp16), axis=1)          # shape (n,)
    # Avoid division by zero: if a row is all zeros, set scale to 1.0
    eps = 1e-12
    scales = np.where(abs_max > eps, abs_max / 127.0, 1.0).astype(np.float32)  # shape (n,)

    # Quantize keys: divide by scale and round to nearest int
    scaled = keys_fp16.astype(np.float32) / scales[:, None]   # broadcast over columns
    rounded = np.rint(scaled)
    clipped = np.clip(rounded, -127, 127).astype(np.int8)

    return clipped, scales
