import numpy as np

def q8_0_quantize(block: np.ndarray):
    """
    Quantise a single block of 32 floats using the ggml Q8_0 format.
    Returns an int8 array of quantised coefficients and a float16 scale.
    """
    x = np.asarray(block, dtype=np.float64)
    amax = np.max(np.abs(x))
    if amax == 0:
        scale = 1.0
    else:
        scale = amax / 127.0

    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, np.float16(scale)
