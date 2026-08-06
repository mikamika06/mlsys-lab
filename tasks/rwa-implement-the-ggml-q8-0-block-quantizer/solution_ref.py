import numpy as np

def q8_0_quantize(block: np.ndarray):
    """
    Quantise a single block of 32 floats using the ggml Q8_0 format.
    Returns an int8 array of quantised coefficients and a float16 scale.
    """
    x = np.asarray(block, dtype=np.float64)
    amax = 0.0
    for i in range(x.shape[0]):
        v = x[i]
        if v < 0.0:
            v = -v
        if v > amax:
            amax = v

    if amax == 0:
        scale = 1.0
    else:
        scale = amax / 127.0

    q = np.empty(x.shape[0], dtype=np.int8)
    for i in range(x.shape[0]):
        val = x[i] / scale
        r = round(val)
        if r < -127:
            r = -127
        elif r > 127:
            r = 127
        q[i] = r

    return q, np.float16(scale)
