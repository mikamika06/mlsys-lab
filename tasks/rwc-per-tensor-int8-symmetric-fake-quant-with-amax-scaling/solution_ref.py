import numpy as np

def per_tensor_int8_symmetric_fake_quant(x):
    x = np.asarray(x)
    amax = np.max(np.abs(x))
    if amax == 0:
        scale = 1.0
    else:
        scale = amax / 127.0
    q = np.round(x / scale)
    codes = np.clip(q, -127, 127).astype(np.int8)
    dequantized = codes.astype(np.float64) * scale
    return codes, dequantized
