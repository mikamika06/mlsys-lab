import numpy as np
from fp8util.quant import encode_e4m3, decode_e4m3


def compute_scale(x, max_val=448.0):
    x = np.asarray(x, dtype=np.float32)
    amax = np.max(np.abs(x))
    if amax == 0.0:
        return 1.0
    scale = max_val / amax
    return float(scale)


def quantize_per_tensor(x):
    x = np.asarray(x, dtype=np.float32)
    scale = compute_scale(x)
    scaled = x * scale
    encoded = encode_e4m3(scaled)
    decoded = decode_e4m3(encoded)
    dequantized = decoded / scale
    return {
        "scale": scale,
        "encoded": encoded,
        "dequantized": dequantized
    }
