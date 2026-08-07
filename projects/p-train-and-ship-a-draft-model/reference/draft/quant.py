import numpy as np

def quantize_weights(weights):
    w = np.array(weights, dtype=np.float32)
    scale = np.max(np.abs(w)) / 127.0
    if scale == 0:
        scale = 1.0
    quantized = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return {"quantized": quantized, "scale": float(scale)}
