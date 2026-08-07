import numpy as np

def quantize_weights(weights: np.ndarray, bits: int = 4) -> dict:
    scale = np.max(np.abs(weights)) / (2**(bits-1) - 1)
    quantized = np.clip(np.round(weights / scale), -(2**(bits-1)), 2**(bits-1)-1).astype(np.int8)
    return {"quantized": quantized, "scale": scale, "bits": bits}
