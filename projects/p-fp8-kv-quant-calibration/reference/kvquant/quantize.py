import numpy as np

def quantize_fp8(tensor: np.ndarray, scale: float) -> np.ndarray:
    clipped = np.clip(tensor / scale, -1.0, 1.0)
    quantized = np.round(clipped * 127.0).astype(np.int8)
    return quantized

def dequantize_fp8(tensor: np.ndarray, scale: float) -> np.ndarray:
    return tensor.astype(np.float32) * (scale / 127.0)
