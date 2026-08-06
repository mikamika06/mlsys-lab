import numpy as np


def naive_int8_quantize(x: np.ndarray):
    scale = np.max(np.abs(x)) / 127.0
    if scale == 0:
        return np.zeros_like(x, dtype=np.int8), 1.0
    quantized = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    dequantized = quantized.astype(np.float32) * scale
    return dequantized, scale


def decomposed_int8_quantize(x: np.ndarray, threshold: float):
    outliers = np.abs(x) > threshold
    fp16_part = np.where(outliers, x, 0.0)
    int8_input = np.where(outliers, 0.0, x)
    scale = np.max(np.abs(int8_input)) / 127.0
    if scale == 0:
        quantized = np.zeros_like(x, dtype=np.int8)
        dequantized_int8 = np.zeros_like(x, dtype=np.float32)
    else:
        quantized = np.clip(np.round(int8_input / scale), -127, 127).astype(np.int8)
        dequantized_int8 = quantized.astype(np.float32) * scale
    dequantized = dequantized_int8 + fp16_part
    return dequantized, scale
