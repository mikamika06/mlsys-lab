import numpy as np

def quantize_int8(matrix, threshold=6.0):
    abs_m = np.abs(matrix)
    outlier_mask = abs_m > threshold
    outliers = np.where(outlier_mask, matrix, 0.0)
    residual = np.where(outlier_mask, 0.0, matrix)
    scales = np.max(np.abs(residual), axis=0, keepdims=True)
    scales = np.maximum(scales, 1e-5)
    quantized = np.round(residual / scales * 127.0)
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    return quantized, scales, outliers

def dequantize_int8(quantized, scales, outliers):
    reconstructed = quantized.astype(np.float32) * (scales / 127.0) + outliers
    return reconstructed
