import numpy as np

def generate_cases():
    np.random.seed(42)
    cases = []
    for _ in range(5):
        M = np.random.randn(32, 64).astype(np.float32)
        M[0, 5] = 50.0
        M[10, 20] = -45.0
        cases.append(M)
    return cases

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

def mixed_precision_matmul(A, B, threshold=6.0):
    qB, scalesB, outB = quantize_int8(B, threshold=threshold)
    B_approx = dequantize_int8(qB, scalesB, outB)
    return A @ B_approx
