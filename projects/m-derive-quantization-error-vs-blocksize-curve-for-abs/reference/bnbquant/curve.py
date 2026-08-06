import numpy as np


def compute_quantization_error_curve(tensor, block_sizes):
    flat = tensor.astype(np.float64).flatten()
    n = flat.size
    errors = []
    for bs in block_sizes:
        pad_len = (bs - (n % bs)) % bs
        padded = np.pad(flat, (0, pad_len), mode="constant")
        reshaped = padded.reshape(-1, bs)
        absmax = np.max(np.abs(reshaped), axis=1, keepdims=True)
        absmax = np.maximum(absmax, 1e-12)
        scale = 127.0 / absmax
        quantized = np.round(reshaped * scale)
        quantized = np.clip(quantized, -128, 127)
        dequantized = quantized / scale
        mse = np.mean((reshaped - dequantized) ** 2)
        errors.append(float(mse))
    return errors
