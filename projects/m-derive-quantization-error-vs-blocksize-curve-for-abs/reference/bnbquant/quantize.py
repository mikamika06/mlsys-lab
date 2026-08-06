import numpy as np


def blockwise_quantize_dequantize(tensor, block_size):
    flat = tensor.astype(np.float32).flatten()
    n = flat.size
    pad_len = (block_size - (n % block_size)) % block_size
    padded = np.pad(flat, (0, pad_len), mode="constant")
    reshaped = padded.reshape(-1, block_size)

    absmax = np.max(np.abs(reshaped), axis=1, keepdims=True)
    absmax_safe = np.maximum(absmax, 1e-12)
    scale = 127.0 / absmax_safe

    quantized = np.round(reshaped * scale)
    quantized = np.clip(quantized, -127, 127)

    dequantized = quantized / scale
    dequantized_flat = dequantized.flatten()[:n]
    return dequantized_flat.reshape(tensor.shape)
