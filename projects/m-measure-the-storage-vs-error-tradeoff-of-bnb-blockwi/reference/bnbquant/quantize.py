import numpy as np


def blockwise_quantize(tensor, block_size, bits=8):
    flat = tensor.flatten()
    n = len(flat)
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode='constant')
    blocks = flat.reshape(-1, block_size)
    max_vals = np.max(np.abs(blocks), axis=1, keepdims=True)
    max_vals = np.maximum(max_vals, 1e-12)
    q_max = float((1 << (bits - 1)) - 1)
    scales = max_vals / q_max
    quantized = np.round(blocks / scales).astype(np.int8)
    return quantized, scales.flatten(), n


def blockwise_dequantize(quantized, scales, block_size, original_len, bits=8):
    scales_reshaped = scales.reshape(-1, 1)
    dequant_blocks = quantized.astype(np.float32) * scales_reshaped
    flat_dequant = dequant_blocks.flatten()
    return flat_dequant[:original_len]
