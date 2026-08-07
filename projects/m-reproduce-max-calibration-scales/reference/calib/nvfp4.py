import numpy as np


def nvfp4_round_trip(tensor, block_size=16):
    flat = tensor.flatten()
    pad_len = (block_size - (flat.size % block_size)) % block_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode="constant")
    blocks = flat.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True) / 6.0
    scales = np.maximum(scales, 1e-12)
    quantized = np.clip(np.round(blocks / scales), -6.0, 6.0)
    dequantized = quantized * scales
    return dequantized.flatten()[:tensor.size].reshape(tensor.shape)
