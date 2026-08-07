import numpy as np


def quantize_blockwise(tensor, block_size):
    flat = tensor.astype(np.float32).flatten()
    padded_len = int(np.ceil(len(flat) / block_size) * block_size)
    padded = np.zeros(padded_len, dtype=np.float32)
    padded[:len(flat)] = flat
    blocks = padded.reshape(-1, block_size)
    absmax = np.max(np.abs(blocks), axis=1, keepdims=True)
    absmax[absmax == 0.0] = 1.0
    scaled = np.round(blocks / absmax * 127.0)
    quantized = np.clip(scaled, -128, 127).astype(np.int8)
    return {"quantized": quantized, "absmax": absmax.flatten(), "original_shape": tensor.shape}
