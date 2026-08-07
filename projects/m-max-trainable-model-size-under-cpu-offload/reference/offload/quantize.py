import numpy as np

def int8_block_quantize(tensor, block_size=64):
    flat = np.asarray(tensor, dtype=np.float32).flatten()
    orig_len = len(flat)
    remainder = orig_len % block_size
    pad_len = (block_size - remainder) if remainder != 0 else 0
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode='constant', constant_values=0)
    blocks = flat.reshape(-1, block_size)
    max_vals = np.max(np.abs(blocks), axis=1, keepdims=True)
    max_vals = np.maximum(max_vals, 1e-8)
    scales = (max_vals / 127.0).astype(np.float32)
    q_blocks = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)
    return q_blocks, scales.squeeze(-1)

def int8_block_dequantize(q_blocks, scales, original_shape):
    scales_col = scales.reshape(-1, 1).astype(np.float32)
    dequant = q_blocks.astype(np.float32) * scales_col
    flat = dequant.flatten()
    total_elements = int(np.prod(original_shape))
    return flat[:total_elements].reshape(original_shape)
