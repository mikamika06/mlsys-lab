import numpy as np

def quantize_blockwise(tensor, block_size):
    """Quantize tensor blockwise into int8 absmax."""
    original_shape = tensor.shape
    flat = tensor.flatten()
    n = len(flat)
    quantized = np.zeros(n, dtype=np.int8)
    scales = []
    for i in range(0, n, block_size):
        block = flat[i:i+block_size]
        if len(block) == 0:
            scales.append(1.0)
            continue
        max_val = np.max(np.abs(block))
        scale = max_val / 127.0 if max_val > 0 else 1.0
        scales.append(scale)
        q = np.round(block / scale).clip(-127, 127).astype(np.int8)
        quantized[i:i+len(block)] = q
    return quantized, np.array(scales, dtype=np.float32), original_shape

def dequantize_blockwise(quantized, scales, block_size, original_shape):
    """Dequantize blockwise int8 tensor."""
    flat_q = quantized.flatten()
    flat_out = np.zeros(len(flat_q), dtype=np.float32)
    for idx, scale in enumerate(scales):
        start = idx * block_size
        end = min(start + block_size, len(flat_q))
        if start >= len(flat_q):
            break
        flat_out[start:end] = flat_q[start:end].astype(np.float32) * scale
    return flat_out.reshape(original_shape)
