import numpy as np

TENSORS = [
    np.array([0.1, -0.5, 0.3, 0.8, -0.2, 0.4, -0.1, 0.6], dtype=np.float32),
    np.linspace(-2.0, 2.0, 64, dtype=np.float32),
]

BLOCK_SIZES = [16, 32]
BITS_LIST = [8]

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

def compute_storage_bytes(num_elements, bits, block_size):
    num_blocks = (num_elements + block_size - 1) // block_size
    data_bits = num_elements * bits
    scale_bits = num_blocks * 32
    return (data_bits + scale_bits + 7) // 8

def compute_mse(original, dequantized):
    return float(np.mean((original - dequantized) ** 2))

def measure_tradeoff(tensor, block_sizes, bits_list):
    results = []
    for bs in block_sizes:
        for b in bits_list:
            q, s, orig_len = blockwise_quantize(tensor, bs, b)
            dq = blockwise_dequantize(q, s, bs, orig_len, b)
            mse = compute_mse(tensor, dq)
            storage = compute_storage_bytes(tensor.size, b, bs)
            results.append({"block_size": bs, "bits": b, "storage_bytes": storage, "mse": mse})
    return results
