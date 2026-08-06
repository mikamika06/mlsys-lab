import numpy as np
from bnbquant.quantize import blockwise_quantize, blockwise_dequantize


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
