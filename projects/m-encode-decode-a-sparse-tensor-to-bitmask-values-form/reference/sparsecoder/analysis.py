import numpy as np


def breakeven_sparsity(tensor_shape, dtype_bytes=2, bitmask_element_bits=1):
    total_elements = int(np.prod(tensor_shape))
    dense_bytes = total_elements * dtype_bytes

    best_s = 0.0
    min_diff = float('inf')

    for s_int in range(1, 100):
        s = s_int / 100.0
        num_nonzeros = int(total_elements * (1.0 - s))
        sparse_bytes = (total_elements * bitmask_element_bits / 8.0) + (num_nonzeros * dtype_bytes)
        diff = abs(dense_bytes - sparse_bytes)
        if diff < min_diff:
            min_diff = diff
            best_s = s

    return round(float(best_s), 2)


def measure_byte_savings(tensor_shape, sparsities, dtype_bytes=2, block_size=8):
    total_elements = int(np.prod(tensor_shape))
    dense_bytes = total_elements * dtype_bytes
    results = {}

    for s in sparsities:
        num_nonzeros = int(round(total_elements * (1.0 - s)))
        num_blocks = (total_elements + block_size - 1) // block_size
        bitmask_bytes = num_blocks * (block_size / 8.0)
        values_bytes = num_nonzeros * dtype_bytes
        sparse_bytes = bitmask_bytes + values_bytes
        savings_pct = (1.0 - (sparse_bytes / dense_bytes)) * 100.0
        results[s] = {
            "dense_bytes": dense_bytes,
            "sparse_bytes": int(sparse_bytes),
            "savings_percent": float(round(savings_pct, 2))
        }

    return results
