import numpy as np


def derive_l2_chunk_size(tensor_shape, element_size_bytes, l2_cache_capacity):
    bytes_per_row = tensor_shape[-1] * element_size_bytes
    if bytes_per_row >= l2_cache_capacity:
        return max(1, l2_cache_capacity // element_size_bytes)
    target_bytes = l2_cache_capacity // 2
    chunk_rows = max(1, target_bytes // bytes_per_row)
    return int(min(chunk_rows, tensor_shape[0]))
