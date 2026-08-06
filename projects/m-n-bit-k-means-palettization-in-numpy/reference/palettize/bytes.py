import numpy as np


def exact_tensor_bytes(num_elements: int, n_bits: int, vector_length: int = 1, centroid_dtype_bytes: int = 4) -> int:
    k = 2 ** n_bits
    centroid_bytes = k * vector_length * centroid_dtype_bytes
    num_blocks = (num_elements + vector_length - 1) // vector_length
    index_bits_total = num_blocks * n_bits
    index_bytes = (index_bits_total + 7) // 8
    return centroid_bytes + index_bytes
