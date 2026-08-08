import numpy as np


def compute_tensor_bytes(num_elements: int, n_bits: int, codebook_size: int, vector_dim: int = 1, element_size_bytes: int = 4) -> int:
    codebook_bytes = codebook_size * vector_dim * element_size_bytes
    num_vectors = num_elements // vector_dim
    index_bits = num_vectors * n_bits
    index_bytes = (index_bits + 7) // 8
    return codebook_bytes + index_bytes
